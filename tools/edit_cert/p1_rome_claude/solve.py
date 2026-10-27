"""Claude's independent ROME rank-1 solve for GPT-2 (P1-3; docs/phase-b-p1-positive-control-protocol.md).

Written from Meng et al. 2022 (§3 and appendix) and the published GPT-2 XL hyper-parameters, read as data. It does
not import or read Codex's solver, and it runs no downloaded code. Notation of the paper:

  k*  = mean over context templates x_j of the c_proj input at the subject's last token in x_j + p(s);
  v*  = argmin_z  mean_j NLL(o* | x_j + p(s); m := z)  +  kl * KL(P(. | p'; m := z) || P(. | p'))
                  +  decay * ||z - z0||^2 / ||z0||^2,      with ||z - z0|| <= clamp * ||z0||,
        where m is the c_proj output at the subject's last token, z0 = W k* + b, and p' = "{s} is a";
  W'  = W + U V^T  with  U = v* - (W k* + b),  V = C^-1 k* / (k*^T C^-1 k*),  so W' k* + b = v*.

Two points the paper leaves open, decided here and recorded in the result: the decay term is the squared relative
L2 distance, and z replaces the module output (it is not added to it). The covariance C is passed in; None means
the identity.
"""
from __future__ import annotations

import contextlib
from dataclasses import asdict, dataclass

import numpy as np

PUBLISHED = dict(layer=17, steps=20, lr=0.5, decay=0.5, clamp=4.0, kl=0.0625, n_short=10, len_short=5,
                 n_long=10, len_long=10)


@dataclass
class Request:
    prompt: str          # with '{}' where the subject goes, e.g. '{} is a company that was founded in the country of'
    subject: str
    target: str          # the new object, with its leading space, e.g. ' Italy'


def module(model, layer: int):
    return model.transformer.h[layer].mlp.c_proj


def subject_last(tok, text: str, subject: str) -> int:
    """Token index of the subject's last token in `text` (its last occurrence ending the subject surface)."""
    end = text.rindex(subject) + len(subject)
    head = tok.encode(text[:end])
    full = tok.encode(text)
    assert full[:len(head)] == head, f'subject boundary merges into the next token: {text!r}'
    return len(head) - 1


def context_templates(model, tok, seed: int, hp=PUBLISHED) -> list[str]:
    """'{}' plus n sampled continuations of EOS, each followed by '. {}' (top-k 5 sampling, fixed seed)."""
    import torch
    g = torch.Generator(device='cpu').manual_seed(seed)
    out = ['{}']
    dev = next(model.parameters()).device
    for n, length in ((hp['n_short'], hp['len_short']), (hp['n_long'], hp['len_long'])):
        for _ in range(n):
            ids = [tok.eos_token_id]
            with torch.inference_mode():
                for _ in range(length):
                    logits = model(torch.tensor([ids], device=dev)).logits[0, -1].float().cpu()
                    top = torch.topk(logits, 5)
                    p = torch.softmax(top.values, -1)
                    ids.append(int(top.indices[torch.multinomial(p, 1, generator=g)]))
            text = tok.decode(ids[1:]).replace('{', ' ').replace('}', ' ')
            out.append(text + '. {}')
    return out


@contextlib.contextmanager
def capture_input(mod, store: list, index_of):
    """Pre-hook: store the module input at the position `index_of(batch_row)`."""
    def pre(m, args):
        x = args[0]
        store.append(x[0, index_of()].detach().float().cpu().numpy())
    h = mod.register_forward_pre_hook(pre)
    try:
        yield
    finally:
        h.remove()


@contextlib.contextmanager
def substitute_output(mod, z, positions: list[int]):
    """Forward hook: replace the module output of batch row i at positions[i] by z (keeps the graph to z)."""
    def post(m, args, out):
        out = out.clone()
        for i, p in enumerate(positions):
            out[i, p] = z.to(out.dtype)
        return out
    h = mod.register_forward_hook(post)
    try:
        yield
    finally:
        h.remove()


def solve(model, tok, req: Request, templates: list[str], C: np.ndarray | None, hp=PUBLISHED, seed: int = 20261027):
    import torch
    torch.manual_seed(seed)
    dev = next(model.parameters()).device
    L = hp['layer']
    mod = module(model, L)
    model.requires_grad_(False)

    # k*: mean key over the context templates at the subject's last token
    texts = [t.format(req.prompt.format(req.subject)) for t in templates]
    keys = []
    for text in texts:
        idx = subject_last(tok, text, req.subject)
        store: list = []
        with capture_input(mod, store, lambda idx=idx: idx), torch.inference_mode():
            model(torch.tensor([tok.encode(text)], device=dev))
        keys.append(store[0])
    K = np.stack(keys).astype(np.float64)
    k_star = K.mean(axis=0)

    # z0 = W k* + b, through the served module itself
    with torch.inference_mode():
        z0 = mod(torch.tensor(k_star, dtype=next(mod.parameters()).dtype, device=dev)[None, None])[0, 0].float()
    z0_np = z0.cpu().numpy().astype(np.float64)

    # rewriting prompts: template + prompt(subject) + target tokens but the last; the loss reads every target token
    target_ids = tok.encode(req.target)
    rows, starts, subj = [], [], []
    for text in texts:
        p_ids = tok.encode(text)
        rows.append(p_ids + target_ids[:-1])
        starts.append(len(p_ids))
        subj.append(subject_last(tok, text, req.subject))
    width = max(len(r) for r in rows)
    pad = tok.eos_token_id
    batch = torch.tensor([r + [pad] * (width - len(r)) for r in rows], device=dev)
    attn = torch.tensor([[1] * len(r) + [0] * (width - len(r)) for r in rows], device=dev)
    kl_text = f'{req.subject} is a'
    kl_ids = torch.tensor([tok.encode(kl_text)], device=dev)
    kl_pos = subject_last(tok, kl_text, req.subject)
    with torch.inference_mode():
        base_kl = torch.log_softmax(model(kl_ids).logits[0, -1].float(), -1)

    z = torch.nn.Parameter(z0.clone())
    opt = torch.optim.Adam([z], lr=hp['lr'])
    trace = []
    for step in range(hp['steps']):
        opt.zero_grad()
        with substitute_output(mod, z, subj):
            logits = model(batch, attention_mask=attn).logits.float()
        lp = torch.log_softmax(logits, -1)
        nll = []
        for i, (s, r) in enumerate(zip(starts, rows)):
            pos = torch.arange(s - 1, s - 1 + len(target_ids), device=dev)
            nll.append(-lp[i, pos, torch.tensor(target_ids, device=dev)].mean())
        nll = torch.stack(nll).mean()
        with substitute_output(mod, z, [kl_pos]):
            cur = torch.log_softmax(model(kl_ids).logits[0, -1].float(), -1)
        kl = torch.sum(cur.exp() * (cur - base_kl))
        decay = ((z - z0) ** 2).sum() / (z0 ** 2).sum()
        loss = nll + hp['kl'] * kl + hp['decay'] * decay
        loss.backward()
        opt.step()
        with torch.no_grad():
            d = z - z0
            lim = hp['clamp'] * z0.norm()
            if d.norm() > lim:
                z.copy_(z0 + d * (lim / d.norm()))
        trace.append(dict(step=step, nll=nll.item(), kl=kl.item(), decay=decay.item(), loss=loss.item()))
        assert np.isfinite(trace[-1]['loss']), 'nonfinite loss'

    v_star = z.detach().cpu().numpy().astype(np.float64)
    q = k_star if C is None else np.linalg.solve(C.astype(np.float64), k_star)
    U = (v_star - z0_np)[:, None]
    V = (q / (k_star @ q))[:, None]
    image = z0_np + U[:, 0] * (V[:, 0] @ k_star)
    return dict(U=U.astype(np.float32), V=V.astype(np.float32), k_star=k_star.astype(np.float32),
                v_star=v_star.astype(np.float32), z0=z0_np.astype(np.float32), context_keys=K.astype(np.float32),
                constraint_error=float(np.abs(image - v_star).max()), trace=trace, templates=templates,
                hp=dict(hp), request=asdict(req), covariance='identity' if C is None else 'given',
                decisions=['decay = squared relative L2', 'z replaces the c_proj output at the subject last token'])
