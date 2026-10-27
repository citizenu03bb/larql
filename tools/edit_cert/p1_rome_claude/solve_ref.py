"""Claude's ROME rank-1 solve for GPT-2, rewritten to the published code (docs/phase-b-p1-amendment-a3.md).

Reference: kmeng01/rome @ 0874014c (rome/compute_u.py, compute_v.py, repr_tools.py, rome_main.py), read as text from
~/Projects/larql-research/datasets/rome-reference-0874014c/ and never imported or run. Codex's port is not read.

  u      = C^-1 k_mean / ||C^-1 k_mean||, k_mean the mean c_proj input at the subject's last token over the templates
  delta  is added to the MLP output at the subject's last token of every row: 21 rewriting rows
           template.format(prompt) + decode(target[:-1]), and the KL row '{} is a'
  loss   = NLL (target tokens, mean per token, mean over rows) + 0.0625 KL(edited || native) at the KL row's
           subject token + 0.5 ||delta|| / ||target_init||^2, with target_init the row-0 output on the first forward
  loop   Adam lr 0.5, clamp ||delta|| <= 4 ||target_init|| after each update
  U      = (target_init + delta - c_proj(k_clean)) / (k_clean . u), k_clean from the prompt with no template
  W'     = W + u U^T (Conv1D), stored as low_rank V0 = u, U0 = U.

The one exception to the reference (A3 §3, Pat): exactly 20 updates and no early stop, where compute_v.py
L150-154 stops below a loss of 5e-2 and before the last backward (at most 19 updates). The model runs in fp16, so
the hooks cast delta to the activation dtype and the losses are taken in float32.
"""
from __future__ import annotations

import contextlib
from copy import deepcopy

import numpy as np

from p1_rome_claude.solve import PUBLISHED, Request, module

KL_PROMPTS = ['{} is a']
STEPS = 20   # A3 §3: 20 effective updates


def lookup_index(tok, template: str, subject: str) -> int:
    """repr_tools.get_words_idxs_in_templates, subtoken 'last': prefix, word and suffix tokenized apart."""
    assert template.count('{}') == 1
    i = template.index('{}')
    prefix, word = template[:i], deepcopy(subject)
    if prefix:
        assert prefix[-1] == ' '
        prefix, word = prefix[:-1], f' {word.strip()}'
    return len(tok.encode(prefix)) + len(tok.encode(word)) - 1


@contextlib.contextmanager
def capture(mod, store: dict, positions: list[int]):
    """Keep the module's input and output at positions[i] of batch row i (rome's Trace, track='both')."""
    def post(m, args, out):
        store['in'] = [args[0][i, p].detach().double().cpu().numpy() for i, p in enumerate(positions)]
        store['out'] = [out[i, p].detach().double().cpu().numpy() for i, p in enumerate(positions)]
    h = mod.register_forward_hook(post)
    try:
        yield
    finally:
        h.remove()


@contextlib.contextmanager
def add_delta(mlp, delta, positions: list[int], record: dict):
    """compute_v.edit_output_fn: record target_init from row 0 once, then add delta at every row's lookup index."""
    def post(m, args, out):
        if 'target_init' not in record:
            record['target_init'] = out[0, positions[0]].detach().float().clone()
        out = out.clone()
        for i, p in enumerate(positions):
            out[i, p] = out[i, p] + delta.to(out.dtype)
        return out
    h = mlp.register_forward_hook(post)
    try:
        yield
    finally:
        h.remove()


def batch(tok, texts: list[str], dev):
    """tok(texts, padding=True) with GPT-2's right padding by EOS."""
    import torch
    ids = [tok.encode(t) for t in texts]
    width = max(len(r) for r in ids)
    input_ids = torch.tensor([r + [tok.eos_token_id] * (width - len(r)) for r in ids], device=dev)
    mask = torch.tensor([[1] * len(r) + [0] * (width - len(r)) for r in ids], device=dev)
    return input_ids, mask, [len(r) for r in ids]


def solve(model, tok, req: Request, templates: list[str], C: np.ndarray | None, hp=PUBLISHED, seed: int = 20261027,
          steps: int = STEPS):
    import torch
    torch.manual_seed(seed)
    dev = next(model.parameters()).device
    L = hp['layer']
    c_proj, mlp = module(model, L), model.transformer.h[L].mlp
    model.requires_grad_(False)

    # compute_u: k_mean over the templates, one forward each (A1's count), then u = C^-1 k_mean, unit norm
    keys = []
    for t in templates:
        tmpl = t.format(req.prompt)
        store: dict = {}
        with capture(c_proj, store, [lookup_index(tok, tmpl, req.subject)]), torch.inference_mode():
            model(torch.tensor([tok.encode(tmpl.format(req.subject))], device=dev))
        keys.append(store['in'][0])
    K = np.stack(keys)
    k_mean = K.mean(axis=0)
    q = k_mean if C is None else np.linalg.solve(C.astype(np.float64), k_mean)
    u = q / np.linalg.norm(q)

    # compute_v: 21 rewriting rows and 1 KL row in one right-padded batch
    target_ids = tok.encode(req.target)
    rewriting = [t.format(req.prompt) + tok.decode(target_ids[:-1]) for t in templates]
    prompts = rewriting + KL_PROMPTS
    input_ids, mask, lens = batch(tok, [p.format(req.subject) for p in prompts], dev)
    positions = [lookup_index(tok, p, req.subject) for p in prompts]
    n = len(rewriting)
    targets = torch.full((n, input_ids.shape[1]), -100, device=dev)
    for i in range(n):
        targets[i, lens[i] - len(target_ids):lens[i]] = torch.tensor(target_ids, device=dev)
    tmask = (targets != -100).float()

    delta = torch.zeros(model.config.n_embd, requires_grad=True, device=dev)
    opt = torch.optim.Adam([delta], lr=hp['lr'])
    record: dict = {}
    kl_init = None
    trace = []
    for it in range(steps):
        opt.zero_grad()
        with add_delta(mlp, delta, positions, record):
            logits = model(input_ids, attention_mask=mask).logits.float()
        kl_lp = torch.log_softmax(torch.stack([logits[n + j, positions[n + j]] for j in range(len(KL_PROMPTS))]), -1)
        if kl_init is None:
            kl_init = kl_lp.detach().clone()
        lp = torch.log_softmax(logits[:n], -1)
        gathered = torch.gather(lp, 2, torch.where(targets != -100, targets, 0).unsqueeze(2)).squeeze(2)
        nll = (-(gathered * tmask).sum(1) / len(target_ids)).mean()
        kl = hp['kl'] * torch.nn.functional.kl_div(kl_init, kl_lp, log_target=True, reduction='batchmean')
        target_init = record['target_init']
        decay = hp['decay'] * (torch.norm(delta) / torch.norm(target_init) ** 2)
        loss = nll + kl + decay
        # A3 §3: no early stop and no break before the last backward
        loss.backward()
        opt.step()
        max_norm = hp['clamp'] * target_init.norm()
        if delta.norm() > max_norm:
            with torch.no_grad():
                delta[...] = delta * max_norm / delta.norm()
        trace.append(dict(step=it, loss=loss.item(), nll=nll.item(), kl=kl.item(), decay=decay.item(),
                          delta_norm=float(delta.norm().item())))
        assert np.isfinite(trace[-1]['loss']), 'nonfinite loss'

    # the constraint at the request's own prompt (get_module_input_output_at_word)
    store = {}
    with capture(c_proj, store, [lookup_index(tok, req.prompt, req.subject)]), torch.inference_mode():
        model(torch.tensor([tok.encode(req.prompt.format(req.subject))], device=dev))
    k_clean, cur_output = store['in'][0], store['out'][0]
    t_init = record['target_init'].double().cpu().numpy()
    d = delta.detach().double().cpu().numpy()
    target = t_init + d
    right = (target - cur_output) / float(k_clean @ u)
    return dict(U=right[:, None].astype(np.float32), V=u[:, None].astype(np.float32),
                k_mean=k_mean.astype(np.float32), k_clean=k_clean.astype(np.float32),
                context_keys=K.astype(np.float32), delta=d.astype(np.float32),
                target_init=t_init.astype(np.float32), cur_output=cur_output.astype(np.float32),
                positions=positions, trace=trace, division_factor=float(k_clean @ u),
                constraint_error=float(np.abs(cur_output + right * float(u @ k_clean) - target).max()),
                hp=dict(hp, steps=steps), request=dict(prompt=req.prompt, subject=req.subject, target=req.target),
                reference='kmeng01/rome@0874014c', exception='20 updates, no early stop (A3 §3)')
