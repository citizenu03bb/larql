"""Rank-one ROME at the *raw down_proj output*, before norm/residual/PLE.

Adapted from the constrained update in Meng et al. (2022), not a port of the
authors' runner. A single shared absolute output vector is optimized over
fixed context prefixes and a native essence KL; parameters never receive
gradients. The algebra constrains the mean key, not every context key.
There is no model loader or automatic execution entry point in this module.
"""
from __future__ import annotations

import contextlib
from dataclasses import asdict, dataclass, replace
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch
import torch.nn.functional as F


DEADLINES = {"E2B": 13, "E4B": 22}
MODULE = "model.language_model.layers.{layer}.mlp.down_proj"
CONTEXT_PREFIXES = ("", "Previously, ", "For reference, ", "In a short encyclopedia entry, ")


@dataclass(frozen=True)
class RomeConfig:
    steps: int = 25
    learning_rate: float = 0.05
    kl_weight: float = 0.0625
    weight_decay: float = 0.001
    clamp_norm_factor: float = 4.0
    seed: int = 20261026
    max_sequence_tokens: int = 256

    def __post_init__(self):
        vals = (self.learning_rate, self.kl_weight, self.weight_decay, self.clamp_norm_factor)
        if not all(np.isfinite(v) for v in vals) or min(vals) < 0 or self.learning_rate == 0:
            raise ValueError("Invalid optimizer parameters")
        if not 1 <= self.steps <= 100 or self.clamp_norm_factor == 0 or self.max_sequence_tokens < 2:
            raise ValueError("Invalid step/sequence/clamp limit")


@dataclass(frozen=True)
class RomeRequest:
    prompt: str
    subject: str
    target: str
    essence_template: str = "{subject} is a"


def request_from_spec(spec):
    """Consume only the explicitly seen S1 and the declared subject/target."""
    questions = [q for q in spec["questions"] if q["id"] == "S1"]
    if len(questions) != 1 or not questions[0].get("seen") or questions[0]["control"] != "efficacy":
        raise ValueError("Exactly one seen efficacy S1 is required")
    prompt = questions[0]["text"]
    subjects = sorted(spec["subject"], key=lambda s: (-len(s), s))
    subject = next((s for s in subjects if prompt.count(s) == 1), None)
    if subject is None:
        raise ValueError("No unique declared subject in S1")
    return RomeRequest(prompt, subject, spec["new"]["accept"][0])


@dataclass
class Budget:
    max_forwards: int = 400
    max_gradient_steps: int = 100
    seconds: float = 1740.0
    forwards: int = 0
    gradient_steps: int = 0

    def __post_init__(self):
        if not 0 < self.max_forwards <= 400 or not 0 < self.max_gradient_steps <= 100 or not 0 < self.seconds <= 1800:
            raise ValueError("Budget exceeds Phase B service limits")
        self.deadline = time.monotonic() + self.seconds

    def take(self, gradient=False):
        if time.monotonic() >= self.deadline:
            raise TimeoutError("Service deadline exhausted")
        field, cap = ("gradient_steps", self.max_gradient_steps) if gradient else ("forwards", self.max_forwards)
        if getattr(self, field) >= cap:
            raise RuntimeError(f"Service {field} cap exhausted")
        setattr(self, field, getattr(self, field) + 1)


def finite(value, name):
    if not np.isfinite(value).all():
        raise ValueError(f"Nonfinite {name}")


def rank_one_update(k_star, v_star, w_k_star, covariance=None):
    """Return float32 U[out,1], V[in,1], with V=C^-1 k/(k^T C^-1 k).

    W is represented by its saved action W k*. No explicit inverse. Float64
    SPD factorization/triangular solve; final adapter and certificate float32.
    """
    k, target, base = [np.asarray(x, dtype=np.float32) for x in (k_star, v_star, w_k_star)]
    if k.ndim != 1 or target.ndim != 1 or target.shape != base.shape:
        raise ValueError("Invalid key/value shapes")
    for name, value in (("key", k), ("target", target), ("Wk", base)):
        finite(value, name)
    key = torch.from_numpy(k.astype(np.float64))
    if covariance is None:
        direction = key
    else:
        c = np.asarray(covariance)
        if c.shape != (len(k), len(k)):
            raise ValueError("Covariance shape differs from key width")
        finite(c, "covariance")
        if not np.allclose(c, c.T, rtol=0, atol=1e-7):
            raise ValueError("Covariance is not symmetric")
        # Refuse non-SPD C; never silently change its regularization here.
        chol = torch.linalg.cholesky(torch.as_tensor(c, dtype=torch.float64, device="cpu"))
        direction = torch.cholesky_solve(key[:, None], chol)[:, 0]
    denominator = torch.dot(key, direction).item()
    if not np.isfinite(denominator) or denominator <= np.finfo(np.float64).tiny:
        raise ValueError("Degenerate key or covariance direction")
    u = (target - base)[:, None]
    v = (direction / denominator).numpy().astype(np.float32)[:, None]
    for name, value in (("U", u), ("V", v)):
        finite(value, name)
    if not np.allclose(base + u @ (v.T @ k), target, rtol=2e-5, atol=2e-5):
        raise ValueError("Float32 rank-one constraint failed")
    return u, v


def site(model, layers, layer, label, synthetic=False, kv_deadline=None):
    if label not in DEADLINES or kv_deadline not in (None, DEADLINES[label]):
        raise ValueError("Unknown instrument or inconsistent K/V deadline")
    if not isinstance(layer, int) or not 0 <= layer <= DEADLINES[label] or layer >= len(layers):
        raise ValueError(f"Layer must be <= {DEADLINES[label]} on {label}")
    path = MODULE.format(layer=layer)
    module = model.get_submodule(path)
    if module is not layers[layer].mlp.down_proj:
        raise ValueError("Layer/module identity mismatch")
    if not synthetic and label == "E2B" and (module.out_features, module.in_features) != (1536, 6144):
        raise ValueError("E2B down_proj header/loaded shape mismatch")
    return path, module


@contextlib.contextmanager
def frozen_model(model):
    """Restore parameter flags, training states and RNG/determinism on errors too."""
    params = [(p, p.requires_grad) for p in model.parameters()]
    training = [(m, m.training) for m in model.modules()]
    deterministic = torch.are_deterministic_algorithms_enabled()
    warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    tf32 = torch.backends.cuda.matmul.allow_tf32
    cudnn_tf32 = torch.backends.cudnn.allow_tf32
    devices = sorted({p.device.index for p, _ in params if p.device.type == "cuda"})
    try:
        with torch.random.fork_rng(devices=devices):
            model.eval().requires_grad_(False)
            torch.use_deterministic_algorithms(True)
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
            yield
    finally:
        for p, flag in params:
            p.requires_grad_(flag)
        for m, flag in training:
            m.training = flag
        torch.use_deterministic_algorithms(deterministic, warn_only=warn_only)
        torch.backends.cuda.matmul.allow_tf32 = tf32
        torch.backends.cudnn.allow_tf32 = cudnn_tf32


@contextlib.contextmanager
def output_value(module, position, value):
    """Replace exactly one row at this module; leave all other rows untouched."""
    calls = []

    def hook(mod, args, output):
        if calls or output.ndim != 3 or output.shape[0] != 1 or not 0 <= position < output.shape[1]:
            raise ValueError("Unexpected projection invocation/position")
        calls.append(True)
        patched = output.clone()
        patched[0, position] = value.to(device=output.device, dtype=output.dtype)
        return patched

    handle = module.register_forward_hook(hook)
    try:
        yield
        if len(calls) != 1:
            raise ValueError("Projection hook did not run")
    finally:
        handle.remove()


def weight_action(module, key):
    """Float32 action of effective W. Dequantize NF4 only after real-run approval.

    This allocates one projection, never a dense whole model and never writes
    quantized weights. Actual fp16 projection arithmetic may differ by rounding.
    """
    weight = module.weight
    if getattr(weight, "quant_state", None) is not None:
        from bitsandbytes.functional import dequantize_4bit
        matrix = dequantize_4bit(weight.data, weight.quant_state)
    else:
        matrix = weight.detach()
    if matrix.shape != (module.out_features, module.in_features):
        raise ValueError("Projection weight layout mismatch")
    with torch.no_grad():
        result = matrix.float() @ torch.as_tensor(key, dtype=torch.float32, device=matrix.device)
    return result.detach().float().cpu().numpy().copy()


@dataclass
class Example:
    text: str
    rendered: str
    ids: list[int]
    position: int
    target_ids: list[int]


def prepare(tokenizer, text, subject, target=None, max_tokens=256):
    if not subject or text.count(subject) != 1:
        raise ValueError("Subject must occur exactly once in each user text")
    rendered = tokenizer.apply_chat_template([{"role": "user", "content": text}],
        tokenize=False, add_generation_prompt=True, enable_thinking=False)
    if not isinstance(rendered, str) or rendered.count(subject) != 1:
        raise ValueError("Ambiguous subject in rendered chat")
    encoded = tokenizer(rendered, add_special_tokens=False, return_offsets_mapping=True)
    ids = list(encoded["input_ids"])
    start = rendered.index(subject)
    end = start + len(subject)
    overlaps = [i for i, (a, b) in enumerate(encoded["offset_mapping"]) if b > start and a < end]
    if not overlaps:
        raise ValueError("No token overlaps subject")
    targets = []
    if target is not None:
        full = tokenizer(rendered + target, add_special_tokens=False)["input_ids"]
        if list(full[:len(ids)]) != ids:
            raise ValueError("Target tokenization changes the chat prefix")
        targets = list(full[len(ids):])
        if not targets:
            raise ValueError("Empty target")
        ids += targets[:-1]
    if not ids or len(ids) > max_tokens:
        raise ValueError("Sequence length outside preregistered limit")
    return Example(text, rendered, ids, overlaps[-1], targets)


@dataclass
class RomeSolution:
    layer: int
    module_path: str
    k_star: np.ndarray
    v_star: np.ndarray
    w_k_star: np.ndarray
    covariance: np.ndarray | None
    keys: np.ndarray
    native_values: np.ndarray
    u: np.ndarray
    v: np.ndarray
    seen_phrasings: list[str]
    metadata: dict

    def with_covariance(self, covariance=None):
        """Model-free C ablation: reuse the SAME key/value optimization."""
        c = None if covariance is None else np.array(covariance, dtype=np.float32, copy=True)
        u, v = rank_one_update(self.k_star, self.v_star, self.w_k_star, c)
        metadata = dict(self.metadata,
                        reused_value_optimization=True,
                        source_forwards=self.metadata["forwards"],
                        source_gradient_steps=self.metadata["gradient_steps"],
                        source_value_witness_sha256={name: hashlib.sha256(a.tobytes()).hexdigest()
                                                   for name, a in (("k_star", self.k_star),
                                                                   ("v_star", self.v_star),
                                                                   ("w_k_star", self.w_k_star))},
                        forwards=0, gradient_steps=0, model_backward_evaluations=0,
                        regularizer_gradient_evaluations=0, history_origin="reused source optimization")
        return replace(self, covariance=c, u=u, v=v, metadata=metadata)

    def backend(self):
        from edit_cert.backends.lowrank import LowRank
        return LowRank([dict(module=self.module_path, layer=self.layer, scale=1.0)],
                       dict(U0=self.u, V0=self.v),
                       label="synthetic-ROME-rank1" if self.metadata["synthetic"] else "ROME-rank1",
                       seen_phrasings=self.seen_phrasings)

    def save(self, path):
        """Harness-compatible edit plus independent algebra/reproduction witnesses."""
        path = Path(path)
        backend = self.backend()
        backend.save(path)  # refuses an existing directory
        arrays = dict(k_star=self.k_star, v_star=self.v_star, w_k_star=self.w_k_star,
                      context_keys=self.keys, context_native_values=self.native_values,
                      C=np.eye(len(self.k_star), dtype=np.float32) if self.covariance is None else self.covariance)
        for name, value in arrays.items():
            np.save(path / f"{name}.npy", np.asarray(value, dtype=np.float32), allow_pickle=False)
        metadata = dict(self.metadata, covariance="identity" if self.covariance is None else "provided_SPD",
                        artifact_sha256={p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                                         for p in sorted(path.iterdir()) if p.is_file()})
        (path / "solver.json").write_text(json.dumps(metadata, indent=2, sort_keys=True, allow_nan=False) + "\n")


class RomeSolver:
    def __init__(self, model, layers, tokenizer, label="E2B", synthetic=False, kv_deadline=None):
        self.model, self.layers, self.tokenizer = model, layers, tokenizer
        self.label, self.synthetic, self.kv_deadline = label, synthetic, kv_deadline

    def forward(self, ids, keep, budget):
        budget.take()
        device = next(self.layers.parameters()).device
        output = self.model(input_ids=torch.tensor([ids], dtype=torch.long, device=device),
                            use_cache=False, logits_to_keep=keep).logits
        if not torch.isfinite(output).all():
            raise ValueError("Nonfinite logits")
        return output.float()

    def extract(self, module, examples, budget):
        keys, values = [], []
        with torch.no_grad():
            for example in examples:
                calls = []

                def capture(mod, args, output):
                    if calls:
                        raise ValueError("Projection ran twice in one extraction")
                    calls.append(True)
                    keys.append(args[0][0, example.position].detach().float().cpu().numpy().copy())
                    values.append(output[0, example.position].detach().float().cpu().numpy().copy())

                handle = module.register_forward_hook(capture)
                try:
                    self.forward(example.ids, 1, budget)
                    if len(calls) != 1:
                        raise ValueError("Extraction hook did not run")
                finally:
                    handle.remove()
        return np.stack(keys), np.stack(values)

    def solve(self, layer, request, context_prefixes, covariance=None, config=None, budget=None):
        if torch.is_inference_mode_enabled():
            raise ValueError("The solver needs autograd, not torch.inference_mode")
        config, budget = config or RomeConfig(), budget or Budget()
        path, module = site(self.model, self.layers, layer, self.label, self.synthetic, self.kv_deadline)
        if covariance is not None:
            covariance = np.array(covariance, dtype=np.float32, copy=True)
            if covariance.shape != (module.in_features, module.in_features):
                raise ValueError("Covariance shape differs from projection input")
            finite(covariance, "covariance")
        prefixes = list(context_prefixes)
        if not prefixes or len(prefixes) != len(set(prefixes)):
            raise ValueError("Context prefixes must be nonempty and distinct")
        examples = [prepare(self.tokenizer, p + request.prompt, request.subject, request.target,
                            config.max_sequence_tokens) for p in prefixes]
        essence_text = request.essence_template.format(subject=request.subject)
        essence = prepare(self.tokenizer, essence_text, request.subject, max_tokens=config.max_sequence_tokens)
        seen = list(dict.fromkeys([request.prompt, *prefixes, request.target, request.essence_template, essence_text,
                                  *[e.text for e in examples], *[e.rendered for e in examples],
                                  *[e.rendered + request.target for e in examples], essence.rendered,
                                  *[self.tokenizer.decode(e.ids) for e in examples]]))
        with frozen_model(self.model), torch.enable_grad():
            torch.manual_seed(config.seed)
            keys, values = self.extract(module, examples, budget)
            k_star = keys.mean(axis=0, dtype=np.float64).astype(np.float32)
            w_k_star = weight_action(module, k_star)
            finite(w_k_star, "effective Wk")
            device = module.weight.device
            initial = torch.as_tensor(w_k_star, device=device).clone()
            if initial.norm() == 0:
                raise ValueError("Cannot normalize a zero initial value")
            value = initial.clone().requires_grad_(True)
            with torch.no_grad():
                native_kl = self.forward(essence.ids, 1, budget)[0, -1].log_softmax(-1).detach()
            optimizer = torch.optim.Adam([value], lr=config.learning_rate)
            history = []
            for step in range(config.steps):
                budget.take(gradient=True)
                optimizer.zero_grad(set_to_none=True)
                gradient = torch.zeros_like(value)
                nll = 0.0
                for example in examples:
                    n = len(example.target_ids)
                    with output_value(module, example.position, value):
                        logits = self.forward(example.ids, n, budget)[0, -n:]
                    target = torch.tensor(example.target_ids, dtype=torch.long, device=logits.device)
                    loss = F.cross_entropy(logits, target) / len(examples)
                    gradient += torch.autograd.grad(loss, value)[0].detach()
                    nll += loss.detach().item()
                with output_value(module, essence.position, value):
                    log_probs = self.forward(essence.ids, 1, budget)[0, -1].log_softmax(-1)
                # KL(edited || native), matching the authors' compute_v direction.
                kl = config.kl_weight * (log_probs.exp() * (log_probs - native_kl)).sum()
                gradient += torch.autograd.grad(kl, value)[0].detach()
                decay = config.weight_decay * (value - initial).square().sum() / initial.square().sum()
                gradient += torch.autograd.grad(decay, value)[0].detach()
                if not torch.isfinite(gradient).all():
                    raise ValueError("Nonfinite value gradient")
                value.grad = gradient
                optimizer.step()
                with torch.no_grad():
                    delta = value - initial
                    limit = config.clamp_norm_factor * initial.norm()
                    if delta.norm() > limit:
                        value.copy_(initial + delta * limit / delta.norm())
                finite(value.detach().cpu().numpy(), "optimized value")
                history.append(dict(step=step, nll=nll, kl=kl.detach().item(), decay=decay.detach().item()))
            v_star = value.detach().float().cpu().numpy().copy()
        u, v = rank_one_update(k_star, v_star, w_k_star, covariance)
        metadata = dict(method="ROME-rank1-raw-down-proj-absolute-value", synthetic=self.synthetic,
                        solver_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                        instrument=self.label, layer=layer, module=path, config=asdict(config),
                        request=asdict(request), context_prefixes=prefixes, history=history,
                        forwards=budget.forwards, gradient_steps=budget.gradient_steps,
                        model_backward_evaluations=config.steps * (len(examples) + 1),
                        regularizer_gradient_evaluations=config.steps,
                        positions=[e.position for e in examples], essence_position=essence.position,
                        rewrite_input_ids=[e.ids for e in examples], essence_input_ids=essence.ids,
                        target_ids=[e.target_ids for e in examples],
                        precision="effective W and adapter float32; SPD solve float64; activation cast follows instrument")
        return RomeSolution(layer, path, k_star, v_star, w_k_star,
                            None if covariance is None else np.asarray(covariance, dtype=np.float32),
                            keys, values, u, v, seen, metadata)
