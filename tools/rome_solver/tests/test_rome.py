"""CPU synthetic tests ONLY. No pretrained model/real instrument load or service."""
import hashlib
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np
import torch

TOOLS = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(TOOLS / "rome_solver"), str(TOOLS / "edit_cert")]
from edit_cert import backends
from rome_solver.solver import (Budget, RomeConfig, RomeRequest, RomeSolver, output_value,
                                prepare, rank_one_update, request_from_spec, site)
from rome_solver.covariance import SecondMoment, collect_model_covariances, fixed_windows

torch.set_num_threads(1)


class Tokenizer:
    eos_token_id = 1

    def apply_chat_template(self, messages, **kwargs):
        return "<user>" + messages[0]["content"] + "</user><model>"

    def encode(self, text, **kwargs):
        return [ord(c) + 3 for c in text]

    def __call__(self, text, return_offsets_mapping=False, **kwargs):
        result = dict(input_ids=self.encode(text))
        if return_offsets_mapping:
            result["offset_mapping"] = [(i, i + 1) for i in range(len(text))]
        return result

    def decode(self, ids):
        return "".join(chr(i - 3) for i in ids)


class Block(torch.nn.Module):
    def __init__(self, width=6, intermediate=10):
        super().__init__()
        self.pre_feedforward_layernorm = torch.nn.LayerNorm(width)
        self.mlp = torch.nn.Module()
        self.mlp.up_proj = torch.nn.Linear(width, intermediate, bias=False)
        self.mlp.down_proj = torch.nn.Linear(intermediate, width, bias=False)
        self.post_feedforward_layernorm = torch.nn.RMSNorm(width)
        self.per_layer_projection = torch.nn.Linear(2, width, bias=False)
        self.register_buffer("layer_scalar", torch.tensor(1.25))

    def forward(self, hidden, ple):
        # A causal mixer transports subject changes to later positions.
        n = torch.arange(1, hidden.shape[1] + 1, device=hidden.device)[None, :, None]
        hidden = hidden + 0.15 * hidden.cumsum(1) / n
        residual = hidden
        raw = self.mlp.down_proj(torch.tanh(self.mlp.up_proj(self.pre_feedforward_layernorm(hidden))))
        hidden = residual + self.post_feedforward_layernorm(raw)
        hidden = hidden + self.per_layer_projection(ple)
        return hidden * self.layer_scalar


class Model(torch.nn.Module):
    def __init__(self, depth=3):
        super().__init__()
        with torch.random.fork_rng():
            torch.manual_seed(17)
            self.embed = torch.nn.Embedding(256, 6)
            self.ple = torch.nn.Embedding(256, 2)
            self.model = torch.nn.Module()
            self.model.language_model = torch.nn.Module()
            self.model.language_model.layers = torch.nn.ModuleList(Block() for _ in range(depth))
            self.lm_head = torch.nn.Linear(6, 256, bias=False)

    @property
    def layers(self):
        return self.model.language_model.layers

    def forward(self, input_ids, use_cache=False, logits_to_keep=1):
        if use_cache:
            raise AssertionError("Synthetic tests require full prefixes")
        hidden = self.embed(input_ids)
        ple = self.ple(input_ids)
        for layer in self.layers:
            hidden = layer(hidden, ple)
        n = torch.arange(1, hidden.shape[1] + 1, device=hidden.device)[None, :, None]
        logits = self.lm_head(hidden.cumsum(1) / n)
        return SimpleNamespace(logits=logits[:, -logits_to_keep:])


def solver(model=None):
    model = model or Model()
    return RomeSolver(model, model.layers, Tokenizer(), synthetic=True)


REQUEST = RomeRequest("Name the city for AB CD.", "AB CD", "XY")
PREFIXES = ("", "For reference, ")
CONFIG = RomeConfig(steps=6, learning_rate=0.1)


class Algebra(unittest.TestCase):
    def test_exact_constraint_and_nullspace_identity_and_spd(self):
        rng = np.random.default_rng(20)
        w = rng.standard_normal((7, 9)).astype(np.float32)
        k = rng.standard_normal(9).astype(np.float32)
        target = rng.standard_normal(7).astype(np.float32)
        a = rng.standard_normal((9, 9))
        for c in (None, (a.T @ a + 2 * np.eye(9)).astype(np.float32)):
            u, v = rank_one_update(k, target, w @ k, c)
            self.assertEqual(u.dtype, np.float32)
            self.assertEqual(v.dtype, np.float32)
            self.assertEqual(u.shape, (7, 1))
            dense = w + u @ v.T
            np.testing.assert_allclose(dense @ k, target, atol=3e-6, rtol=3e-6)
            direction = k.astype(np.float64) if c is None else np.linalg.solve(c.astype(np.float64), k)
            # A spanning set of the orthogonal hyperplane, not a single sample.
            for basis in np.eye(len(k)):
                orth = basis - direction * np.dot(basis, direction) / np.dot(direction, direction)
                np.testing.assert_allclose(dense @ orth, w @ orth, atol=2e-6, rtol=2e-6)

    def test_real_e2b_adapter_dimensions_without_real_weights(self):
        rng = np.random.default_rng(3)
        k, base, target = [rng.standard_normal(n).astype(np.float32) for n in (6144, 1536, 1536)]
        u, v = rank_one_update(k, target, base)
        self.assertEqual(u.shape, (1536, 1))
        self.assertEqual(v.shape, (6144, 1))
        np.testing.assert_allclose(base + u @ (v.T @ k), target, rtol=2e-5, atol=2e-5)

    def test_refuses_degenerate_nonfinite_and_non_spd(self):
        with self.assertRaises(ValueError):
            rank_one_update(np.zeros(3), np.ones(2), np.zeros(2))
        with self.assertRaises(ValueError):
            rank_one_update(np.ones(3), np.array([np.nan, 1]), np.zeros(2))
        with self.assertRaises(ValueError):
            rank_one_update(np.ones(3), np.ones(2), np.zeros(2), np.ones((2, 2)))
        with self.assertRaises(RuntimeError):
            rank_one_update(np.ones(3), np.ones(2), np.zeros(2), -np.eye(3))
        bad = np.eye(3); bad[0, 1] = 1
        with self.assertRaises(ValueError):
            rank_one_update(np.ones(3), np.ones(2), np.zeros(2), bad)


class HooksAndSolver(unittest.TestCase):
    def test_request_only_uses_explicit_seen_s1(self):
        spec = dict(subject=["AB", "AB CD"], new=dict(accept=["XY"]), questions=[
            dict(id="S1", seen=True, control="efficacy", text=REQUEST.prompt),
            dict(id="E1", control="efficacy", text="Never train on this.")])
        self.assertEqual(request_from_spec(spec), REQUEST)
        spec["questions"][0]["seen"] = False
        with self.assertRaises(ValueError):
            request_from_spec(spec)

    def test_identity_witness_is_explicit_and_reloadable(self):
        source = solver().solve(1, REQUEST, PREFIXES, covariance=np.diag(np.arange(1, 11, dtype=np.float32)),
                                config=RomeConfig(steps=2))
        result = source.with_covariance()
        for field in ("k_star", "v_star", "w_k_star"):
            np.testing.assert_array_equal(getattr(result, field), getattr(source, field))
        self.assertEqual(result.metadata["forwards"], 0)
        self.assertEqual(result.metadata["gradient_steps"], 0)
        self.assertIsNotNone(source.covariance)
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / "edit"
            result.save(path)
            np.testing.assert_array_equal(np.load(path / "C.npy"), np.eye(10, dtype=np.float32))
            self.assertEqual(backends.load(path).seen_phrasings, result.seen_phrasings)

    def test_last_subject_token_in_chat_and_context(self):
        e = prepare(Tokenizer(), "Intro. " + REQUEST.prompt, REQUEST.subject, REQUEST.target)
        self.assertEqual(e.position, e.rendered.index(REQUEST.subject) + len(REQUEST.subject) - 1)
        self.assertEqual(e.ids[e.position], ord("D") + 3)
        self.assertEqual(e.target_ids, Tokenizer().encode("XY"))
        with self.assertRaises(ValueError):
            prepare(Tokenizer(), "AB CD and AB CD", "AB CD")

    def test_key_extraction_matches_raw_projection_at_selected_layer(self):
        s = solver(); model = s.model.eval()
        e = prepare(s.tokenizer, REQUEST.prompt, REQUEST.subject, REQUEST.target)
        module = model.layers[1].mlp.down_proj
        observed = {}
        def capture(mod, args, output):
            observed.update(key=args[0][0, e.position].detach().numpy().copy(),
                            value=output[0, e.position].detach().numpy().copy())
        handle = module.register_forward_hook(capture)
        try:
            s.forward(e.ids, 1, Budget())
        finally:
            handle.remove()
        parameters = {n: p.detach().clone() for n, p in model.named_parameters()}
        keys, values = s.extract(module, [e], Budget())
        np.testing.assert_array_equal(keys[0], observed["key"])
        np.testing.assert_array_equal(values[0], observed["value"])
        self.assertEqual(len(module._forward_hooks), 0)
        for name, p in model.named_parameters():
            self.assertTrue(torch.equal(p, parameters[name]))

    def test_replacement_only_selected_row_before_norm_not_layer_output(self):
        model = Model().eval(); module = model.layers[1].mlp.down_proj
        x = torch.randn(1, 9, 10); value = torch.arange(6, dtype=torch.float32) + 2
        plain = module(x).detach()
        with output_value(module, 3, value):
            changed = module(x).detach()
        self.assertTrue(torch.equal(changed[0, 3], value))
        self.assertTrue(torch.equal(changed[:, :3], plain[:, :3]))
        self.assertTrue(torch.equal(changed[:, 4:], plain[:, 4:]))
        self.assertTrue(torch.equal(module(x).detach(), plain))
        e = prepare(Tokenizer(), REQUEST.prompt, REQUEST.subject)
        raw = []
        h = model.layers[1].post_feedforward_layernorm.register_forward_pre_hook(
            lambda m, args: raw.append(args[0].detach().clone()))
        try:
            with output_value(module, e.position, value):
                model(torch.tensor([e.ids]))
        finally:
            h.remove()
        self.assertTrue(torch.equal(raw[0][0, e.position], value))
        self.assertNotEqual(list(raw[0].shape), list(value.shape))
        self.assertTrue(all(not l.mlp.down_proj._forward_hooks for l in model.layers))

    def test_optimization_freezes_weights_restores_flags_and_counts_budget(self):
        model = Model().train(); model.layers[0].eval()
        modes = [m.training for m in model.modules()]
        flags = [p.requires_grad for p in model.parameters()]
        before = {n: p.detach().clone() for n, p in model.named_parameters()}
        s = solver(model); budget = Budget()
        result = s.solve(1, REQUEST, PREFIXES, config=CONFIG, budget=budget)
        self.assertEqual(result.metadata["forwards"], len(PREFIXES) + 1 + CONFIG.steps * (len(PREFIXES) + 1))
        self.assertEqual(result.metadata["gradient_steps"], CONFIG.steps)
        self.assertFalse(np.array_equal(result.v_star, result.w_k_star))
        self.assertLess(result.metadata["history"][-1]["nll"], result.metadata["history"][0]["nll"])
        self.assertEqual([m.training for m in model.modules()], modes)
        self.assertEqual([p.requires_grad for p in model.parameters()], flags)
        for name, p in model.named_parameters():
            self.assertTrue(torch.equal(p, before[name]))
            self.assertIsNone(p.grad)
        for phrase in (REQUEST.prompt, *PREFIXES, "AB CD is a", REQUEST.target,
                       *[p + REQUEST.prompt for p in PREFIXES]):
            self.assertIn(phrase, result.seen_phrasings)

    def test_bit_determinism_and_backend_dense_roundtrip(self):
        s = solver()
        # Nonidentity C is saved exactly as used by the solve.
        covariance = np.diag(np.linspace(1, 3, 10)).astype(np.float32)
        a = s.solve(1, REQUEST, PREFIXES, covariance=covariance, config=CONFIG)
        b = s.solve(1, REQUEST, PREFIXES, covariance=covariance, config=CONFIG)
        with tempfile.TemporaryDirectory() as root:
            p, q = Path(root) / "a", Path(root) / "b"
            a.save(p); b.save(q)
            self.assertEqual({x.name: x.read_bytes() for x in p.iterdir()},
                             {x.name: x.read_bytes() for x in q.iterdir()})
            edit = backends.load(p)
            self.assertEqual(edit.content_sha256(), a.backend().content_sha256())
            meta = json.loads((p / "backend.json").read_text())
            self.assertEqual(meta["kind"], "low_rank")
            self.assertEqual(meta["label"], "synthetic-ROME-rank1")
            self.assertEqual(meta["params"]["targets"][0]["module"], "model.language_model.layers.1.mlp.down_proj")
            for name in ("k_star", "v_star", "w_k_star", "C"):
                self.assertTrue((p / f"{name}.npy").is_file())
            proof = json.loads((p / "solver.json").read_text())
            for name, digest in proof["artifact_sha256"].items():
                self.assertEqual(hashlib.sha256((p / name).read_bytes()).hexdigest(), digest)
            np.testing.assert_array_equal(np.load(p / "C.npy"), covariance)
            u, v = rank_one_update(np.load(p / "k_star.npy"), np.load(p / "v_star.npy"),
                                   np.load(p / "w_k_star.npy"), np.load(p / "C.npy"))
            np.testing.assert_array_equal(u, np.load(p / "U0.npy"))
            np.testing.assert_array_equal(v, np.load(p / "V0.npy"))
            module = s.model.layers[1].mlp.down_proj
            x = torch.randn(1, 5, 10)
            dense = x @ (module.weight.detach() + torch.from_numpy(a.u @ a.v.T)).T
            with edit.apply(SimpleNamespace(model=s.model)):
                hooked = module(x)
            self.assertTrue(torch.allclose(hooked, dense, atol=2e-6, rtol=2e-6))

    def test_deadlines_before_any_forward(self):
        s = solver(Model(depth=24))
        for label, good, bad in (("E2B", 13, 14), ("E4B", 22, 23)):
            site(s.model, s.layers, good, label, synthetic=True)
            s.label = label
            budget = Budget()
            with self.assertRaisesRegex(ValueError, "Layer must"):
                s.solve(bad, REQUEST, PREFIXES, config=CONFIG, budget=budget)
            self.assertEqual(budget.forwards, 0)

    def test_error_cleanup_after_forward_budget_exhaustion(self):
        s = solver(); modes = [m.training for m in s.model.modules()]
        flags = [p.requires_grad for p in s.model.parameters()]
        with self.assertRaisesRegex(RuntimeError, "forwards cap"):
            s.solve(1, REQUEST, PREFIXES, config=CONFIG, budget=Budget(max_forwards=5))
        self.assertEqual([m.training for m in s.model.modules()], modes)
        self.assertEqual([p.requires_grad for p in s.model.parameters()], flags)
        self.assertTrue(all(not l.mlp.down_proj._forward_hooks for l in s.layers))
        for param in s.model.parameters():
            self.assertIsNone(param.grad)

    def test_budget_and_inference_mode_refusals(self):
        for kwargs in (dict(max_forwards=401), dict(max_gradient_steps=101), dict(seconds=1801)):
            with self.assertRaises(ValueError):
                Budget(**kwargs)
        budget = Budget(max_gradient_steps=1); budget.take(gradient=True)
        with self.assertRaises(RuntimeError):
            budget.take(gradient=True)
        budget.deadline = 0
        with self.assertRaises(TimeoutError):
            budget.take()
        with torch.inference_mode(), self.assertRaises(ValueError):
            solver().solve(1, REQUEST, PREFIXES)


class Covariance(unittest.TestCase):
    def test_streaming_second_moment_and_relative_ridge(self):
        x = np.random.default_rng(2).standard_normal((30, 10))
        stat = SecondMoment(10); stat.add(x[:11]); stat.add(x[11:])
        result, meta = stat.finish(1e-4)
        raw = x.T @ x / len(x)
        expected = raw + np.eye(10) * (1e-4 * np.trace(raw) / 10)
        np.testing.assert_array_equal(result, expected.astype(np.float32))
        self.assertEqual(meta["count"], 30)
        self.assertTrue(np.linalg.eigvalsh(result).min() > 0)
        with self.assertRaises(ValueError):
            SecondMoment(3).finish()

    def test_fixed_windows_independent_of_record_input_order(self):
        records = {s: [(str(i), "English local text " * 5 + str(i)) for i in range(6)] for s in ("docs", "emotion", "banking77")}
        quotas = dict(docs=2, emotion=2, banking77=2)
        a, ma = fixed_windows(records, Tokenizer(), quotas, window=16)
        b, mb = fixed_windows({s: list(reversed(rows)) for s, rows in records.items()}, Tokenizer(), quotas, window=16)
        np.testing.assert_array_equal(a, b); self.assertEqual(ma, mb)
        self.assertEqual(a.shape, (6, 16))
        with self.assertRaises(ValueError):
            fixed_windows(records, Tokenizer(), quotas, window=10000)

    def test_covariance_same_forward_all_sites_and_cleanup(self):
        model = Model().eval()
        windows = np.arange(64).reshape(4, 16) + 3
        positions = (1, 7, 15)
        actual, meta = collect_model_covariances(model, model.layers, windows, (0, 1, 2),
                                                synthetic=True, positions=positions)
        self.assertEqual(meta["forwards"], 4)
        self.assertEqual(meta["gradient_steps"], 0)
        for layer in (0, 1, 2):
            expected = SecondMoment(10)
            def capture(mod, args):
                expected.add(args[0][0, list(positions)].detach().numpy())
            h = model.layers[layer].mlp.down_proj.register_forward_pre_hook(capture)
            try:
                with torch.no_grad():
                    for window in windows:
                        model(torch.as_tensor(window[None]))
            finally:
                h.remove()
            np.testing.assert_array_equal(actual[layer], expected.finish()[0])
            self.assertEqual(meta["layers"][layer]["count"], 12)
        self.assertTrue(all(not l.mlp.down_proj._forward_pre_hooks for l in model.layers))


if __name__ == "__main__":
    unittest.main(verbosity=2)
