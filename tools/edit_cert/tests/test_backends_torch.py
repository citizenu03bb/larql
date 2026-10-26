"""Torch backends on a synthetic stack. unittest-style so the model venv (torch, no pytest) can run it:
    /mnt/models/finetune/venv/bin/python -m unittest discover -s tests -p 'test_backends_torch.py'
"""
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None

from edit_cert import backends  # noqa: E402
from edit_cert.backends.activation import ActivationStatic  # noqa: E402
from edit_cert.backends.lowrank import LowRank  # noqa: E402
from edit_cert.instruments.nf4 import Nf4Instrument, kv_write_deadline  # noqa: E402


def stack(width=6, depth=4, vocab=32):
    class Block(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.proj = torch.nn.Linear(width, width, bias=False)

        def forward(self, h):
            return h + 0.1 * self.proj(h)

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            torch.manual_seed(0)
            self.embed = torch.nn.Embedding(vocab, width)
            self.layers = torch.nn.ModuleList(Block() for _ in range(depth))
            self.head = torch.nn.Linear(width, vocab, bias=False)

        def forward(self, input_ids, use_cache=False, logits_to_keep=1):
            h = self.embed(input_ids)
            for layer in self.layers:
                h = layer(h)
            return SimpleNamespace(logits=self.head(h)[:, -logits_to_keep:])

    return Model().eval()


class Tok:
    def apply_chat_template(self, msgs, **kw):
        return [1] + [3 + (ord(c) % 20) for c in msgs[0]['content']] + [2]

    def encode(self, s, add_special_tokens=False):
        return [3 + (ord(c) % 20) for c in s]

    def decode(self, ids):
        return ' '.join(map(str, ids))


def instrument(model):
    return Nf4Instrument('TEST', model, model.layers, Tok(), {0}, dict(kv_store_layers={'2': 's', '3': 'f'}))


@unittest.skipIf(torch is None, 'torch not installed')
class TorchBackends(unittest.TestCase):
    def test_deadline_rule(self):
        self.assertEqual(kv_write_deadline({'22': 's', '23': 'f'}), 22)
        self.assertEqual(kv_write_deadline({'13': 's', '14': 'f'}), 13)
        self.assertEqual(instrument(stack()).kv_write_deadline, 2)

    def test_static_activation_touches_trigger_rows_only_and_is_removed(self):
        model = stack()
        inst = instrument(model)
        ids = [1, 5, 7, 5, 9]
        vec = np.arange(6, dtype=np.float32)
        edit = ActivationStatic(1, vec, 2.0, trigger_ids=[5])
        seen = {}
        # observe the input of block 2, i.e. block 1's output after every hook on block 1
        model.layers[2].register_forward_pre_hook(lambda m, a: seen.__setitem__('out', a[0].detach().clone()))
        with torch.inference_mode():
            inst.forward(ids)
            base = seen['out']
            with edit.apply(inst):
                inst.forward(ids)
                patched = seen['out']
            inst.forward(ids)
            after = seen['out']
        diff = (patched - base)[0]
        for row in (1, 3):
            self.assertTrue(torch.allclose(diff[row], torch.tensor(2.0 * vec)))
        for row in (0, 2, 4):
            self.assertTrue(torch.equal(diff[row], torch.zeros(6)))
        self.assertTrue(torch.equal(after, base))

    def test_static_activation_is_inert_without_trigger(self):
        model = stack()
        inst = instrument(model)
        edit = ActivationStatic(1, np.ones(6, dtype=np.float32), 1.0, trigger_ids=[31])
        a = inst.forward([1, 4, 6])
        with edit.apply(inst):
            b = inst.forward([1, 4, 6])
        self.assertTrue(np.array_equal(a, b))

    def test_low_rank_equals_dense_update(self):
        model = stack()
        inst = instrument(model)
        rng = np.random.default_rng(1)
        u, v = rng.standard_normal((6, 1)).astype(np.float32), rng.standard_normal((6, 1)).astype(np.float32)
        edit = LowRank([dict(module='layers.2.proj', layer=2, scale=0.5)], dict(U0=u, V0=v))
        x = torch.randn(1, 3, 6)
        with torch.inference_mode():
            plain = model.layers[2].proj(x)
            with edit.apply(inst):
                hooked = model.layers[2].proj(x)
            again = model.layers[2].proj(x)
        dense = x @ torch.as_tensor(0.5 * u @ v.T).T
        self.assertTrue(torch.allclose(hooked - plain, dense, atol=1e-6))
        self.assertTrue(torch.equal(again, plain))
        self.assertEqual(edit.form()['max_write_layer'], 2)

    def test_save_load_round_trip_and_random_norms(self):
        rng = np.random.default_rng(2)
        static = ActivationStatic(3, rng.standard_normal(6).astype(np.float32), 1.5, [5, 6], label='s', seen_phrasings=['q'])
        low = LowRank([dict(module='layers.1.proj', layer=1)], dict(U0=rng.standard_normal((6, 1)).astype(np.float32),
                                                                    V0=rng.standard_normal((6, 1)).astype(np.float32)))
        with tempfile.TemporaryDirectory() as d:
            for i, e in enumerate((static, low)):
                e.save(Path(d) / str(i))
                back = backends.load(Path(d) / str(i))
                self.assertEqual(back.content_sha256(), e.content_sha256())
                self.assertEqual(back.form(), e.form())
        r = static.random_counterpart(7)
        self.assertAlmostEqual(float(np.linalg.norm(r.vector.astype(np.float16).astype(np.float64))),
                               float(np.linalg.norm((1.5 * static.vector).astype(np.float16).astype(np.float64))), places=2)
        self.assertEqual(r.trigger_ids, static.trigger_ids)
        rl = low.random_counterpart(7)
        for k in ('U0', 'V0'):
            self.assertAlmostEqual(float(np.linalg.norm(rl.arrays()[k])), float(np.linalg.norm(low.arrays()[k])), places=4)

    def test_tampered_artifact_is_refused_on_load(self):
        with tempfile.TemporaryDirectory() as d:
            ActivationStatic(3, np.ones(6, dtype=np.float32), 1.0, [5]).save(Path(d) / 'e')
            np.save(Path(d) / 'e' / 'vector.npy', np.zeros(6, dtype=np.float32))
            with self.assertRaises(ValueError):
                backends.load(Path(d) / 'e')


if __name__ == '__main__':
    unittest.main()
