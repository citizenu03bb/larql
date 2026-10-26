"""The local transformers + bitsandbytes NF4 bench (Codex's frozen `setup()` in scripts/run_nf4_local_bench.py).

Loading needs the GPU and the HF weights; it is never called by the model-free tests. Every forward is a full
prefix, no cache, last-position logits, under inference mode with deterministic settings.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[4]
KNOWN_DEADLINES = {'E4B': 22, 'E2B': 13}   # last block whose output still feeds a K/V-storing layer


def kv_write_deadline(kv_store_layers: dict) -> int:
    return max(int(k) for k in kv_store_layers) - 1


class Nf4Instrument:
    synthetic = False

    def __init__(self, label, model, layers, tokenizer, eos_ids, info):
        self.label, self.name = label, f'nf4-{label.lower()}'
        self.model, self.layers, self.tok = model, layers, tokenizer
        self.eos_ids = frozenset(eos_ids)
        self.info = info
        self.kv_write_deadline = kv_write_deadline(info['kv_store_layers'])
        if label in KNOWN_DEADLINES:
            assert self.kv_write_deadline == KNOWN_DEADLINES[label], (label, info['kv_store_layers'])
        self.current_ids: list[int] = []

    @classmethod
    def load(cls, label: str, model_root: Path) -> 'Nf4Instrument':
        import torch
        sys.path.insert(0, str(REPO / 'scripts'))
        from run_nf4_local_bench import SEED, setup
        from transformers import AutoTokenizer
        torch.manual_seed(SEED)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.use_deterministic_algorithms(True)
        model, layers, _, stops, _, info, device = setup(label, Path(model_root), False)
        assert device == 'cuda'
        tok = AutoTokenizer.from_pretrained(Path(model_root) / f'gemma-4-{label}-it', local_files_only=True)
        return cls(label, model, layers, tok, stops, info)

    def chat_prompt(self, question: str) -> list[int]:
        ids = self.tok.apply_chat_template([{'role': 'user', 'content': question}], tokenize=True,
                                           add_generation_prompt=True, enable_thinking=False, return_dict=False)
        assert isinstance(ids, list) and all(isinstance(i, int) for i in ids)
        return ids

    def token_ids(self, surface: str) -> list[int]:
        return self.tok.encode(surface, add_special_tokens=False)

    def decode(self, ids) -> str:
        return self.tok.decode([i for i in ids if i not in self.eos_ids])

    def forward(self, ids: list[int]) -> np.ndarray:
        import torch
        self.current_ids = list(ids)
        device = next(self.layers.parameters()).device
        with torch.inference_mode():
            out = self.model(input_ids=torch.tensor([ids], dtype=torch.long, device=device),
                             use_cache=False, logits_to_keep=1)
        logits = out.logits[0, -1].float().cpu().numpy()
        assert np.isfinite(logits).all(), 'nonfinite logits'
        return logits

    def fingerprint(self) -> dict:
        keep = ('metadata_sha256', 'quantization', 'kv_store_layers', 'kv_shared_layers',
                'transformers_modeling_source_sha256', 'weight_files_stat_only_not_hashed')
        return dict(instrument=self.name, synthetic=False, **{k: self.info.get(k) for k in keep})
