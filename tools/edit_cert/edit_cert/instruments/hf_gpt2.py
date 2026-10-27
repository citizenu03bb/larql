"""GPT-2 XL in fp16 on the local GPU, for the P1 positive control (docs/phase-b-p1-positive-control-design.md).

A base model: prompts are plain completion prompts, with no chat template. It rarely emits EOS, so an answer
ends at its first *stop token*: a token whose text contains a newline or ends with one of . , ; ! ? (after
trailing spaces). This reading is declared for this instrument only, before any forward; an answer that reaches
the cap without a stop token stays truncated, as for every instrument. Stop tokens stay in the decoded text,
since one may carry the value itself (" London."); only the real EOS is dropped. GPT-2 has no K/V sharing, so
the write deadline is the last block.

Loading needs the GPU and the HF weights; the model-free tests use only `stop_token_ids` on the tokenizer.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np

STOP_ENDINGS = ('.', ',', ';', '!', '?')
NAME = 'hf-gpt2xl'


def stop_token_ids(tokenizer) -> frozenset[int]:
    """Every vocabulary id whose text holds a newline or ends with a stop punctuation, plus EOS."""
    out = {tokenizer.eos_token_id}
    for i in range(len(tokenizer)):
        s = tokenizer.decode([i])
        if '\n' in s or s.rstrip(' ').endswith(STOP_ENDINGS):
            out.add(i)
    return frozenset(out)


def file_sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


class Gpt2Instrument:
    synthetic = False
    name = NAME

    def __init__(self, model, tokenizer, info):
        self.model, self.tok, self.info = model, tokenizer, info
        self.layers = model.transformer.h
        self.eos_ids = stop_token_ids(tokenizer)
        self.kv_write_deadline = len(self.layers) - 1
        self.current_ids: list[int] = []

    @classmethod
    def load(cls, model_root: Path) -> 'Gpt2Instrument':
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        root = Path(model_root) / 'gpt2-xl'
        torch.manual_seed(20261027)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.use_deterministic_algorithms(True)
        tok = AutoTokenizer.from_pretrained(root, local_files_only=True)
        model = AutoModelForCausalLM.from_pretrained(root, local_files_only=True, torch_dtype=torch.float16).cuda().eval()
        model.requires_grad_(False)
        info = dict(weights_sha256=(root / 'SHA256SUMS').read_text() if (root / 'SHA256SUMS').exists() else None,
                    n_layer=model.config.n_layer, dtype='float16',
                    tokenizer_sha256=file_sha256(root / 'tokenizer.json'))
        return cls(model, tok, info)

    def chat_prompt(self, question: str) -> list[int]:
        """The harness's prompt hook: here the question text is the completion prefix itself."""
        return self.tok.encode(question)

    def token_ids(self, surface: str) -> list[int]:
        return self.tok.encode(surface)

    def decode(self, ids) -> str:
        return self.tok.decode([i for i in ids if i != self.tok.eos_token_id])

    def forward(self, ids: list[int]) -> np.ndarray:
        import torch
        self.current_ids = list(ids)
        with torch.inference_mode():
            out = self.model(input_ids=torch.tensor([ids], dtype=torch.long, device='cuda'), use_cache=False)
        logits = out.logits[0, -1].float().cpu().numpy()
        assert np.isfinite(logits).all(), 'nonfinite logits'
        return logits

    def fingerprint(self) -> dict:
        return dict(instrument=self.name, synthetic=False, **self.info)
