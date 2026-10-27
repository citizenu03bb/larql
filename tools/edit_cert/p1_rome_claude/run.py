"""P1-3 services for Claude's side (docs/phase-b-p1-amendment-a1.md), run under run_p1_service.sh.

  python3 -m p1_rome_claude.run templates --i-have-approval     # P1-3a: the shared context templates
  python3 -m p1_rome_claude.run solve FACT --i-have-approval    # P1-3b: Claude's solve of one feasible fact

Outputs go to ~/Projects/larql-research/phase-b-p1-v2-20261027/. A solve needs the P1-2 covariance file and the
shared templates, both hash-checked; it saves the edit in the harness's low_rank format plus its witnesses.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))

ROOT = Path.home() / 'Projects/larql-research/phase-b-p1-v2-20261027'
MODEL_ROOT = Path('/mnt/models/hf-models')
SEED = 20261027
REQUESTS = {   # amendment A1: S1 with the subject as '{}', the first new value with its leading space
    'p1-bmw-origin-italy': ('{} is a company that was founded in the country of', 'BMW', ' Italy'),
    'p1-germany-language-finnish': ('The official language of {} is', 'Germany', ' Finnish'),
    'p1-gretzky-sport-baseball': ('{} is famous for playing the sport of', 'Wayne Gretzky', ' baseball'),
    'p1-nintendo-origin-france': ('{} is a company that was founded in the country of', 'Nintendo', ' France'),
    'p1-shakespeare-nationality-spanish': ('{} was a native of the country of', 'William Shakespeare', ' Spain'),
}
SEQ_CAP, STEP_CAP = 500, 25


def sha(p: Path) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


class Counter:
    """Counts sequence-forwards through the model (a batch of n rows counts n) and refuses past the cap."""

    def __init__(self, model, cap):
        self.n, self.cap = 0, cap
        self.h = model.register_forward_pre_hook(self.pre, with_kwargs=True)

    def pre(self, mod, args, kwargs):
        x = kwargs.get('input_ids', args[0] if args else None)
        self.n += int(x.shape[0])
        if self.n > self.cap:
            raise RuntimeError(f'sequence-forward cap {self.cap} exceeded')


def load():
    from edit_cert.instruments.hf_gpt2 import Gpt2Instrument
    return Gpt2Instrument.load(MODEL_ROOT)


def templates() -> int:
    from p1_rome_claude import solve as rome
    out = ROOT / 'templates'
    out.mkdir(parents=True, exist_ok=False)
    t0 = time.time()
    inst = load()
    c = Counter(inst.model, 400)
    tmpl = rome.context_templates(inst.model, inst.tok, SEED)
    (out / 'templates.json').write_text(json.dumps(dict(templates=tmpl, seed=SEED, author='Claude Code',
                                                        hp=rome.PUBLISHED), indent=1, ensure_ascii=False) + '\n')
    rec = dict(forwards=c.n, seconds=round(time.time() - t0, 1), templates_sha256=sha(out / 'templates.json'),
               fingerprint=inst.fingerprint(), n=len(tmpl))
    (out / 'status.json').write_text(json.dumps(rec, indent=1) + '\n')
    print(json.dumps(dict(ok=True, n=len(tmpl), forwards=c.n)))
    return 0


def solve(fact: str) -> int:
    from p1_rome_claude import solve as rome
    from edit_cert.backends.lowrank import LowRank
    prompt, subject, target = REQUESTS[fact]
    cov_index = json.loads((ROOT / 'covariance' / 'status.json').read_text())
    C = np.load(ROOT / 'covariance' / 'C17.npy')
    assert sha(ROOT / 'covariance' / 'C17.npy') == cov_index['covariance_sha256'], 'covariance hash mismatch'
    tstat = json.loads((ROOT / 'templates' / 'status.json').read_text())
    assert sha(ROOT / 'templates' / 'templates.json') == tstat['templates_sha256'], 'templates hash mismatch'
    tmpl = json.loads((ROOT / 'templates' / 'templates.json').read_text())['templates']
    out = ROOT / 'solves-claude' / fact
    out.mkdir(parents=True, exist_ok=False)
    t0 = time.time()
    inst = load()
    c = Counter(inst.model, SEQ_CAP)
    r = rome.solve(inst.model, inst.tok, rome.Request(prompt, subject, target), tmpl, C)
    assert len(r['trace']) <= STEP_CAP
    for k in ('U', 'V', 'k_star', 'v_star', 'z0', 'context_keys'):
        np.save(out / f'{k}.npy', r[k])
    edit = LowRank([dict(module='transformer.h.17.mlp.c_proj', layer=17, scale=1.0)], dict(U0=r['U'], V0=r['V']),
                   label=f'ROME rank 1, Claude solve, {fact}', seen_phrasings=[prompt.format(subject)])
    edit.save(out / 'edit')
    rec = dict(fact=fact, forwards=c.n, gradient_steps=len(r['trace']), seconds=round(time.time() - t0, 1),
               constraint_error=r['constraint_error'], trace=r['trace'], decisions=r['decisions'],
               covariance_sha256=cov_index['covariance_sha256'], templates_sha256=tstat['templates_sha256'],
               edit_content_sha256=edit.content_sha256(), fingerprint=inst.fingerprint(),
               norms=dict(U=float(np.linalg.norm(r['U'])), V=float(np.linalg.norm(r['V'])),
                          z0=float(np.linalg.norm(r['z0'])), v_minus_z0=float(np.linalg.norm(r['v_star'] - r['z0']))))
    (out / 'status.json').write_text(json.dumps(rec, indent=1) + '\n')
    print(json.dumps(dict(ok=True, fact=fact, forwards=c.n, steps=len(r['trace']), constraint_error=r['constraint_error'])))
    return 0


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('cmd', choices=['templates', 'solve'])
    ap.add_argument('fact', nargs='?')
    ap.add_argument('--i-have-approval', action='store_true')
    a = ap.parse_args()
    if not a.i_have_approval:
        sys.exit('refused: no --i-have-approval')
    sys.exit(templates() if a.cmd == 'templates' else solve(a.fact))
