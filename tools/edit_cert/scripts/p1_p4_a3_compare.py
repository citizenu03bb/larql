"""P1-4 under amendment A3: Claude A3 against Codex's existing solves, model-free (written by Claude Code).

The gate is A1's, unchanged: cos U >= 0.95, cos V >= 0.95 and ||U||.||V|| within 10%. It also reports the pieces both
solves share by construction (key direction u, clean key, base value, shift delta, division factor), so that any
disagreement can be traced.

  python3 scripts/p1_p4_a3_compare.py OUT.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
from p1_rome_claude.run import REQUESTS  # noqa: E402

ROOT = Path.home() / 'Projects/larql-research/phase-b-p1-v2-20261027'


def cos(a, b):
    return float(a @ b / np.linalg.norm(a) / np.linalg.norm(b))


def rel(a, b):
    return float(np.linalg.norm(a - b) / max(np.linalg.norm(b), 1e-30))


def main(out: Path) -> int:
    facts = {}
    for fact in REQUESTS:
        a, b = ROOT / 'solves-claude-a3' / fact, ROOT / 'solves-codex' / fact
        load = lambda d, k: np.load(d / f'{k}.npy').astype(np.float64)  # noqa: E731
        Ua, Va, Ub, Vb = load(a, 'U')[:, 0], load(a, 'V')[:, 0], load(b, 'U0')[:, 0], load(b, 'V0')[:, 0]
        na, nb = float(np.linalg.norm(Ua) * np.linalg.norm(Va)), float(np.linalg.norm(Ub) * np.linalg.norm(Vb))
        ratio = max(na, nb) / min(na, nb)
        da, db = load(a, 'delta').ravel(), load(b, 'delta').ravel()
        facts[fact] = dict(
            cos_U=cos(Ua, Ub), cos_V=cos(Va, Vb), norm_claude=na, norm_codex=nb, ratio=ratio,
            agree=cos(Ua, Ub) >= 0.95 and cos(Va, Vb) >= 0.95 and ratio <= 1.10,
            cos_delta=cos(da, db), norm_delta_claude=float(np.linalg.norm(da)), norm_delta_codex=float(np.linalg.norm(db)),
            k_clean_rel=rel(load(a, 'k_clean'), load(b, 'k_star')),
            division_claude=float(load(a, 'k_clean') @ Va), division_codex=float(load(b, 'k_star') @ (Vb / np.linalg.norm(Vb))),
            context_keys_identical=bool(np.array_equal(load(a, 'context_keys'), load(b, 'context_keys'))))
    n = sum(f['agree'] for f in facts.values())
    res = dict(author='Claude Code', gate='A1, unchanged', reads='solves-claude-a3 and solves-codex arrays; no forward',
               agreements=n, p1_5='runs on the agreeing facts' if 5 - n < 3 else 'blocked: P1 closes (A3 §5)', facts=facts)
    out.write_text(json.dumps(res, indent=1) + '\n')
    for f, v in facts.items():
        print(f"{f:38s} cosU {v['cos_U']:.4f} cosV {v['cos_V']:.4f} ratio {v['ratio']:.4f} cos(delta) {v['cos_delta']:.4f} "
              f"agree {v['agree']}")
    print('agreements', n, '/ 5 ->', res['p1_5'])
    return 0


if __name__ == '__main__':
    sys.exit(main(Path(sys.argv[1])))
