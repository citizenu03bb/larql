"""P1-4, Claude's model-free recomputation of the pair comparison and of its causes (written by Claude Code).

No forward. It reads both agents' saved factors, C17 and the shared templates, and recomputes the agreement gate
of docs/phase-b-p1-amendment-a1.md. It then splits each norm ratio into two factors:
  - objective: ||delta_codex|| / ||v*_claude - z0_claude||, the size of the solved output shift;
  - key: (u . k_mean) / (u . k_clean), with u the unit C^-1 k_mean direction that both V share. The constraint
    key is k_mean for Claude, and the clean prompt's key (template '{}') for Codex.
The two factors multiply to the norm ratio exactly, since U_codex is parallel to delta_codex.

  python3 scripts/p1_p4_xcheck_claude.py OUT.json
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


def main(out: Path) -> int:
    import transformers
    from p1_rome_claude.solve import subject_last
    tok = transformers.AutoTokenizer.from_pretrained('/mnt/models/hf-models/gpt2-xl', local_files_only=True)
    assert json.loads((ROOT / 'templates/templates.json').read_text())['templates'][0] == '{}'
    C = np.load(ROOT / 'covariance/C17.npy').astype(np.float64)
    facts = {}
    for fact, (prompt, subject, _) in REQUESTS.items():
        a, b = ROOT / 'solves-claude' / fact, ROOT / 'solves-codex' / fact
        Ua, Va = (np.load(a / f'{k}.npy')[:, 0].astype(np.float64) for k in ('U', 'V'))
        Ub, Vb = (np.load(b / f'{k}.npy')[:, 0].astype(np.float64) for k in ('U0', 'V0'))
        K, Kb = np.load(a / 'context_keys.npy').astype(np.float64), np.load(b / 'context_keys.npy').astype(np.float64)
        k_mean, k_clean = K.mean(axis=0), K[0]
        q = np.linalg.solve(C, k_mean)
        u = q / np.linalg.norm(q)
        shift_a = float(np.linalg.norm(np.load(a / 'v_star.npy').astype(np.float64) - np.load(a / 'z0.npy').astype(np.float64)))
        delta = np.load(b / 'delta.npy').astype(np.float64).ravel()
        na, nb = float(np.linalg.norm(Ua) * np.linalg.norm(Va)), float(np.linalg.norm(Ub) * np.linalg.norm(Vb))
        objective, key = float(np.linalg.norm(delta)) / shift_a, float((u @ k_mean) / (u @ k_clean))
        text = prompt.format(subject)
        facts[fact] = dict(
            cos_U=cos(Ua, Ub), cos_V=cos(Va, Vb), norm_claude=na, norm_codex=nb, ratio=max(na, nb) / min(na, nb),
            agree=cos(Ua, Ub) >= 0.95 and cos(Va, Vb) >= 0.95 and max(na, nb) / min(na, nb) <= 1.10,
            context_keys_identical=bool(np.array_equal(K, Kb)),
            codex_key_is_clean=bool(np.allclose(np.load(b / 'k_star.npy').astype(np.float64), k_clean, atol=1e-5)),
            cos_delta_U_codex=cos(delta, Ub),
            objective_factor=objective, key_factor=key, product=objective * key,
            u_dot_k_mean=float(u @ k_mean), u_dot_k_clean=float(u @ k_clean),
            clean_subject_last_position=subject_last(tok, text, subject),
            clean_tokens=tok.convert_ids_to_tokens(tok.encode(text))[:subject_last(tok, text, subject) + 1],
            norm_k_clean=float(np.linalg.norm(k_clean)), median_norm_other_keys=float(np.median(np.linalg.norm(K[1:], axis=1))))
    res = dict(author='Claude Code', reads='both solve outputs, C17, templates; no peer source; no forward',
               agreements=sum(f['agree'] for f in facts.values()), facts=facts)
    out.write_text(json.dumps(res, indent=1, ensure_ascii=False) + '\n')
    for f, v in facts.items():
        print(f"{f:38s} cosU {v['cos_U']:.4f} cosV {v['cos_V']:.4f} ratio {v['ratio']:8.3f} = objective "
              f"{v['objective_factor']:.2f} x key {v['key_factor']:.2f}  (subject last at {v['clean_subject_last_position']})")
    print('agreements', res['agreements'], '/ 5')
    return 0


if __name__ == '__main__':
    sys.exit(main(Path(sys.argv[1])))
