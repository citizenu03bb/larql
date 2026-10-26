"""Independent, model-free check of a ROME rank-1 edit's algebra (Claude Code; does not import Codex's solver).

  python3 scripts/xcheck_rome_algebra.py SOLVE_DIR B1_DIR [--out JSON]

From the saved witnesses only:
  k* = mean(context_keys) in float64, cast to float32         -> equals saved k_star
  q  = solve(C, k*) in float64;  U = v* - Wk*;  V = q / (k*^T q) -> equals saved U0 / V0
  W' k* = Wk* + U (V^T k*)                                       -> equals v*
Plus: every artifact hash against solver.json, C against the B1 covariance for that layer, the solve receipt,
and the edit's content hash recomputed by the harness's own loader. Differences are reported, never repaired.
This checks the rank-1 algebra and provenance. It does not replicate the value optimisation, and it does not
establish an edit of a fact.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from edit_cert import backends  # noqa: E402

TOL = 2e-5


def sha(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def close(a, b) -> dict:
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    diff = np.abs(a - b)
    ok = bool(np.all(diff <= TOL + TOL * np.abs(b)))
    return dict(ok=ok, max_abs=float(diff.max()), max_rel=float((diff / np.maximum(np.abs(b), 1e-30)).max()))


def check(solve: Path, b1: Path) -> dict:
    edit = solve / 'edit'
    status = json.loads((solve / 'status.json').read_text())
    meta = json.loads((edit / 'solver.json').read_text())
    layer = int(status['layer'])
    out = dict(solve=str(solve), fact_id=status['fact_id'], layer=layer, checks={}, author='Claude Code')
    c = out['checks']

    c['receipt_complete'] = dict(ok=status['status'] == 'complete' and not status['synthetic']
                                 and status['forwards'] == 130 and status['gradient_steps'] == 25,
                                 forwards=status['forwards'], gradient_steps=status['gradient_steps'])
    hashes = {name: sha(edit / name) for name in meta['artifact_sha256']}
    c['artifact_hashes'] = dict(ok=hashes == meta['artifact_sha256'],
                                differ=[n for n in hashes if hashes[n] != meta['artifact_sha256'][n]])
    b1s = json.loads((b1 / 'status.json').read_text())
    c['covariance_is_b1'] = dict(ok=hashes['C.npy'] == b1s['covariance_sha256'][str(layer)] == sha(b1 / f'C{layer}.npy'),
                                 b1_sha256=b1s['covariance_sha256'][str(layer)])

    loaded = backends.load(edit)
    c['content_hash'] = dict(ok=loaded.content_sha256() == status['edit_content_sha256'],
                             form=loaded.form(), module=loaded.targets[0]['module'])
    c['site'] = dict(ok=loaded.targets == [dict(module=f'model.language_model.layers.{layer}.mlp.down_proj',
                                                layer=layer, scale=1.0)] and layer <= 13)

    keys = np.load(edit / 'context_keys.npy')
    k_star = np.load(edit / 'k_star.npy')
    v_star = np.load(edit / 'v_star.npy')
    wk = np.load(edit / 'w_k_star.npy')
    C = np.load(edit / 'C.npy')
    U0, V0 = np.load(edit / 'U0.npy'), np.load(edit / 'V0.npy')
    k_re = keys.astype(np.float64).mean(axis=0).astype(np.float32)
    c['k_star_mean'] = dict(**close(k_re, k_star), bit_equal=bool(np.array_equal(k_re, k_star)), contexts=int(keys.shape[0]))

    k64 = k_star.astype(np.float64)
    C64 = C.astype(np.float64)
    c['C_symmetric'] = dict(ok=bool(np.array_equal(C, C.T)))
    q = np.linalg.solve(C64, k64)
    U = (v_star.astype(np.float64) - wk.astype(np.float64))[:, None]
    V = (q / (k64 @ q))[:, None]
    c['U0'] = close(U.astype(np.float32), U0)
    c['V0'] = close(V.astype(np.float32), V0)
    image = wk.astype(np.float64) + (U0.astype(np.float64) @ (V0.astype(np.float64).T @ k64))
    c['constraint_Wprime_kstar_eq_vstar'] = close(image, v_star)
    out['norms'] = dict(U=float(np.linalg.norm(U0)), V=float(np.linalg.norm(V0)),
                        update_frobenius=float(np.linalg.norm(U0) * np.linalg.norm(V0)),
                        v_minus_wk=float(np.linalg.norm(v_star - wk)), wk=float(np.linalg.norm(wk)))
    out['ok'] = all(v['ok'] for v in c.values() if 'ok' in v)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('solve', type=Path)
    ap.add_argument('b1', type=Path)
    ap.add_argument('--out', type=Path)
    a = ap.parse_args()
    r = check(a.solve, a.b1)
    text = json.dumps(r, indent=1, sort_keys=True)
    if a.out:
        a.out.write_text(text + '\n')
    print(json.dumps(dict(fact=r['fact_id'], layer=r['layer'], ok=r['ok'],
                          failed=[k for k, v in r['checks'].items() if not v.get('ok', True)])))
    return 0 if r['ok'] else 1


if __name__ == '__main__':
    sys.exit(main())
