"""Cross-check of Codex's nf4-h3-timing-20261026 by a different agent (Claude Code).

Written from the frozen protocol docstring of run_nf4_h3_timing.py and the raw artifacts; Codex's
checker not consulted. Model-free.
"""
import hashlib, json, sys
from pathlib import Path
import numpy as np

F = Path(sys.argv[1] if len(sys.argv) > 1 else '/home/pat/Projects/larql-research/exploratory-natural-fact-edit/nf4-h3-timing-20261026/core')
A = Path('/home/pat/Projects/larql-research/exploratory-natural-fact-edit/nf4-asymmetry-20261026/interventions')
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def calls(root):
    s = json.loads((root / 'summary.json').read_text()); return s, {c['name']: c for c in s['calls']}
s, C = calls(F); _, CA = calls(A)
def load(root, c):
    assert sha(root / c['artifact']) == c['artifact_sha256'], c['name']
    with np.load(root / c['artifact']) as z: return {k: z[k].copy() for k in z.files}
assert s['status'] == 'complete' and len(s['calls']) == 24 and [c['ordinal'] for c in s['calls']] == list(range(24))
f16 = lambda x: x.astype(np.float16).astype(np.float32)
R21 = np.load(A / 'residual.npy'); RND = np.load(A / 'random.npy')
assert np.array_equal(np.load(F / 'residual-L21.npy'), R21) and np.array_equal(np.load(F / 'random.npy'), RND)
N = float(np.linalg.norm(R21.astype(np.float16).astype(np.float64)))
sb, sd = load(F, C['symbol_base']), load(F, C['symbol_donor'])
vec = {21: R21}
for L in (8, 14):
    e = (sd[f'entity_L{L:02}'] - sb[f'entity_L{L:02}']).astype(np.float64)
    v = (e / np.linalg.norm(e) * N).astype(np.float32); assert np.array_equal(np.load(F / f'residual-L{L:02}.npy'), v); vec[L] = v
    assert abs(np.linalg.norm(v.astype(np.float16).astype(np.float64)) / N - 1) <= 0.001
TGT = dict(symbol=(236814, 2209), number=(886, 1156))
out = {}
for q, (nat, tgt) in TGT.items():
    base, donor = load(F, C[f'{q}_base']), load(F, C[f'{q}_donor'])
    ids = C[f'{q}_base']['input_ids']; d_ids = C[f'{q}_donor']['input_ids']
    assert d_ids[:5] == ids[:5] and d_ids[6:] == ids[6:] and d_ids[5] == 42404
    for m in ('base', 'donor'): assert np.array_equal(load(F, C[f'{q}_{m}'])['logits'], load(A, CA[f'{q}_{m}'])['logits']), (q, m)
    mg = lambda d: float(d['logits'][tgt] - d['logits'][nat]); m0 = mg(base)
    zeros = [f'{q}_zero_L{L:02}' for L in (8, 14, 21) if f'{q}_zero_L{L:02}' in C]
    for z in zeros: assert np.array_equal(load(F, C[z])['logits'], base['logits']), z
    res = {}
    for L in (8, 14, 21):
        for kind, v in (('R', vec[L]), ('X', RND)):
            name = f'{q}_L{L:02}' if kind == 'R' else f'{q}_random_L{L:02}'
            d = load(F, C[name]); assert C[name]['input_ids'] == ids and C[name]['patch_layer'] == L
            assert np.array_equal(d['requested_delta'], f16(v))
            assert np.array_equal(np.delete(d['patch_after'], 5, 0), np.delete(d['patch_before'], 5, 0))
            ch = np.linalg.norm(d['patch_after'][5].astype(np.float64) - d['patch_before'][5].astype(np.float64))
            assert abs(ch / N - 1) <= 0.01, (name, ch / N)
            assert np.array_equal(d['ple_before'], base['ple_before'])
            if L == 21: assert np.array_equal(d['logits'], load(A, CA[f'{q}_{"residual" if kind == "R" else "random"}'])['logits'])
            res[(L, kind)] = mg(d) - m0
            m = s['metrics'][q][str(L)]
            assert (m['effect'] if kind == 'R' else m['random_effect']) == res[(L, kind)]
    out[q] = {f'L{L}': dict(effect=res[(L, 'R')], random=res[(L, 'X')]) for L in (8, 14, 21)} | dict(native_margin=m0, donor_margin=mg(donor))
# frozen readings recomputed
E = lambda q, L: out[q][f'L{L}']['effect']; X = lambda q, L: out[q][f'L{L}']['random']
spec = {k: E('number', k) >= 1 and E('number', k) - E('number', 21) >= 1 and E('number', k) > 2 * max(X('number', k), 0) for k in (8, 14)}
match = {k: spec[k] and E('symbol', k) <= E('symbol', 21) + 1 for k in (8, 14)}
ordered = all(match.values()) and E('number', 8) >= E('number', 14) + 1 and E('number', 14) >= E('number', 21) + 1
reading = 'ordered' if ordered else 'partial/local' if any(match.values()) else 'earlier number sensitivity without relative selectivity' if any(spec.values()) else 'no support'
v = dict(status='cross_checked', checker='Claude Code (different agent from the runner author, Codex)',
         basis='frozen protocol docstring and raw artifacts; Codex checker not consulted', forwards=24,
         common_norm=N, results=out, specific_number=spec, relative_timing_match=match, reading=reading,
         recorded_readings=s['readings'])
print(json.dumps(v, indent=1, default=str))
