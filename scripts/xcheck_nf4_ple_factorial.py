"""Cross-check of Codex's nf4-ple-factorial-20261026 by a different agent (Claude Code).

Written from the frozen protocol text and the raw artifacts, not from Codex's checker. Model-free.
"""
import hashlib, json, sys
from pathlib import Path
import numpy as np

F = Path(sys.argv[1] if len(sys.argv) > 1 else '/home/pat/Projects/larql-research/exploratory-natural-fact-edit/nf4-ple-factorial-20261026/factorial')
A = Path('/home/pat/Projects/larql-research/exploratory-natural-fact-edit/nf4-asymmetry-20261026/interventions')
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def calls(root):
    s = json.loads((root / 'summary.json').read_text()); return s, {c['name']: c for c in s['calls']}
s, C = calls(F); sa, CA = calls(A)
def load(root, c):
    assert sha(root / c['artifact']) == c['artifact_sha256'], c['name']
    with np.load(root / c['artifact']) as z: return {k: z[k].copy() for k in z.files}
assert s['status'] == 'complete' and len(s['calls']) == 22 and [c['ordinal'] for c in s['calls']] == list(range(22))
R = np.load(F / 'residual.npy'); assert np.array_equal(R, np.load(A / 'residual.npy'))
f16 = lambda x: x.astype(np.float16).astype(np.float32)
out = {}
TARGETS = dict(symbol=(236814, 2209), number=(886, 1156))
for q, (nat, tgt) in TARGETS.items():
    d = {m: load(F, C[f'{q}_{m}']) for m in ('base', 'residual', 'sham', 'ple', 'both', 'random_ple', 'residual_random_ple', 'donor')}
    ids = C[f'{q}_base']['input_ids']
    for m in d:
        x = C[f'{q}_{m}']['input_ids']
        if m == 'donor': assert x[:5] == ids[:5] and x[6:] == ids[6:] and x[5] == 42404
        else: assert x == ids
    # replays against the archived asymmetry run
    for m in ('base', 'residual', 'donor'):
        assert np.array_equal(d[m]['logits'], load(A, CA[f'{q}_{m}'])['logits']), (q, m)
    assert np.array_equal(d['sham']['logits'], d['base']['logits'])
    # residual application, exact on row 5 only
    for m in ('residual', 'both', 'residual_random_ple'):
        x = d[m]; assert np.array_equal(x['requested_residual'], f16(R))
        assert np.array_equal(np.delete(x['l21_after'], 5, 0), np.delete(x['l21_before'], 5, 0))
        assert np.allclose(x['l21_after'][5], x['l21_before'][5] + x['requested_residual'], atol=0.02)
    # PLE: replacement by the donor's own row-5 PLE inputs, all 42 layers; sham = native's own
    donor_ple = d['donor']['ple_before']; base_ple = d['base']['ple_before']
    assert donor_ple.shape == (42, 256) and not np.array_equal(donor_ple, base_ple)
    for m in ('ple', 'both'):
        assert np.array_equal(d[m]['ple_after'], f16(donor_ple)), (q, m)
    assert np.array_equal(d['sham']['ple_after'], f16(base_ple))
    for m in ('base', 'residual', 'donor'): assert 'ple_after' not in d[m]
    # random PLE: per-layer norm matched to the donor-minus-native change
    rnd = np.load(F / f'{q}-random-ple.npy'); change = donor_ple.astype(np.float64) - base_ple.astype(np.float64)
    draw = np.random.default_rng(20261027).standard_normal(change.shape)
    expect = (draw / np.linalg.norm(draw, axis=1)[:, None] * np.linalg.norm(change, axis=1)[:, None]).astype(np.float32)
    assert np.array_equal(rnd, expect)
    for m in ('random_ple', 'residual_random_ple'):
        added = d[m]['ple_after'].astype(np.float64) - d[m]['ple_before'].astype(np.float64)
        rel = np.abs(np.linalg.norm(added, axis=1) / np.linalg.norm(change, axis=1) - 1)
        assert rel.max() <= 0.001, (q, m, rel.max())
    mg = {m: float(d[m]['logits'][tgt] - d[m]['logits'][nat]) for m in d}
    am = {m: int(d[m]['logits'].argmax()) for m in d}
    ep, epr = mg['ple'] - mg['base'], mg['both'] - mg['residual']
    rp, rr = mg['random_ple'] - mg['base'], mg['residual_random_ple'] - mg['residual']
    passed = lambda e, r: bool(e >= 1.0 and e > 2 * max(r, 0.0))
    m = s['metrics'][q]
    assert m['margins'] == mg and m['argmax'] == am
    assert (m['ple_effect'], m['conditional_ple_effect'], m['random_ple_effect'], m['conditional_random_ple_effect']) == (ep, epr, rp, rr)
    assert m['interaction'] == mg['both'] - mg['residual'] - mg['ple'] + mg['base']
    assert (m['ple_endpoint_passed'], m['conditional_endpoint_passed']) == (passed(ep, rp), passed(epr, rr))
    out[q] = dict(margins=mg, ple_effect=ep, conditional_ple_effect=epr, random_ple_effect=rp,
                  conditional_random_ple_effect=rr, interaction=m['interaction'],
                  endpoints=(passed(ep, rp), passed(epr, rr)), argmax=am)
reading = s['interpretation']
assert not any(out['number']['endpoints']) and reading.startswith('Neither')
for name in ('lithium', 'carbon'):
    for mode in ('base', 'both', 'residual_random_ple'): load(F, C[f'{name}_{mode}'])
v = dict(status='cross_checked', checker='Claude Code (different agent from the runner author, Codex)',
         basis='frozen protocol text in run_nf4_ple_factorial.py and raw artifacts; Codex checker not consulted',
         forwards=22, reading=reading, results=out)
print(json.dumps(v, indent=1))
