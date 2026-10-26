"""Model-free checker for run_nf4_h1_name_form.py (same author as the runner: Claude Code)."""
import hashlib, json, sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_nf4_h1_name_form import PREFIX, NON_PREFIX, readings, SECOND_SEED

R = Path(sys.argv[1])
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
s = json.loads((R / 'summary.json').read_text()); assert s['status'] == 'complete', s['status']
h1 = s['h1']; calls = {c['name']: c for c in s['calls']}
assert len(s['calls']) == h1['forwards'] <= 20 and [c['ordinal'] for c in s['calls']] == list(range(len(s['calls'])))
def load(name):
    c = calls[name]; assert sha(R / c['artifact']) == c['artifact_sha256']
    with np.load(R / c['artifact']) as z: return {k: z[k].copy() for k in z.files}
base = load('native'); ids = calls['native']['input_ids']
assert np.array_equal(load('zero')['logits'], base['logits'])
norm = h1['norm']
draw = np.random.default_rng(SECOND_SEED).standard_normal(2560); r2 = (draw / np.linalg.norm(draw) * norm).astype(np.float32)
assert np.array_equal(np.load(R / 'random2.npy'), r2)
def applied(data, vector):
    req = data['requested_residual']; assert np.array_equal(req, vector.astype(np.float16).astype(np.float32))
    assert np.array_equal(data['l21_after'][5], (data['l21_before'][5].astype(np.float16) + req.astype(np.float16)).astype(np.float32)) or \
        np.allclose(data['l21_after'][5], data['l21_before'][5] + req, atol=1e-2)
    assert np.array_equal(np.delete(data['l21_after'], 5, 0), np.delete(data['l21_before'], 5, 0))
rand = {k: load(k) for k in ('random1', 'random2')}
applied(rand['random2'], r2)
H = 236814
rows = []
for group, donors in (('prefix', PREFIX), ('non_prefix', NON_PREFIX)):
    for name, sym in donors:
        row = next(r for r in h1['rows'] if r['donor'] == name); t = row['token_id']
        d_ids = calls[f'{name}_donor']['input_ids']; assert d_ids[:5] == ids[:5] and d_ids[6:] == ids[6:] and d_ids[5] != ids[5]
        assert calls[f'{name}_residual']['input_ids'] == ids
        full, res = load(f'{name}_donor'), load(f'{name}_residual'); vec = np.load(R / f'residual-{name}.npy')
        if name != 'helium':
            e = (full['l21_before'][5] - base['l21_before'][5]).astype(np.float64)
            assert np.array_equal(vec, (e / np.linalg.norm(e) * norm).astype(np.float32))
        applied(res, vec)
        m = lambda d: float(d['logits'][t]) - float(d['logits'][H])
        m0, mc, mr = m(base), m(full), m(res)
        assert (m0, mc, mr) == (row['m0'], row['ceiling'], row['residual'])
        assert row['random_increment_max'] == max(m(rand['random1']) - m0, m(rand['random2']) - m0)
        assert row['eligible'] == (int(full['logits'].argmax()) == t and mc - m0 > 0)
        rows.append(dict(group=group, eligible=row['eligible'], rho=(mr - m0) / (mc - m0) if mc - m0 > 0 else None))
assert readings(rows)['reading'] == h1['readings']['reading']
v = dict(status='verified_complete', forwards=len(s['calls']), replays=h1['replays'], readings=h1['readings'],
         rows=[{k: r[k] for k in ('donor', 'symbol', 'group', 'eligible', 'rho', 'increment', 'random_increment_max', 'residual_flips', 'direction_specific')} for r in h1['rows']],
         author='Claude Code (same as runner); not an independent replication')
(R / 'verification.json').write_text(json.dumps(v, indent=2) + '\n'); print(json.dumps(v, indent=2))
