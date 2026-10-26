"""Model-free checker for run_nf4_phase3_static.py (same author as the runner: Claude Code).

Other rows being untouched is asserted inside the hook at run time; only patched rows are saved.
"""
import hashlib, json, sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_nf4_phase3_static import TRIGGER, GAINS, HE, H, ONE, TWO, ASYM, RANDOM_SEED, CAP_TOKENS, read_test

ROOT = Path(sys.argv[1])
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
f16 = lambda x: x.astype(np.float16).astype(np.float32)
V = np.load(ASYM / 'residual.npy')
report = dict(groups={})

def check_group(name):
    d = ROOT / name; s = json.loads((d / 'summary.json').read_text()); assert s['status'] == 'complete', (name, s.get('error'))
    assert len(s['calls']) == s['forwards'] <= 24 and [c['ordinal'] for c in s['calls']] == list(range(len(s['calls'])))
    arrays = {}
    for c in s['calls']:
        assert sha(d / c['artifact']) == c['artifact_sha256']
        with np.load(d / c['artifact']) as z: a = {k: z[k].copy() for k in z.files}
        assert int(a['logits'].argmax()) == c['argmax']
        rows = [i for i, t in enumerate(c['input_ids']) if t in TRIGGER]; assert list(a['rows']) == rows
        if c['gain_vector'] and rows:
            assert c['patched_rows'] == rows
            assert np.allclose(a['after'] - a['before'], a['requested'][None, :], atol=0.05)
        arrays[c['name']] = (c, a)
    return s, arrays

s, A = check_group('calib'); cal = s['calibration']
ca = {c['name']: c for c in json.loads((ASYM / 'summary.json').read_text())['calls']}
for q, base, tgt, nat in (('symbol', 'symbol_base', HE[0], H[0]), ('number', 'number_base', TWO, ONE)):
    c, a = A[f'{q}_native']; assert c['input_ids'] == ca[base]['input_ids']
    with np.load(ASYM / ca[base]['artifact']) as z: assert np.array_equal(a['logits'], z['logits'].reshape(-1))
    for g in GAINS:
        c, a = A[f'{q}_g{g}']; assert c['input_ids'] == ca[base]['input_ids']
        assert np.array_equal(a['requested'], f16((np.float32(g) * V).astype(np.float32)))
        row = next(x for x in cal['rows'] if x['question'] == q and x['gain'] == g)
        assert row['argmax'] == c['argmax'] and row['margin'] == float(a['logits'][tgt] - a['logits'][nat])
flips = [g for g in GAINS if A[f'symbol_g{g}'][0]['argmax'] == HE[0]]
g_star = min(flips) if flips else None
assert cal['g_star'] == g_star and cal['go'] == bool(g_star is not None and A[f'number_g{g_star}'][0]['argmax'] == ONE)
report['calibration'] = dict(g_star=g_star, go=cal['go'], stop_reason=cal['stop_reason'],
                             rows=[{k: x[k] for k in ('question', 'gain', 'argmax', 'margin')} for x in cal['rows']])
if cal['go']:
    vec = (np.float32(g_star) * V).astype(np.float32); results = {}
    for grp in ('T1', 'T2', 'T3', 'T4', 'T5', 'R1T6'):
        s, A = check_group(grp); assert s['g_star'] == g_star
        for key, res in s['results'].items():
            if key == 'T6':
                n, p = A['T6_native'][1], A['T6_patched'][1]
                assert res['inert'] == bool(np.array_equal(n['logits'], p['logits'])) and res['patched_rows'] == []
                results[key] = res; continue
            calls = [A[n][0] for n in A if n.startswith(key)]
            prompt = calls[0]['input_ids']; gen = []
            for c in calls:
                assert c['input_ids'] == prompt + gen; gen.append(c['argmax'])
            assert gen == res['ids'] and len(gen) <= CAP_TOKENS
            if key.endswith('_patched'):
                want = vec
                if key == 'R1_patched':
                    rnd = np.load(ROOT / grp / 'random.npy'); draw = np.random.default_rng(RANDOM_SEED).standard_normal(V.shape)
                    assert np.array_equal(rnd, (draw / np.linalg.norm(draw) * np.linalg.norm(vec.astype(np.float64))).astype(np.float32)); want = rnd
                for c in calls:
                    a = A[c['name']][1]
                    if len(a['rows']): assert np.array_equal(a['requested'], f16(want))
            results[key] = res
    reading = read_test(results)
    assert reading == json.loads((ROOT / 'reading.json').read_text())
    report['reading'] = reading
    report['texts'] = {k: v['text'] for k, v in results.items() if k != 'T6'}
    report['T6_inert'] = results['T6']['inert']
report.update(status='verified_complete', author='Claude Code (same as runner); not an independent replication')
(ROOT / 'verification.json').write_text(json.dumps(report, indent=2) + '\n'); print(json.dumps(report, indent=2))
