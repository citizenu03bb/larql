"""Synthetic archives with the real directory layout and planted logits, so replay adapters run without the
private archives. The planted pattern mirrors the archived findings; the numbers are invented."""
import hashlib
import json
from pathlib import Path

import numpy as np

from edit_cert.backends.base import same_norm_gaussian
from edit_cert.replay.archives import H, HE, ONE, Q4K_1, Q4K_2, Q4K_4, THREE, TWO

V = 262144


def logits(**planted):
    x = np.zeros(V, dtype=np.float32)
    for k, val in planted.items():
        x[int(k[1:])] = val
    return x


def m(x, t, n):
    return float(np.float32(x[t]) - np.float32(x[n]))


def save_run(d: Path, calls: dict, extra: dict) -> None:
    d.mkdir(parents=True, exist_ok=True)
    recs = []
    for i, (name, x) in enumerate(calls.items()):
        f = f'{i:03}-{name}.npz'
        np.savez_compressed(d / f, logits=x)
        recs.append(dict(name=name, artifact=f, artifact_sha256=hashlib.sha256((d / f).read_bytes()).hexdigest(),
                         argmax=int(np.argmax(x))))
    (d / 'summary.json').write_text(json.dumps(dict(calls=recs, **extra)))


def build(root: Path) -> Path:
    L = lambda **kw: logits(**{f't{k}': v for k, v in kw.items()})   # noqa: E731
    # --- asymmetry: symbol moves a lot but does not flip; number barely moves; random inert; donor flips both
    asym = root / 'nf4-asymmetry-20261026' / 'interventions'
    c = dict(symbol_base=L(**{str(H): 10, str(HE): -11}), symbol_residual=L(**{str(H): 10, str(HE): 8}),
             symbol_random=L(**{str(H): 10, str(HE): -10}), symbol_donor=L(**{str(H): 1, str(HE): 12}),
             number_base=L(**{str(ONE): 10, str(TWO): -8}), number_residual=L(**{str(ONE): 10, str(TWO): -7.95}),
             number_random=L(**{str(ONE): 10, str(TWO): -8.02}), number_donor=L(**{str(ONE): 2, str(TWO): 12}),
             lithium_base=L(**{'500': 5}), lithium_residual=L(**{'500': 5}), lithium_random=L(**{'500': 5}),
             carbon_base=L(**{'600': 5}), carbon_residual=L(**{'600': 5}), carbon_random=L(**{'600': 5}))
    metrics = {}
    for q, (t, n) in dict(symbol=(HE, H), number=(TWO, ONE)).items():
        b = m(c[f'{q}_base'], t, n)
        metrics[q] = dict(baseline_margin=b, residual_margin=m(c[f'{q}_residual'], t, n),
                          residual_increment=m(c[f'{q}_residual'], t, n) - b,
                          random_increment=m(c[f'{q}_random'], t, n) - b,
                          residual_argmax=int(np.argmax(c[f'{q}_residual'])), donor_argmax=int(np.argmax(c[f'{q}_donor'])))
    vec = np.linspace(-1, 1, 16).astype(np.float32)
    rnd = same_norm_gaussian(vec, 1)
    off = [dict(name=n, baseline_argmax=a, residual_argmax=a, random_argmax=a) for n, a in (('lithium', 500), ('carbon', 600))]
    save_run(asym, c, dict(metrics=metrics, off_target=off, norm_rule=dict(
        actual_norm=float(np.linalg.norm(vec.astype(np.float16).astype(np.float64))))))
    np.save(asym / 'residual.npy', vec)
    np.save(asym / 'random.npy', rnd)

    # --- Phase 3 static calibration: symbol flips, number goes to three at g=1
    p3 = root / 'nf4-phase3-static-20261026'
    cal = dict(symbol_native=c['symbol_base'], **{'symbol_g1.0': L(**{str(H): 1, str(HE): 20})},
               number_native=c['number_base'], **{'number_g1.0': L(**{str(ONE): 5, str(TWO): 4.5, str(THREE): 9})})
    save_run(p3 / 'calib', cal, {})
    rows = [dict(question=q, gain=1.0, argmax=int(np.argmax(cal[f'{q}_g1.0'])), margin=m(cal[f'{q}_g1.0'], t, n))
            for q, (t, n) in dict(symbol=(HE, H), number=(TWO, ONE)).items()]
    rows.append(dict(question='number', gain=2.0, argmax=ONE, margin=-8.0))
    (p3 / 'verification.json').write_text(json.dumps(dict(calibration=dict(g_star=1.0, rows=rows))))

    # --- Phase 3b row split: the answer mention (row 30) carries the number effect
    p3b = root / 'nf4-phase3b-rows-20261026'
    split = dict(symbol_native=c['symbol_base'], symbol_row5=c['symbol_residual'],
                 symbol_row30=L(**{str(H): 1, str(HE): 30}), number_native=c['number_base'],
                 number_row5=c['number_residual'], number_row30=L(**{str(ONE): 5, str(TWO): 3, str(THREE): 9}))
    save_run(p3b, split, {})
    res = {q: {r: dict(argmax=int(np.argmax(split[f'{q}_{r}'])),
                       increment=m(split[f'{q}_{r}'], t, n) - m(split[f'{q}_native'], t, n)) for r in ('row5', 'row30')}
           for q, (t, n) in dict(symbol=(HE, H), number=(TWO, ONE)).items()}
    (p3b / 'verification.json').write_text(json.dumps(dict(results=res)))

    # --- LARQL Q4K attribute swap: number and mass stay 1; "4" climbs from 48th to 4th in mass
    q4 = root / 'entity-first-attribute-swap-20261025'
    (q4 / 'results').mkdir(parents=True)
    filler = {str(1000 + i): 3.0 + i * 0.01 for i in range(46)}   # plus "1": 47 tokens above "4"
    jobs = {'007-atomic_number-native': L(**{str(Q4K_1): 20, str(Q4K_2): 0.5}),
            '008-atomic_number-high': L(**{str(Q4K_1): 20, str(Q4K_2): 1.25}),
            '009-atomic_number-random-20261025': L(**{str(Q4K_1): 20, str(Q4K_2): 0.25}),
            '022-atomic_number-random-20261026': L(**{str(Q4K_1): 20, str(Q4K_2): 0.5}),
            '018-atomic_mass-native': L(**{str(Q4K_1): 25, str(Q4K_4): 1.0, **filler}),
            '019-atomic_mass-high': L(**{str(Q4K_1): 25, str(Q4K_4): 6.0, '1000': 7, '1001': 7})}
    for i, (job, x) in enumerate(jobs.items()):
        np.savez_compressed(q4 / 'results' / f'{i:03}.npz', logits=x)
        (q4 / 'results' / f'{job}.json').write_text(json.dumps(dict(steps=[dict(artifact=f'{i:03}.npz', top1=int(np.argmax(x)))])))
    num = dict(native_margin=m(jobs['007-atomic_number-native'], Q4K_2, Q4K_1),
               high_margin=m(jobs['008-atomic_number-high'], Q4K_2, Q4K_1),
               random=[dict(seed=s, delta_margin=m(jobs[j], Q4K_2, Q4K_1) - m(jobs['007-atomic_number-native'], Q4K_2, Q4K_1))
                       for s, j in ((20261025, '009-atomic_number-random-20261025'), (20261026, '022-atomic_number-random-20261026'))])
    mass = dict(native_margin=m(jobs['018-atomic_mass-native'], Q4K_4, Q4K_1),
                high_margin=m(jobs['019-atomic_mass-high'], Q4K_4, Q4K_1), random=[])
    (q4 / 'verification.json').write_text(json.dumps(dict(questions=[num, mass], options=[dict(
        job='023-symbol-random-20261025', baseline_he_minus_h=-12.4, entity_high_he_minus_h=4.1, he_minus_h=-10.3, token=' H')])))
    (q4 / 'session.json').write_text(json.dumps(dict(gain=2.585)))
    np.save(q4 / 'delta.npy', vec)
    return root
