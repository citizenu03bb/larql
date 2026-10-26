"""H1 "form of the name" protocol, frozen 2026-10-26 before any model forward (Claude Code).

Question. On the verified NF4 E4B symbol boundary ("Hydrogen's chemical symbol is " + native H), a
post-L21 entity-row delta toward helium moves He-H by about +19.7 while the same vector barely moves the
atomic number. H1: the symbol read-out partly relies on the FORM of the donor's name, since "He" is how
"helium" starts. This protocol tests H1 alone; H2 (PLE) is Codex's separate line and is not touched here.

Design. Same boundary, same layer and hook as scripts/run_nf4_asymmetry.py, same norm. Eight donors, each
a single-token name aligned at position 5 and a single-token symbol:
  prefix donors (symbol = start of the English name): helium He, lithium Li, calcium Ca, nickel Ni
  non-prefix donors:                                  sodium Na, potassium K, iron Fe, copper Cu
Pairs are roughly matched by atomic number (Li/Na, Ca/K, Ni with Fe/Cu). For each donor X:
  full donor forward (question with X at position 5, same native assistant prefix) = ceiling;
  R_X = donor-minus-native post-L21 row 5, rescaled to the float16 norm of the verified helium R
  (63.8967); helium reuses that verified R exactly; one residual forward with R_X.
Shared forwards: native replay (must equal the saved bench logits bit for bit), zero residual (must equal
native), random direction of the same norm (seed 20261026, the verified random vector) and a second random
direction (seed 20261027). Budget: 20 forwards in one service (cap 24), 22G/24G/8G, 1800 s.

Metric per donor X with symbol token t_X and native token H:
  m(.) = logit(t_X) - logit(H); m0 native, mc ceiling, mR residual;
  rho_X = (mR - m0) / (mc - m0), the fraction of the full-donor shift reached by the row-5 delta.
Eligibility: ceiling argmax is t_X and mc - m0 > 0; ineligible donors are reported and excluded.
Readings, fixed before data (need at least 3 eligible donors per group):
  H1 supported   if min rho(prefix) > max rho(non-prefix) and mean rho(prefix) >= 2 * mean rho(non-prefix);
  H1 not supported (at this setting) if mean rho(non-prefix) >= 0.75 * mean rho(prefix);
  otherwise undetermined.
A donor whose larger random increment reaches half of its R increment is flagged "not direction-specific".
Secondary only: whether the R argmax becomes t_X. No directional prediction from mechanism.
Limits stated in advance: one question, one template, one boundary, four donors per group, one trial
per case; resumed boundary, not edited free generation; confounds left open (symbol-token frequency,
chemical family, name length in characters). No gain search, no new layer, no PLE change, no persistence
claim, no weights or patches written.
"""
import argparse
import json
from pathlib import Path
import subprocess

import numpy as np

PREFIX = [('helium', 'He'), ('lithium', 'Li'), ('calcium', 'Ca'), ('nickel', 'Ni')]
NON_PREFIX = [('sodium', 'Na'), ('potassium', 'K'), ('iron', 'Fe'), ('copper', 'Cu')]
SECOND_SEED = 20261027
BUDGET = 20


def readings(rows):
    ok = [r for r in rows if r['eligible']]
    pre = [r['rho'] for r in ok if r['group'] == 'prefix']
    non = [r['rho'] for r in ok if r['group'] == 'non_prefix']
    out = dict(eligible_prefix=len(pre), eligible_non_prefix=len(non))
    if len(pre) < 3 or len(non) < 3:
        out['reading'] = 'undetermined: fewer than 3 eligible donors in a group'; return out
    mp, mn = float(np.mean(pre)), float(np.mean(non))
    out.update(mean_rho_prefix=mp, mean_rho_non_prefix=mn, min_rho_prefix=min(pre), max_rho_non_prefix=max(non))
    if min(pre) > max(non) and mp >= 2 * mn:
        out['reading'] = 'H1 supported'
    elif mn >= 0.75 * mp:
        out['reading'] = 'H1 not supported at this setting'
    else:
        out['reading'] = 'undetermined'
    return out


def experiment(h, bench, asym, out, token, saved, source_symbol):
    prompt, src, saved_native = source_symbol(bench)
    assert prompt == h.prompt
    ids = src['input_ids']; H = token('H'); assert src['argmax'] == H
    R_he = np.load(asym / 'residual.npy'); random1 = np.load(asym / 'random.npy')
    norm = json.loads((asym / 'summary.json').read_text())['norm_rule']['actual_norm']
    draw = np.random.default_rng(SECOND_SEED).standard_normal(R_he.shape)
    random2 = (draw / np.linalg.norm(draw) * norm).astype(np.float32)
    _, r_saved = saved(asym, 'symbol_residual'); _, d_saved = saved(asym, 'symbol_donor'); _, x_saved = saved(asym, 'symbol_random')
    budget = dict(cap=BUDGET, used=0)

    def fwd(name, x, residual=None):
        assert budget['used'] < BUDGET; budget['used'] += 1
        return h.forward(name, x, residual=residual)

    _, base = fwd('native', ids)
    assert np.array_equal(base['logits'], saved_native['logits'].reshape(-1)), 'native replay differs'
    _, zero = fwd('zero', ids, residual=np.zeros_like(R_he)); assert np.array_equal(zero['logits'], base['logits'])
    _, rand1 = fwd('random1', ids, residual=random1)
    replay = dict(random1_bit_equal=bool(np.array_equal(rand1['logits'], x_saved['logits'])))
    _, rand2 = fwd('random2', ids, residual=random2)
    rows = []
    for group, donors in (('prefix', PREFIX), ('non_prefix', NON_PREFIX)):
        for name, sym in donors:
            t = token(sym); donor = list(ids); donor[5] = token(' ' + name)
            assert donor[:5] == ids[:5] and donor[6:] == ids[6:]
            _, full = fwd(f'{name}_donor', donor)
            if name == 'helium':
                R = R_he; replay['helium_donor_bit_equal'] = bool(np.array_equal(full['logits'], d_saved['logits']))
            else:
                e = (full['l21_before'][5] - base['l21_before'][5]).astype(np.float64)
                R = (e / np.linalg.norm(e) * norm).astype(np.float32)
                cast = float(np.linalg.norm(R.astype(np.float16).astype(np.float64)))
                assert abs(cast / norm - 1) <= 0.001
            np.save(out / f'residual-{name}.npy', R)
            _, res = fwd(f'{name}_residual', ids, residual=R)
            if name == 'helium': replay['helium_residual_bit_equal'] = bool(np.array_equal(res['logits'], r_saved['logits']))
            m = lambda d: float(d['logits'][t]) - float(d['logits'][H])
            m0, mc, mr = m(base), m(full), m(res)
            ri = max(m(rand1) - m0, m(rand2) - m0)
            eligible = int(full['logits'].argmax()) == t and mc - m0 > 0
            rows.append(dict(donor=name, symbol=sym, group=group, token_id=t, m0=m0, ceiling=mc, residual=mr,
                             increment=mr - m0, random_increment_max=ri, eligible=bool(eligible),
                             rho=(mr - m0) / (mc - m0) if mc - m0 > 0 else None,
                             residual_argmax=int(res['logits'].argmax()), residual_flips=int(res['logits'].argmax()) == t,
                             ceiling_argmax=int(full['logits'].argmax()),
                             direction_specific=bool(ri < 0.5 * (mr - m0)) if mr - m0 > 0 else False))
    np.save(out / 'random2.npy', random2)
    return dict(rows=rows, replays=replay, readings=readings(rows), forwards=budget['used'], norm=norm)


def fake():
    """Synthetic stand-ins for a model-free dry run. Never evidence."""
    class FakeHarness:
        def __init__(self):
            self.prompt = list(range(30)); self.calls = 0
        def forward(self, name, ids, residual=None):
            self.calls += 1
            rng = np.random.default_rng(abs(hash(tuple(ids))) % (2**32))
            l21 = rng.standard_normal((len(ids), 2560)).astype(np.float32)
            logits = rng.standard_normal(1000).astype(np.float32); logits[1] = 5
            if residual is not None and np.any(residual):
                logits[2:12] += float(np.linalg.norm(residual)) / 20
            if ids[5] != 5: logits[ids[5] % 10 + 2] += 30
            return dict(name=name), dict(logits=logits, l21_before=l21)
    return FakeHarness()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('destination', type=Path)
    p.add_argument('--bench-source', type=Path, default=Path('/home/pat/Projects/larql-research/exploratory-natural-fact-edit/nf4-local-bench-20261026/e4b'))
    p.add_argument('--asym-source', type=Path, default=Path('/home/pat/Projects/larql-research/exploratory-natural-fact-edit/nf4-asymmetry-20261026/interventions'))
    p.add_argument('--model-root', type=Path, default=Path('/mnt/models/hf-models'))
    p.add_argument('--dry-run', action='store_true')
    a = p.parse_args()
    if a.dry_run:
        a.destination.mkdir(parents=True, exist_ok=False)
        h = fake(); names = {n: i + 2 for i, (n, _) in enumerate(PREFIX + NON_PREFIX)}
        syms = {s: names[n] for n, s in PREFIX + NON_PREFIX}
        def token(text): return {'H': 1, **syms, **{' ' + n: v + 100 for n, v in names.items()}}[text]
        asym = a.destination / 'asym'; asym.mkdir()
        rng = np.random.default_rng(0); R = rng.standard_normal(2560).astype(np.float32); R *= 63.9 / np.linalg.norm(R)
        np.save(asym / 'residual.npy', R); np.save(asym / 'random.npy', -R)
        (asym / 'summary.json').write_text(json.dumps(dict(norm_rule=dict(actual_norm=float(np.linalg.norm(R.astype(np.float16).astype(np.float64)))))))
        ids = list(range(36)); src = dict(input_ids=ids, argmax=1)
        base_logits = h.forward('x', ids)[1]['logits']
        def source_symbol(_): return h.prompt, src, dict(logits=base_logits)
        def saved(_, name): return None, dict(logits=h.forward('x', ids, residual=R if name != 'symbol_random' else -R)[1]['logits'])
        # the fake donor ids are offset by +100 and land in index ids[5]%10+2; identical prompts give identical logits
        result = experiment(h, None, asym, a.destination, token, saved, source_symbol)
        (a.destination / 'summary.json').write_text(json.dumps(dict(synthetic=True, **result), indent=2, default=float))
        print('SYNTHETIC dry run:', result['forwards'], 'forwards;', result['readings']['reading'])
        return
    import run_nf4_asymmetry as asymmod
    from run_nf4_local_bench import sha, write
    h = asymmod.Harness(a.destination, a.model_root, 'interventions')
    h.summary.update(protocol=__doc__, runner_sha256=sha(Path(__file__)), harness_protocol='run_nf4_asymmetry.Harness (hooks reused, Codex)')
    write(a.destination / 'summary.json', h.summary)
    try:
        result = experiment(h, a.bench_source, a.asym_source, a.destination, h.token, asymmod.saved_call, asymmod.source_symbol)
        h.summary.update(status='complete', h1=result,
                         source_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=Path(__file__).resolve().parent, text=True).strip())
    except Exception as exc:
        h.finish(exc); raise
    else:
        h.finish()


if __name__ == '__main__':
    main()
