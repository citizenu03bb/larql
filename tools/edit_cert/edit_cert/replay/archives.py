"""Replay of archived runs (design §7): read raw artifacts, recompute every number, run the same controls.

No model is loaded. Each adapter checks artifact hashes against the archived summary, recomputes margins and
top-1s from the saved logits, compares them with the archived values (exact float equality), and returns
the evidence. Archived runs hold resumed-boundary diagnostics only, so replay can refuse, never certify.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from ..backends.activation import ActivationOneShot, ActivationStatic
from ..evidence import Boundary, Evidence
from ..spec import FactSpec, Question, Reading

DEFAULT_ROOT = Path.home() / 'Projects/larql-research/exploratory-natural-fact-edit'
H, HE, ONE, TWO, THREE = 236814, 2209, 886, 1156, 1806
Q4K_1, Q4K_2, Q4K_4 = 236770, 236778, 236812
TRIGGER = (13076, 50604, 70599, 154624)
SYMBOL_Q = "Give hydrogen's chemical symbol in one short sentence, with the element's name before the symbol."
NUMBER_Q = "Give hydrogen's atomic number in one short sentence, with the element's name before the number."
MASS_Q = "Give hydrogen's atomic mass in one short sentence, with the element's name before the value."
BEFORE = 'subject_before_relation'


@dataclass
class Case:
    id: str
    title: str
    spec: FactSpec
    form: dict
    evidence: Evidence
    edit_label: str
    reproduced: list[dict] = field(default_factory=list)

    def check(self, name, archived, recomputed):
        self.reproduced.append(dict(quantity=name, archived=archived, recomputed=recomputed,
                                    equal=archived == recomputed))


def nf4_spec(instrument='nf4-e4b') -> FactSpec:
    one = Reading(accept_ids={instrument: (ONE,)}, reject_ids={instrument: (TWO, THREE)})
    return FactSpec('hydrogen-symbol-He', ('hydrogen', 'Hydrogen'), 'chemical symbol',
                    Reading(accept=('H',), accept_ids={instrument: (H,)}),
                    Reading(accept=('He',), accept_ids={instrument: (HE,)}), ('helium',),
                    (Question('SYM', 'efficacy', BEFORE, SYMBOL_Q, seen=True),
                     Question('NUM', 'attribute', BEFORE, NUMBER_Q, expect=one),
                     Question('LI', 'off_target', 'none', SYMBOL_Q.replace('hydrogen', 'lithium')),
                     Question('C', 'off_target', 'none', SYMBOL_Q.replace('hydrogen', 'carbon'))),
                    donor='helium')


def q4k_spec() -> FactSpec:
    inst = 'larql-q4k'
    return FactSpec('hydrogen-symbol-He', ('hydrogen', 'Hydrogen'), 'chemical symbol',
                    Reading(accept=('H',)), Reading(accept=('He',)), ('helium',),
                    (Question('NUM', 'attribute', BEFORE, NUMBER_Q,
                              expect=Reading(accept_ids={inst: (Q4K_1,)}, reject_ids={inst: (Q4K_2,)})),
                     Question('MASS', 'attribute', BEFORE, MASS_Q,
                              expect=Reading(accept_ids={inst: (Q4K_1,)}, reject_ids={inst: (Q4K_4,)}))),
                    donor='helium')


def _sha(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class Run:
    """An NF4 run directory: summary.json with calls [{name, artifact, artifact_sha256}], npz with 'logits'."""

    def __init__(self, path: Path):
        self.path = Path(path)
        self.summary = json.loads((self.path / 'summary.json').read_text())
        self.calls = {c['name']: c for c in self.summary['calls']}

    def logits(self, name: str) -> np.ndarray:
        c = self.calls[name]
        f = self.path / c['artifact']
        if _sha(f) != c['artifact_sha256']:
            raise ValueError(f'{f}: hash differs from the archived summary')
        with np.load(f, allow_pickle=False) as z:
            return z['logits'].reshape(-1).copy()


def _bnd(qid, cond, logits, native_id, target_id, mention='all', partial=False, variant='', source=''):
    # float32 subtraction, as in every archived checker (exact for NF4's float16-valued logits too)
    margin = float(np.float32(logits[target_id]) - np.float32(logits[native_id])) if target_id is not None else None
    return Boundary(qid, cond, int(native_id), target_id, int(np.argmax(logits)), margin, mention, partial,
                    variant, source)


def _evidence(instrument, deadline, note):
    return Evidence(instrument, False, deadline, notes=[note])


# ---------------------------------------------------------------- R1, R3, R4 (asymmetry run)

def _asymmetry(root: Path):
    run = Run(root / 'nf4-asymmetry-20261026' / 'interventions')
    return run, run.summary['metrics']


def r1(root: Path = DEFAULT_ROOT) -> Case:
    run, m = _asymmetry(root)
    vector = np.load(run.path / 'residual.npy', allow_pickle=False)
    edit = ActivationOneShot(21, vector, [5], label='L21 entity delta, one-shot at row 5 (NF4)',
                             seen_phrasings=[SYMBOL_Q])
    ev = _evidence('nf4-e4b', 22, 'replay of nf4-asymmetry-20261026: resumed boundaries only')
    case = Case('R1', 'One-shot L21 entity delta at the native boundary (NF4)', nf4_spec(), edit.form(), ev, edit.label)
    for qid, q, native, target in (('SYM', 'symbol', H, HE), ('NUM', 'number', ONE, TWO)):
        for cond, call in (('native', 'base'), ('edit', 'residual'), ('random', 'random')):
            b = _bnd(qid, cond, run.logits(f'{q}_{call}'), native, target, source=f'{q}_{call}')
            ev.boundaries.append(b)
            if cond == 'native':
                case.check(f'{q} baseline margin', m[q]['baseline_margin'], b.margin)
            if cond == 'edit':
                case.check(f'{q} residual margin', m[q]['residual_margin'], b.margin)
                case.check(f'{q} residual argmax', m[q]['residual_argmax'], b.top1)
        gain = ev.boundaries[-2].margin - ev.boundaries[-3].margin
        rgain = ev.boundaries[-1].margin - ev.boundaries[-3].margin
        case.check(f'{q} residual increment', m[q]['residual_increment'], gain)
        case.check(f'{q} random increment', m[q]['random_increment'], rgain)
    for off in run.summary['off_target']:
        qid = {'lithium': 'LI', 'carbon': 'C'}[off['name']]
        base = run.logits(off['name'] + '_base')
        native = int(np.argmax(base))
        case.check(f"{off['name']} baseline argmax", off['baseline_argmax'], native)
        for cond, call in (('native', 'base'), ('edit', 'residual')):
            b = _bnd(qid, cond, run.logits(f"{off['name']}_{call}"), native, None, source=f"{off['name']}_{call}")
            ev.boundaries.append(b)
        case.check(f"{off['name']} residual argmax", off['residual_argmax'], ev.boundaries[-1].top1)
    ev.notes.append('off-target prefixes were supplied, not generated (asymmetry protocol)')
    return case


def r3(root: Path = DEFAULT_ROOT) -> Case:
    run, m = _asymmetry(root)
    vector = np.load(run.path / 'random.npy', allow_pickle=False)
    edit = ActivationOneShot(21, vector, [5], label='random direction of the R1 norm, one-shot (NF4)',
                             seen_phrasings=[SYMBOL_Q])
    ev = _evidence('nf4-e4b', 22, 'replay of nf4-asymmetry-20261026 random condition taken as the edit')
    case = Case('R3', 'Random direction of equal norm taken as the edit (NF4)', nf4_spec(), edit.form(), ev, edit.label)
    for qid, q, native, target in (('SYM', 'symbol', H, HE), ('NUM', 'number', ONE, TWO)):
        nb = _bnd(qid, 'native', run.logits(f'{q}_base'), native, target, source=f'{q}_base')
        eb = _bnd(qid, 'edit', run.logits(f'{q}_random'), native, target, source=f'{q}_random')
        ev.boundaries += [nb, eb]
        case.check(f'{q} random increment', m[q]['random_increment'], eb.margin - nb.margin)
    ratio = float(np.linalg.norm(vector.astype(np.float16).astype(np.float64))) / run.summary['norm_rule']['actual_norm']
    case.check('random float16 norm within 0.1% of the edit norm (protocol rule)', True, abs(ratio - 1) <= 0.001)
    return case


def r4(root: Path = DEFAULT_ROOT) -> Case:
    run, m = _asymmetry(root)
    form = dict(answer_time_trigger=False, donor_pass=True, uses_native_proposal=False,
                substitutes_subject_tokens=True, single_artifact=False, applied_every_forward=True)
    ev = _evidence('nf4-e4b', 22, 'replay of nf4-asymmetry-20261026 donor condition taken as the edit')
    case = Case('R4', 'Full helium subject substitution taken as an edit (NF4)', nf4_spec(), form, ev,
                'hydrogen token replaced by helium in the prompt')
    for qid, q, native, target in (('SYM', 'symbol', H, HE), ('NUM', 'number', ONE, TWO)):
        ev.boundaries.append(_bnd(qid, 'native', run.logits(f'{q}_base'), native, target, source=f'{q}_base'))
        b = _bnd(qid, 'edit', run.logits(f'{q}_donor'), native, target, source=f'{q}_donor')
        ev.boundaries.append(b)
        case.check(f'{q} donor argmax', m[q]['donor_argmax'], b.top1)
    return case


# ---------------------------------------------------------------- R2 (Phase 3 static + 3b row split)

def r2(root: Path = DEFAULT_ROOT) -> Case:
    calib = Run(root / 'nf4-phase3-static-20261026' / 'calib')
    rows = Run(root / 'nf4-phase3b-rows-20261026')
    cal = json.loads((root / 'nf4-phase3-static-20261026' / 'verification.json').read_text())['calibration']
    split = json.loads((root / 'nf4-phase3b-rows-20261026' / 'verification.json').read_text())['results']
    vector = np.load(root / 'nf4-asymmetry-20261026' / 'interventions' / 'residual.npy', allow_pickle=False)
    edit = ActivationStatic(21, vector, 1.0, TRIGGER, label='L21 entity delta, static on every hydrogen token, g*=1.0',
                            seen_phrasings=[SYMBOL_Q, NUMBER_Q])
    ev = _evidence('nf4-e4b', 22, 'replay of nf4-phase3-static-20261026 (calibration) and nf4-phase3b-rows-20261026')
    case = Case('R2', 'Static L21 patch at the selected gain, with its row split (NF4)', nf4_spec(), edit.form(), ev,
                edit.label)
    case.check('selected gain g*', cal['g_star'], 1.0)
    for qid, q, native, target in (('SYM', 'symbol', H, HE), ('NUM', 'number', ONE, TWO)):
        ev.boundaries.append(_bnd(qid, 'native', calib.logits(f'{q}_native'), native, target, source=f'{q}_native'))
        b = _bnd(qid, 'edit', calib.logits(f'{q}_g1.0'), native, target, variant='g1.0', source=f'{q}_g1.0')
        ev.boundaries.append(b)
        row = next(r for r in cal['rows'] if r['question'] == q and r['gain'] == 1.0)
        case.check(f'{q} g1.0 margin', row['margin'], b.margin)
        case.check(f'{q} g1.0 argmax', row['argmax'], b.top1)
        native_split = _bnd(qid, 'native', rows.logits(f'{q}_native'), native, target)
        for name, mention in (('row5', BEFORE), ('row30', 'subject_after_relation')):
            sb = _bnd(qid, 'edit', rows.logits(f'{q}_{name}'), native, target, mention, partial=True,
                      variant=name, source=f'phase3b {q}_{name}')
            ev.boundaries.append(sb)
            case.check(f'{q} {name} argmax', split[q][name]['argmax'], sb.top1)
            case.check(f'{q} {name} increment', split[q][name]['increment'], sb.margin - native_split.margin)
    others = [f"{r['question']} g{r['gain']}: top-1 {r['argmax']}" for r in cal['rows'] if r['gain'] != 1.0]
    ev.notes.append('other calibration gains (not the selected edit): ' + '; '.join(others))
    ev.notes.append('row 5 = prompt mention before the relation; row 30 = answer mention, after the question was read')
    return case


# ---------------------------------------------------------------- R5 (LARQL Q4K attribute-swap run)

def r5(root: Path = DEFAULT_ROOT) -> Case:
    d = root / 'entity-first-attribute-swap-20261025'
    v = json.loads((d / 'verification.json').read_text())
    session = json.loads((d / 'session.json').read_text())
    edit = ActivationOneShot(21, np.load(d / 'delta.npy', allow_pickle=False), [5],
                             label=f"L21 entity delta x{session['gain']:.3f}, one-shot (LARQL Q4K)",
                             seen_phrasings=[SYMBOL_Q])
    ev = _evidence('larql-q4k', 22, 'replay of entity-first-attribute-swap-20261025 (LARQL Q4K, resumed boundaries)')
    case = Case('R5', 'One-shot L21 entity delta at gain 2.585 (LARQL Q4K)', q4k_spec(), edit.form(), ev, edit.label)

    def logits(job):
        meta = json.loads((d / 'results' / f'{job}.json').read_text())
        step = meta['steps'][-1]
        with np.load(d / 'results' / step['artifact'], allow_pickle=False) as z:
            x = z['logits'].reshape(-1).copy()
        assert int(np.argmax(x)) == step['top1'], f'{job}: archived top-1 differs from its logits'
        return x

    jobs = dict(NUM=('007-atomic_number-native', '008-atomic_number-high',
                     ['009-atomic_number-random-20261025', '022-atomic_number-random-20261026']),
                MASS=('018-atomic_mass-native', '019-atomic_mass-high', []))
    for (qid, (nat, high, rands)), q in zip(jobs.items(), v['questions']):
        target = Q4K_2 if qid == 'NUM' else Q4K_4
        nb = _bnd(qid, 'native', logits(nat), Q4K_1, target, source=nat)
        hb = _bnd(qid, 'edit', logits(high), Q4K_1, target, source=high)
        ev.boundaries += [nb, hb] + [_bnd(qid, 'random', logits(r), Q4K_1, target, source=r) for r in rands]
        case.check(f'{qid} native margin', q['native_margin'], nb.margin)
        case.check(f'{qid} edited margin', q['high_margin'], hb.margin)
        for r, rec in zip(rands, q['random']):
            rb = next(b for b in ev.boundaries if b.source == r)
            case.check(f'{qid} random {rec["seed"]} increment', rec['delta_margin'], rb.margin - nb.margin)
    mass_native, mass_high = logits('018-atomic_mass-native'), logits('019-atomic_mass-high')
    rank = lambda x, t: int((x > x[t]).sum()) + 1   # noqa: E731
    case.check('mass: rank of "4" native', 48, rank(mass_native, Q4K_4))
    case.check('mass: rank of "4" edited', 4, rank(mass_high, Q4K_4))
    sym = v['options'][0]
    ev.notes.append(f"symbol under this delta (from the norm-match source run, not this archive): He-H "
                    f"{sym['baseline_he_minus_h']:.2f} -> {sym['entity_high_he_minus_h']:.2f}; "
                    f"random seed {sym['job'].split('-')[-1]}: {sym['he_minus_h']:.2f} (top-1 {sym['token']!r})")
    return case


CASES = dict(R1=r1, R2=r2, R3=r3, R4=r4, R5=r5)

# Written before the first replay (design §10). One amendment after it, recorded rather than hidden:
AMENDMENTS = [dict(case='R4', control='C3', written='NOT_EVALUATED', now='FAIL',
                   why='C3 groups whole-edit outcomes by the order of the question. The donor flips the number on '
                       'a subject-before-relation phrasing, which is a C3 failure under the rule as written; the '
                       'expectation, not the rule, was wrong.')]
EXPECTED = {
    'R1': dict(verdict='REFUSED', C0='FAIL', C1='FAIL', C2='NOT_EVALUATED', C3='NOT_EVALUATED',
               C4='NOT_EVALUATED', C6='NOT_EVALUATED', C7='NOT_EVALUATED'),
    'R2': dict(verdict='REFUSED', C0='PASS', C1='NOT_EVALUATED', C2='FAIL', C3='FAIL',
               C4='NOT_EVALUATED', C6='NOT_EVALUATED', C7='NOT_EVALUATED'),
    'R3': dict(verdict='REFUSED', C0='FAIL', C1='FAIL', C2='NOT_EVALUATED', C3='NOT_EVALUATED',
               C4='NOT_EVALUATED', C6='NOT_EVALUATED', C7='NOT_EVALUATED'),
    'R4': dict(verdict='REFUSED', C0='FAIL', C1='NOT_EVALUATED', C2='FAIL', C3='FAIL',
               C4='NOT_EVALUATED', C6='NOT_EVALUATED', C7='NOT_EVALUATED'),
    'R5': dict(verdict='REFUSED', C0='FAIL', C1='NOT_EVALUATED', C2='NOT_EVALUATED', C3='NOT_EVALUATED',
               C4='NOT_EVALUATED', C6='NOT_EVALUATED', C7='NOT_EVALUATED'),
}
