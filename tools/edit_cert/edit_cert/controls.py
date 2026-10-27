"""Controls C0..C8: pure functions over (spec, backend form, evidence).

Evidence rule, fixed in the design (§3): a free generation from an empty answer can PASS or FAIL a
question; a resumed-boundary diagnostic can only FAIL it. A control PASSes only when every question it
needs was evaluated by generation and passed, and its minimum counts are met.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from .evidence import Boundary, Evidence, Generation, contains, has_swap
from .spec import FactSpec, Question

PASS, FAIL, NOT_EVALUATED = 'PASS', 'FAIL', 'NOT_EVALUATED'
BEFORE, AFTER = 'subject_before_relation', 'subject_after_relation'
MANDATORY = ('C0', 'C1', 'C2', 'C3', 'C4', 'C6', 'C7')
NAMES = dict(C0='persistent form', C1='efficacy on unseen phrasings', C2='other attributes of the entity',
             C3='mentions before and after the relation', C4='random direction of equal norm',
             C5='donor ceiling (phrasing qualification)', C6='off-target questions',
             C7='persistence after reload', C8='propagation (reported)')
FORM_FALSE = ('answer_time_trigger', 'donor_pass', 'uses_native_proposal', 'substitutes_subject_tokens')
FORM_TRUE = ('single_artifact', 'applied_every_forward')


@dataclass
class Outcome:
    """One question under one control."""
    question_id: str
    result: str               # 'pass' | 'fail' | 'unevaluated'
    basis: str                # 'generation' | 'boundary' | 'none'
    why: str
    order: str = ''


@dataclass
class ControlResult:
    id: str
    status: str
    basis: str
    detail: str
    outcomes: list[Outcome] = field(default_factory=list)
    data: dict = field(default_factory=dict)

    @property
    def name(self) -> str:
        return NAMES[self.id]

    @property
    def mandatory(self) -> bool:
        return self.id in MANDATORY


# ---------------------------------------------------------------- native qualification

def qualify(spec: FactSpec, ev: Evidence) -> dict[str, tuple[str, str]]:
    """qid -> (level, why). level: 'generation' (usable to pass or fail), 'boundary' (fail only), 'excluded'."""
    out = {}
    for q in spec.questions:
        gen = ev.generation(q.id, 'native')
        if gen is not None:
            ok, why = _native_ok(spec, ev.instrument, q, gen)
            out[q.id] = ('generation', 'native correct') if ok else ('excluded', why)
            continue
        natives = ev.boundaries_for(q.id, 'native')
        if natives and all(b.top1 == b.native_id for b in natives):
            out[q.id] = ('boundary', 'native top-1 at the resumed boundary')
        elif natives:
            out[q.id] = ('excluded', 'native top-1 differs from the native value at the boundary')
        else:
            out[q.id] = ('excluded', 'no native evidence')
    return out


def _native_ok(spec, inst, q: Question, gen: Generation) -> tuple[bool, str]:
    if gen.capped or not gen.eos:
        return False, 'native answer truncated at the cap'
    if q.control == 'efficacy':
        if not contains(spec.old, gen, inst) or contains(spec.new, gen, inst):
            return False, 'native answer does not state the old value'
    if q.control == 'attribute':
        if not contains(q.expect, gen, inst) or contains(q.expect, gen, inst, 'reject'):
            return False, 'native answer does not state the expected attribute'
    if q.control in ('efficacy', 'attribute', 'answer_only') and has_swap(spec.swap_markers, gen):
        return False, 'native answer already contains a swap marker'
    if q.control == 'answer_only' and contains(q.expect, gen, inst, 'reject'):
        return False, 'native answer already contains a rejected value'
    if q.control in ('answer_only', 'off_target') and _declares(q.expect) and not contains(q.expect, gen, inst):
        return False, 'native answer does not state the declared value'
    if q.control == 'off_target' and _declares(q.expect) and contains(q.expect, gen, inst, 'reject'):
        return False, 'native answer already states a rejected value'
    return True, ''


# ---------------------------------------------------------------- per-question readings

def _edited_ok(spec, ev, q: Question, gen: Generation, native: Generation | None) -> tuple[bool | None, str]:
    """True/False, or None when the edited answer is unreadable (cap hit)."""
    inst = ev.instrument
    if q.control == 'off_target' and _declares(q.expect):
        # B2 v2 (design §13): judge the declared value, not the wording
        if gen.capped or not gen.eos:
            return None, 'edited answer truncated at the cap'
        if contains(q.expect, gen, inst, 'reject'):
            return False, 'states the edited value (bleed to a neighbour)'
        return (True, 'declared value kept') if contains(q.expect, gen, inst) else (False, 'declared value lost')
    if q.control == 'off_target':
        if native is None:
            return None, 'no native generation to compare'
        return (gen.ids == native.ids), ('identical to native' if gen.ids == native.ids else 'answer differs from native')
    if gen.capped or not gen.eos:
        return None, 'edited answer truncated at the cap'
    if q.control in ('efficacy', 'attribute', 'answer_only') and has_swap(spec.swap_markers, gen):
        return False, 'swap marker in the answer'
    if q.control == 'efficacy':
        if contains(spec.old, gen, inst):
            return False, 'old value still stated'
        return (True, 'new value stated') if contains(spec.new, gen, inst) else (False, 'new value absent')
    if q.control == 'attribute':
        if contains(q.expect, gen, inst, 'reject'):
            return False, 'rejected value stated'
        return (True, 'native value kept') if contains(q.expect, gen, inst) else (False, 'native value lost')
    if q.control == 'propagation':
        return (True, 'expected value stated') if contains(q.expect, gen, inst) else (False, 'expected value absent')
    if q.control == 'answer_only' and contains(q.expect, gen, inst, 'reject'):
        return False, 'rejected value in an answer that only mentions the subject'
    if q.control == 'answer_only' and _declares(q.expect) and not contains(q.expect, gen, inst):
        return False, 'the subject is no longer named (entity substitution)'
    return True, 'no swap marker'


def _declares(reading) -> bool:
    return bool(reading.accept or reading.accept_ids)


def _boundary_fails(q: Question, b: Boundary) -> str | None:
    if q.control == 'efficacy' and b.target_id is not None and b.top1 != b.target_id:
        return f'no flip at the resumed boundary (top-1 {b.top1}, target {b.target_id})'
    if q.control in ('attribute', 'off_target') and b.top1 != b.native_id:
        return f'top-1 moved from {b.native_id} to {b.top1} at the resumed boundary'
    return None


def _evaluate(spec, ev, quals, q: Question, condition='edit', partial=False) -> list[Outcome]:
    """partial=False: the full edit (generation, then whole-edit boundaries). partial=True: row-split
    boundaries only, ordered by the mention the split acted on."""
    level, why = quals[q.id]
    if level == 'excluded':
        return [] if partial else [Outcome(q.id, 'unevaluated', 'none', f'excluded: {why}', q.order)]
    outcomes = []
    gen = ev.generation(q.id, condition) if level == 'generation' and not partial else None
    if gen is not None:
        ok, reason = _edited_ok(spec, ev, q, gen, ev.generation(q.id, 'native'))
        outcomes.append(Outcome(q.id, {True: 'pass', False: 'fail', None: 'unevaluated'}[ok], 'generation', reason, q.order))
    for b in ev.boundaries_for(q.id, condition):
        if b.partial != partial:
            continue
        # A generation, when present, decides; the whole-edit boundary is then a diagnostic only.
        failure = None if gen is not None else _boundary_fails(q, b)
        order = b.mention_order if partial else q.order
        tag = f' [{b.variant or b.source}]' if (b.variant or b.source) else ''
        outcomes.append(Outcome(q.id, 'fail' if failure else 'unevaluated', 'boundary',
                                (failure or 'boundary diagnostic cannot pass a control') + tag, order))
    if not outcomes and not partial:
        outcomes.append(Outcome(q.id, 'unevaluated', 'none', f'no {condition} evidence', q.order))
    return outcomes


def _aggregate(cid, outcomes, required, enough: bool, detail_ok: str) -> ControlResult:
    """FAIL if any outcome fails; PASS if every required question passed by generation and `enough`."""
    fails = [o for o in outcomes if o.result == 'fail']
    if fails:
        basis = 'generation' if any(o.basis == 'generation' for o in fails) else 'boundary'
        return ControlResult(cid, FAIL, basis, '; '.join(f'{o.question_id}: {o.why}' for o in fails), outcomes)
    passed = {o.question_id for o in outcomes if o.result == 'pass'}
    if required and set(required) <= passed and enough:
        return ControlResult(cid, PASS, 'generation', detail_ok, outcomes)
    missing = [o for o in outcomes if o.result != 'pass']
    why = '; '.join(f'{o.question_id}: {o.why}' for o in missing) or 'no qualified question'
    if required and set(required) <= passed and not enough:
        why = 'minimum question count or order coverage not met'
    return ControlResult(cid, NOT_EVALUATED, 'none', why, outcomes)


# ---------------------------------------------------------------- the controls

def c0_form(form: dict, ev: Evidence) -> ControlResult:
    missing = [k for k in FORM_FALSE + FORM_TRUE if k not in form]
    bad = [k for k in FORM_FALSE if form.get(k) is True] + [k for k in FORM_TRUE if form.get(k) is False]
    layer = form.get('max_write_layer')
    data = dict(form=form, kv_write_deadline=ev.kv_write_deadline)
    if layer is not None and ev.kv_write_deadline is not None and layer > ev.kv_write_deadline:
        bad.append(f'writes at block {layer}, after the K/V write deadline {ev.kv_write_deadline}')
    if bad:
        return ControlResult('C0', FAIL, 'form', '; '.join(bad), data=data)
    if missing:
        return ControlResult('C0', NOT_EVALUATED, 'none', 'form does not declare ' + ', '.join(missing), data=data)
    if layer is not None and ev.kv_write_deadline is None:
        return ControlResult('C0', NOT_EVALUATED, 'none', 'instrument K/V write deadline unknown', data=data)
    return ControlResult('C0', PASS, 'form', 'static, stored, no answer-time trigger or donor', data=data)


def donor_unsuitable(spec, ev) -> dict[str, str]:
    out = {}
    for q in spec.by_control('efficacy'):
        if q.seen:          # C5 qualifies C1 phrasings; seen phrasings never enter C1
            continue
        for b in ev.boundaries_for(q.id, 'donor'):
            if b.target_id is not None and b.top1 != b.target_id:
                out[q.id] = f'donor ceiling does not flip (top-1 {b.top1})'
    return out


def efficacy_set(spec, ev) -> list[Question]:
    bad = donor_unsuitable(spec, ev)
    return [q for q in spec.by_control('efficacy') if not q.seen and q.id not in bad]


def c1_efficacy(spec, ev, quals) -> ControlResult:
    qs = efficacy_set(spec, ev)
    outcomes = [o for q in qs for o in _evaluate(spec, ev, quals, q)]
    # A phrasing used to build or calibrate the edit can refuse it (it fails even there), never certify it.
    for q in spec.by_control('efficacy'):
        if q.seen:
            outcomes += [o if o.result == 'fail' else Outcome(q.id, 'unevaluated', o.basis, 'seen phrasing cannot pass', q.order)
                         for o in _evaluate(spec, ev, quals, q)]
    required = [q.id for q in qs if quals[q.id][0] == 'generation']
    passed_orders = {o.order for o in outcomes if o.result == 'pass'}
    enough = len(required) >= spec.min_efficacy and {BEFORE, AFTER} <= passed_orders
    res = _aggregate('C1', outcomes, required, enough, f'{len(required)} unseen phrasings, both subject orders')
    res.data['donor_unsuitable'] = donor_unsuitable(spec, ev)
    return res


def c2_attributes(spec, ev, quals) -> ControlResult:
    qs = spec.by_control('attribute')
    outcomes = [o for q in qs for o in _evaluate(spec, ev, quals, q)]
    outcomes += [o for q in qs for o in _evaluate(spec, ev, quals, q, partial=True)]
    required = [q.id for q in qs if quals[q.id][0] == 'generation']
    # Amendment (Pat, 3671139b; prospective from B4): a qualified attribute in each subject order is required.
    orders = {q.order for q in qs if q.id in required}
    enough = {BEFORE, AFTER} <= orders
    return _aggregate('C2', outcomes, required, enough, f'{len(required)} attributes kept, both subject orders')


def c3_mentions(spec, ev, quals) -> ControlResult:
    eff = {q.id for q in efficacy_set(spec, ev)}
    qs = [q for q in spec.questions if q.id in eff or q.control in ('attribute', 'answer_only')]
    outcomes = []
    for q in qs:
        outcomes += _evaluate(spec, ev, quals, q)
        if q.control == 'attribute':   # a row split informs preservation only: one mention breaking an attribute
            outcomes += _evaluate(spec, ev, quals, q, partial=True)
    groups = {}
    for o in outcomes:
        groups.setdefault(o.order, []).append(o)
    fails = [o for o in outcomes if o.result == 'fail']
    if fails:
        orders = sorted({o.order for o in fails})
        basis = 'generation' if any(o.basis == 'generation' for o in fails) else 'boundary'
        return ControlResult('C3', FAIL, basis, 'fails when the mention is ' + ', '.join(orders) + ': '
                             + '; '.join(f'{o.question_id}: {o.why}' for o in fails), outcomes)
    control_of = {q.id: q.control for q in qs}
    ok = True
    for order in (BEFORE, AFTER):
        passed = {control_of[o.question_id] for o in groups.get(order, []) if o.result == 'pass'}
        ok &= {'efficacy', 'attribute'} <= passed
    answer_only = [q.id for q in qs if q.control == 'answer_only' and quals[q.id][0] == 'generation']
    ok &= bool(answer_only) and all(any(o.question_id == a and o.result == 'pass' for o in outcomes) for a in answer_only)
    required = [q.id for q in qs if quals[q.id][0] == 'generation']
    return _aggregate('C3', outcomes, required, ok, 'efficacy and attributes hold in both orders; answer-only clean')


def c4_random(spec, ev, quals) -> ControlResult:
    qs = efficacy_set(spec, ev)
    outcomes, data = [], {}
    for q in qs:
        if quals[q.id][0] == 'excluded':
            continue
        gen = ev.generation(q.id, 'random') if quals[q.id][0] == 'generation' else None
        if gen is not None:
            emits = contains(spec.new, gen, ev.instrument) and not contains(spec.old, gen, ev.instrument)
            outcomes.append(Outcome(q.id, 'fail' if emits else 'pass', 'generation',
                                    'random direction states the new value' if emits else 'random direction inert', q.order))
        for b in ([] if gen is not None else ev.boundaries_for(q.id, 'random')):
            if b.target_id is not None and b.top1 == b.target_id:
                outcomes.append(Outcome(q.id, 'fail', 'boundary', 'random direction flips at the boundary', q.order))
        ratio = _gain_ratio(ev, q.id)
        if ratio is not None:
            data[q.id] = ratio
            if not ratio['edit_gain'] > 2 * max(ratio['random_gain'], 0.0):
                outcomes.append(Outcome(q.id, 'fail', 'boundary', 'edit gain not above twice the random gain', q.order))
    fails = [o for o in outcomes if o.result == 'fail']
    if fails:
        basis = 'generation' if any(o.basis == 'generation' for o in fails) else 'boundary'
        return ControlResult('C4', FAIL, basis, '; '.join(f'{o.question_id}: {o.why}' for o in fails), outcomes, data)
    required = [q.id for q in qs if quals[q.id][0] == 'generation']
    passed = {o.question_id for o in outcomes if o.result == 'pass'}
    if required and set(required) <= passed and data:
        return ControlResult('C4', PASS, 'generation', 'random inert; edit gain > 2x random gain', outcomes, data)
    return ControlResult('C4', NOT_EVALUATED, 'none',
                         'random generations or boundary gain ratio missing', outcomes, data)


def _gain_ratio(ev, qid) -> dict | None:
    native = [b for b in ev.boundaries_for(qid, 'native') if b.margin is not None]
    edit = [b for b in ev.boundaries_for(qid, 'edit') if b.margin is not None and not b.partial]
    rand = [b for b in ev.boundaries_for(qid, 'random') if b.margin is not None]
    if not (native and edit and rand):
        return None
    base = native[0].margin
    return dict(native_margin=base, edit_gain=edit[0].margin - base,
                random_gain=max(b.margin - base for b in rand))


def c5_donor(spec, ev) -> ControlResult:
    eff = spec.by_control('efficacy')
    tested = [q.id for q in eff if not q.seen and ev.boundaries_for(q.id, 'donor')]
    observed = {q.id: [b.top1 for b in ev.boundaries_for(q.id, 'donor')] for q in eff
                if q.seen and ev.boundaries_for(q.id, 'donor')}
    bad = donor_unsuitable(spec, ev)
    if not tested:
        detail = ('donor observed only on seen phrasings (reported, cannot qualify C1): '
                  + ', '.join(f'{k} top-1 {v}' for k, v in observed.items())) if observed else 'no donor evidence'
        return ControlResult('C5', NOT_EVALUATED, 'none', detail, data=dict(observed_on_seen=observed))
    if bad:
        return ControlResult('C5', FAIL, 'boundary', 'unsuitable phrasings left C1: ' + ', '.join(bad), data=bad)
    return ControlResult('C5', PASS, 'boundary', f'ceiling flips on {len(tested)} phrasings')


def c6_off_target(spec, ev, quals, form) -> ControlResult:
    qs = spec.by_control('off_target')
    outcomes = [o for q in qs for o in _evaluate(spec, ev, quals, q)]
    if form.get('trigger_token_ids'):
        for q in qs:
            gen, nat = ev.generation(q.id, 'edit'), ev.generation(q.id, 'native')
            if gen and nat and gen.trigger_in_prompt is False and gen.first_logits_sha256 != nat.first_logits_sha256:
                outcomes.append(Outcome(q.id, 'fail', 'generation',
                                        'first forward not bit-identical although no trigger token is present', q.order))
    required = [q.id for q in qs if quals[q.id][0] == 'generation']
    return _aggregate('C6', outcomes, required, len(required) >= 1, f'{len(required)} off-target answers unchanged')


def c7_reload(spec, ev, quals) -> ControlResult:
    qs = [q for q in efficacy_set(spec, ev) if quals[q.id][0] == 'generation']
    sha_edit, sha_reload = ev.artifact_sha256.get('edit'), ev.artifact_sha256.get('reloaded')
    if sha_edit and sha_reload and sha_edit != sha_reload:
        return ControlResult('C7', FAIL, 'generation', 'artifact hash changed between save and reload')
    outcomes = []
    for q in qs:
        a, b = ev.generation(q.id, 'edit'), ev.generation(q.id, 'reloaded')
        if a is None or b is None:
            outcomes.append(Outcome(q.id, 'unevaluated', 'none', 'missing in-session or reloaded answer', q.order))
        elif a.capped or not a.eos or b.capped or not b.eos:
            # design §14: a cap hit is unreadable, not a pass, in C7 as everywhere else
            outcomes.append(Outcome(q.id, 'unevaluated', 'none', 'in-session or reloaded answer truncated at the cap', q.order))
        else:
            same = a.ids == b.ids
            outcomes.append(Outcome(q.id, 'pass' if same else 'fail', 'generation',
                                    'reloaded answer identical' if same else 'reloaded answer differs', q.order))
    enough = bool(sha_edit and sha_reload)
    return _aggregate('C7', outcomes, [q.id for q in qs], enough, 'answers identical after a fresh load')


def c8_propagation(spec, ev, quals) -> ControlResult:
    qs = spec.by_control('propagation')
    outcomes = [o for q in qs for o in _evaluate(spec, ev, quals, q)]
    required = [q.id for q in qs if quals[q.id][0] == 'generation']
    return _aggregate('C8', outcomes, required, len(required) >= 1, 'edit propagates to the probes')


def run_all(spec: FactSpec, form: dict, ev: Evidence) -> list[ControlResult]:
    quals = qualify(spec, ev)
    return [c0_form(form, ev), c1_efficacy(spec, ev, quals), c2_attributes(spec, ev, quals),
            c3_mentions(spec, ev, quals), c4_random(spec, ev, quals), c5_donor(spec, ev),
            c6_off_target(spec, ev, quals, form), c7_reload(spec, ev, quals), c8_propagation(spec, ev, quals)]
