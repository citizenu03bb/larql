"""Entity-swap test: does the L21 hydrogen-row delta that yields He also make hydrogen's other attributes helium's?

Preregistered 2026-10-25, before any forward (revised after a stub dry-run and a protocol review):
- Vector: delta[5] from entity-first-local-contrast-20261023 (helium minus hydrogen at the ' hydrogen' row, L21).
  Both attribute prompts share their first six token ids with the symbol prompt, so under causal attention the
  hydrogen row at L21 sees only "Give hydrogen": the vector carries no relation information.
- Procedure as in the verified He result (norm-match entity_high): generate the native answer from an empty
  answer, then at the native proposal of the first value digit apply gain*delta[5] once at L21 row 5, then release.
- Gain: float32(prompt_norm/entity_norm) = 2.58510, the gain that gave He on the symbol prompt.
- Questions: atomic number (hydrogen 1, helium 2) and atomic mass (hydrogen 1.008, helium 4.0026).
- Primary metric: Delta = margin(patched) - margin(native) at the same boundary, where
  margin = logit(helium digit) - logit(hydrogen digit). Secondary: top-1.
- Random control: an isotropic Gaussian direction (seed 20261025) of the same norm, applied at row 5 at the
  atomic-number boundary. A second seed and the symbol-prompt boundary are optional if budget remains.
- Readings, fixed in advance:
  * atomic number counts as swapped only if the helium digit is top-1 AND Delta(delta) > 2 * max(Delta(random), 0);
  * atomic mass (4 is not a generic neighbour of 1) counts as swapped if the helium digit is top-1;
  * both swapped -> entity swap; mass only or number only -> partial swap, reported as such;
  * neither -> "selective at this amplitude, mechanism undetermined", never "fact edit"; a formatting token as
    top-1 is an undetermined outcome.
- Ceiling control, only for a question whose patched top-1 is not the helium digit: the same boundary with the
  donor prompt (' helium' at position 5) and the same native seed, to check that a full helium subject would
  give the helium digit at all.
- Prediction (mechanistic): entity swap on both questions.
- Budget: 24 forwards (hard cap of the frozen binary), one worker, MemoryHigh=22G/MemoryMax=24G/MemorySwapMax=8G,
  30 minutes. Core first: Q1 native, Q1 high, Q1 random, Q2 native, Q2 high; then ceiling controls; then options.
"""
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
from tokenizers import Tokenizer

QUESTIONS = [
    dict(name='atomic_number', text="Give hydrogen's atomic number in one short sentence, with the element's name before the number.",
         hydrogen_digit='1', helium_digit='2', native_cap=12),
    dict(name='atomic_mass', text="Give hydrogen's atomic mass in one short sentence, with the element's name before the value.",
         hydrogen_digit='1', helium_digit='4', native_cap=12),
]
RANDOM_SEEDS = [20261025, 20261026]
BUDGET = 24
EOS = [1, 106]
DIGITS = set('0123456789')
HELIUM_ID = 42404


def read(p): return json.loads(p.read_text())
def write(p, x): p.write_text(json.dumps(x, indent=2) + '\n')
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()


def main(destination, repo_path):
    root, repo = Path(destination), Path(repo_path)
    local = root.parent / 'entity-first-local-contrast-20261023'
    norm = root.parent / 'entity-first-norm-match-20261023'
    for source in [local, norm]:
        assert read(source / 'verification.json')['status'] == 'verified_complete'
    nm = read(norm / 'session.json'); nms = read(norm / 'summary.json')
    binary = nm['binary']; assert sha(Path(binary)) == nm['binary_sha256']
    lv = read(local / 'summary.json')['variants'][0]
    contrast = local / lv['contrast']; assert sha(contrast) == nm['contrast_sha256']
    z = dict(np.load(contrast)); delta = z['delta']; assert np.array_equal(delta, z['donor'] - z['baseline'])
    position = nm['position']; assert position == 5
    entity_high = next(c for c in nm['conditions'] if c['name'] == 'entity_high')
    high_gain = entity_high['gain']; target_norm = entity_high['injected_norm']
    nm_base_step = read(norm / 'results' / f"{nms['baseline_job']}.json")['steps'][0]
    nm_base = dict(np.load(norm / 'results' / nm_base_step['artifact']))
    nm_entity_high = next(c for c in nms['conditions'] if c['name'] == 'entity_high')

    vindex = '/mnt/models/larql-vindexes/gemma-4-e4b-upstream-20260811-q4k.vindex'
    tok = Tokenizer.from_file(vindex + '/tokenizer.json')
    symbol_prompt = nm['prompt_ids']; head, tail = symbol_prompt[:4], symbol_prompt[-5:]
    assert tok.decode([symbol_prompt[position]]) == ' hydrogen' and tok.decode([HELIUM_ID]) == ' helium'
    questions = []
    for q in QUESTIONS:
        ids = head + tok.encode(q['text'], add_special_tokens=False).ids + tail
        assert ids[:position + 1] == symbol_prompt[:position + 1], q['name']
        digit = {d: tok.encode(d, add_special_tokens=False).ids for d in [q['hydrogen_digit'], q['helium_digit']]}
        assert all(len(v) == 1 for v in digit.values())
        donor = list(ids); donor[position] = HELIUM_ID
        questions.append(dict(q, prompt_ids=ids, donor_prompt_ids=donor,
                              hydrogen_id=digit[q['hydrogen_digit']][0], helium_id=digit[q['helium_digit']][0]))

    randoms = []
    for seed in RANDOM_SEEDS:
        v = np.random.default_rng(seed).standard_normal(delta.shape[1])
        v = (v / np.linalg.norm(v) * target_norm).astype(np.float32)
        randoms.append(dict(seed=seed, vector=v, norm=float(np.linalg.norm(v.astype('float64')))))
        assert abs(randoms[-1]['norm'] / target_norm - 1) < 1e-6
    effect_norm = float(np.linalg.norm((np.float32(high_gain) * delta[position]).astype('float64')))
    assert abs(effect_norm / target_norm - 1) < 1e-6

    (root / 'jobs').mkdir(parents=True, exist_ok=False)
    np.save(root / 'delta.npy', delta)
    np.save(root / 'random-directions.npy', np.stack([r['vector'] for r in randoms]))
    write(root / 'session.json', dict(
        start=time.time(), source_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=repo, text=True).strip(),
        binary=binary, binary_sha256=nm['binary_sha256'], contrast=str(contrast), contrast_sha256=nm['contrast_sha256'],
        norm_match_source=str(norm), position=position, gain=high_gain, target_norm=target_norm,
        questions=questions, random_seeds=RANDOM_SEEDS, random_norms=[r['norm'] for r in randoms],
        symbol_prompt_ids=symbol_prompt, symbol_seed_ids=nm['seed_ids'],
        max_forwards=BUDGET, max_runtime_seconds=1800, preregistration=__doc__))

    child = subprocess.Popen([sys.executable, '-u', str(repo / 'scripts/explore_natural_fact_edit.py'), str(root), binary, vindex, str(BUDGET)])
    count = 0; jobs = 0; deadline = time.monotonic() + 1740
    summary = dict(status='running', questions=[], random_control=[], options=[])

    def progress(): write(root / 'progress.json', summary)

    def run(label, prompt, seed, max_new, spec=None):
        nonlocal count, jobs
        assert count + max_new <= BUDGET or max_new == 1 and count < BUDGET
        name = f'{jobs:03}-{label}'; jobs += 1
        job = dict(prompt_ids=prompt, seed_ids=list(seed), max_new_tokens=max_new)
        if spec is not None: job['intervention'] = spec
        pending = root / 'jobs' / f'{name}.pending'; write(pending, job); pending.rename(pending.with_suffix('.json'))
        path = root / 'results' / f'{name}.json'
        while True:
            if path.exists():
                try: result = read(path); break
                except json.JSONDecodeError: pass
            if child.poll() is not None: raise RuntimeError(f'runner exited {child.returncode}')
            if time.monotonic() > deadline: raise TimeoutError('attribute-swap deadline')
            time.sleep(1)
        assert [s['ordinal'] for s in result['steps']] == list(range(count, count + result['new_forwards']))
        count += result['new_forwards']
        steps = []
        for step in result['steps']:
            arrays = dict(np.load(root / 'results' / step['artifact']))
            assert all(np.isfinite(x).all() for x in arrays.values()) and int(arrays['logits'].argmax()) == step['top1']
            steps.append((step, arrays))
        print(name + ': ' + result['text'], flush=True)
        return name, result, steps

    def spec(vector, gain):
        return dict(layer=21, step=0, gain=gain, row_deltas=[dict(position=position, delta=vector.tolist())])

    def margin(arrays, q): return float(arrays['logits'][q['helium_id']] - arrays['logits'][q['hydrogen_id']])

    def rank(arrays, token): return int((arrays['logits'] > arrays['logits'][token]).sum())

    def apply_check(base, arrays, vector, gain):
        expected = base['prefix_layer21'].copy(); expected[position] += np.float32(gain) * vector
        return bool(np.array_equal(expected, arrays['prefix_layer21']))

    state = {}

    def native(q, reserve):
        record = dict(name=q['name'], native_jobs=[]); summary['questions'].append(record)
        generated = []; base = None; boundary = None
        while len(generated) < q['native_cap'] and BUDGET - count > reserve:
            name, result, steps = run(f"{q['name']}-native", q['prompt_ids'], generated, 1)
            step, arrays = steps[0]; record['native_jobs'].append(name)
            if tok.decode([step['top1']]) in DIGITS:
                boundary, base = len(generated), arrays; generated.append(step['top1']); break
            generated.append(step['top1'])
            if step['top1'] in EOS: break
            progress()
        record.update(native_text=tok.decode(generated, skip_special_tokens=False), boundary=boundary)
        if boundary is None:
            record['status'] = 'budget reserve reached during native generation' if BUDGET - count <= reserve else 'no native digit'
            return None
        if generated[boundary] != q['hydrogen_id']:
            record['status'] = 'native first digit is not the hydrogen digit; patched conditions skipped'
            return None
        record.update(boundary_seed=generated[:boundary], native_top1=generated[boundary], native_margin=margin(base, q),
                      native_helium_rank=rank(base, q['helium_id']),
                      row0_5_l21_maxabs_vs_symbol_prompt=float(np.abs(base['prefix_layer21'][:position + 1] - nm_base['prefix_layer21'][:position + 1]).max()))
        state[q['name']] = dict(seed=generated[:boundary], base=base, record=record)
        progress(); return record

    def high(q):
        s = state[q['name']]
        name, result, steps = run(f"{q['name']}-high", q['prompt_ids'], s['seed'], 1, spec(delta[position], high_gain))
        step, arrays = steps[0]
        cond = dict(job=name, top1=step['top1'], token=step['text'], margin=margin(arrays, q),
                    delta_margin=margin(arrays, q) - s['record']['native_margin'], helium_digit_top1=step['top1'] == q['helium_id'],
                    helium_rank=rank(arrays, q['helium_id']), exact_application=apply_check(s['base'], arrays, delta[position], high_gain),
                    applied_positions=step.get('applied_positions'), top5=step['top5'])
        s['record']['high'] = cond; s['high_top1'] = step['top1']; progress(); return cond

    def random_at(q, r):
        s = state[q['name']]
        name, result, steps = run(f"{q['name']}-random-{r['seed']}", q['prompt_ids'], s['seed'], 1, spec(r['vector'], 1.0))
        step, arrays = steps[0]
        rec = dict(question=q['name'], seed=r['seed'], job=name, top1=step['top1'], token=step['text'], margin=margin(arrays, q),
                   delta_margin=margin(arrays, q) - s['record']['native_margin'], exact_application=apply_check(s['base'], arrays, r['vector'], 1.0))
        summary['random_control'].append(rec); progress(); return rec

    def ceiling(q):
        s = state[q['name']]
        name, result, steps = run(f"{q['name']}-donor-ceiling", q['donor_prompt_ids'], s['seed'], 1)
        step, arrays = steps[0]
        s['record']['ceiling'] = dict(job=name, top1=step['top1'], token=step['text'], margin=margin(arrays, q),
                                      helium_digit_top1=step['top1'] == q['helium_id']); progress()

    try:
        q1, q2 = questions
        # Core: Q1 native + high + random, Q2 native + high. Reserve keeps the remaining core affordable.
        if native(q1, reserve=2 + q2['native_cap'] // 2):
            high(q1)
            random_at(q1, randoms[0])
        if BUDGET - count > 1 and native(q2, reserve=1):
            high(q2)
        # Ceiling controls, only for questions that did not swap.
        for q in questions:
            s = state.get(q['name'])
            if s and 'high_top1' in s and s['high_top1'] != q['helium_id'] and count < BUDGET: ceiling(q)
        # Options in fixed order while budget remains.
        if q1['name'] in state and count < BUDGET: random_at(q1, randoms[1])
        if count < BUDGET:
            name, result, steps = run(f"symbol-random-{randoms[0]['seed']}", symbol_prompt, nm['seed_ids'], 1, spec(randoms[0]['vector'], 1.0))
            step, arrays = steps[0]
            summary['options'].append(dict(kind='symbol_random', job=name, top1=step['top1'], token=step['text'],
                                           he_minus_h=float(arrays['logits'][1293] - arrays['logits'][640]),
                                           baseline_he_minus_h=float(nm_base['logits'][1293] - nm_base['logits'][640]),
                                           entity_high_he_minus_h=nm_entity_high['he_minus_h'],
                                           exact_application=apply_check(nm_base, arrays, randoms[0]['vector'], 1.0))); progress()
        s = state.get(q1['name'])
        if s and 'high_top1' in s and s['high_top1'] not in EOS and count < BUDGET:
            name, result, _ = run(f"{q1['name']}-high-released", q1['prompt_ids'], s['seed'] + [s['high_top1']], min(3, BUDGET - count))
            summary['options'].append(dict(kind='released_continuation', question=q1['name'], job=name, text=result['text'],
                                           eos=result['generated_ids'][-1] in EOS)); progress()
        complete = all('high' in r for r in summary['questions']) and len(summary['questions']) == 2 and summary['random_control']
        summary['status'] = 'attribute-swap test finished' if complete else 'incomplete: core conditions missing'
    except Exception as exc:
        summary.update(status='incomplete', error=repr(exc)); raise
    finally:
        (root / 'STOP').touch()
        try: child.wait(timeout=20)
        except subprocess.TimeoutExpired:
            child.terminate()
            try: child.wait(timeout=20)
            except subprocess.TimeoutExpired: child.kill(); child.wait()
        path = root / 'events.jsonl'; events = [json.loads(x) for x in path.read_text().splitlines()] if path.exists() else []
        summary.update(attempted_forwards=sum(x['type'] == 'attempt' for x in events),
                       completed_forwards=sum(x['type'] == 'complete' for x in events), child_returncode=child.returncode, end=time.time())
        write(root / 'summary.json', summary)
    if child.returncode: raise RuntimeError(f'runner failed {child.returncode}')


if __name__ == '__main__': main(*sys.argv[1:])
