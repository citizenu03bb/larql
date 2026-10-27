#!/usr/bin/env python3
"""Independent offline P1-5 verdict/probability replay for Pat's brief 46.

No harness judge imports or pretrained model. Reuses Codex's B2 control
implementation with Codex's independently written P1 negation reading.
Probe rules: p1_probe.py at 3205f778, read as text, not imported/executed.
Expected headlines were supplied in the brief; this is not a blind audit.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import subprocess
import zipfile

import xcheck_b2_v2 as controls
from xcheck_p1_qualification import stated
from xcheck_p1_pairs import FACTS, ROOT, MODEL_SHA, edit, read, require, sha

REPO = Path(__file__).resolve().parents[2]
ARCHIVE = Path('/home/pat/Documents/Codex/2026-09-24/our/outputs/phase-b-p1-5-certification-20261027.zip')
ARCHIVE_SHA = 'd82c89c716e7ba395cfe585d381b65642c540e26b40bb26e9350947e11a2ebc0'
PLAN_REV = 'dee168bb'
PROBE_PLAN_REV = '41779bca'
RULE_REV = '3205f778'


def gitbytes(rev, name):
    return subprocess.check_output(['git', 'show', rev + ':' + name], cwd=REPO)


def gitjson(rev, name):
    return json.loads(gitbytes(rev, name))


def manifests(root, archive):
    require(sha(archive) == ARCHIVE_SHA, 'Archive SHA mismatch: stop')
    counts = {}
    with zipfile.ZipFile(archive) as z:
        require(z.testzip() is None, 'Archive CRC failure')
        for part in ('certification', 'probe'):
            directory = root / part; manifest = directory / 'MANIFEST.sha256'
            require(z.read(root.name + '/' + part + '/MANIFEST.sha256') == manifest.read_bytes(), 'Archived/live manifest differs')
            count = 0
            for line in manifest.read_text().splitlines():
                digest, name = line.split(None, 1)
                rel = Path(name.removeprefix('./'))
                require(not rel.is_absolute() and '..' not in rel.parts, 'Unsafe manifest path')
                path = directory / rel
                require(sha(path) == digest, 'Manifest mismatch: ' + str(path))
                require(z.read(root.name + '/' + part + '/' + rel.as_posix()) == path.read_bytes(), 'Archive/live payload differs')
                count += 1
            counts[part] = count
    return dict(archive_sha256=ARCHIVE_SHA, files=counts, archive_matches_live_payloads=True)


def rates(margins):
    result = {}
    for kind in ('ES', 'PS', 'NS'):
        qids = [q for q in margins if (q == 'S1' if kind == 'ES' else q.startswith('E') if kind == 'PS' else q.startswith('O'))]
        count = sum(margins[q] < 0 if kind == 'NS' else margins[q] > 0 for q in qids)
        result[kind] = dict(successes=count, count=len(qids), rate=count / len(qids) if qids else None, questions=qids)
    return result


def paper_threshold(r):
    if any(r[k]['rate'] is None for k in ('ES', 'PS', 'NS')):
        return None
    return r['ES']['rate'] == 1 and r['PS']['rate'] >= .75 and r['NS']['rate'] >= .75


def replay(spec, evidence, backend, plan):
    original = controls.has
    controls.has = lambda generation, values: stated(generation['text'], values)
    try:
        return controls.replay(spec, evidence, backend, plan)
    finally:
        controls.has = original


def run(root, archive):
    integrity = manifests(root, archive)  # Stop before interpreting any mismatched archive.
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained('/mnt/models/hf-models/gpt2-xl', local_files_only=True)
    stops = {tok.eos_token_id} | {i for i in range(len(tok)) if '\n' in tok.decode([i]) or tok.decode([i]).rstrip(' ').endswith(('.', ',', ';', '!', '?'))}
    index = read(root / 'certification/index.json')
    require(set(index['facts']) == set(FACTS), 'Unexpected certification inventory')
    frozen_plans = gitjson(PLAN_REV, 'docs/research-data/phase-b-p1-5-plans-claude.json')
    frozen_probe = gitjson(PROBE_PLAN_REV, 'docs/research-data/phase-b-p1-5-probe-plan-claude.json')
    handoff = read(REPO / 'docs/research-data/phase-b-p1-a3-certification-handoff-codex.json')
    result = dict(author='Codex', model_forwards=0, harness_judge_imports=False, blind_audit=False,
        integrity=integrity, plan_revision=PLAN_REV, probe_plan_revision=PROBE_PLAN_REV,
        probe_rule_revision=RULE_REV, probe_rule_sha256=hashlib.sha256(gitbytes(RULE_REV, 'tools/edit_cert/scripts/p1_probe.py')).hexdigest(),
        cases={}, differences=[], certification_forwards=0, certification_services=0, probe_forwards=0,
        probe_margin_comparisons=[], probe_valid=True, per_fact_paper_rule_is_posthoc=True)
    pooled = {cond: {} for cond in ('edit', 'random')}
    for fact in FACTS:
        directory = root / 'certification' / fact / 'run'
        spec, plan = read(directory / 'spec.json'), read(directory / 'plan.json')
        sealed = frozen_plans['facts'][fact]; entry = index['facts'][fact]
        require(sha(directory / 'spec.json') == plan['spec_sha256'] == sealed['spec_sha256'] == handoff['cases'][fact]['spec_sha256'], 'Spec changed')
        require(hashlib.sha256(json.dumps({k:v for k,v in plan.items() if k != 'plan_sha256'}, sort_keys=True).encode()).hexdigest() == plan['plan_sha256'] == sealed['plan_sha256'], 'Plan changed')
        require(plan['total_forwards'] == entry['forwards'] == sealed['forwards'] == handoff['cases'][fact]['bench_forward_cap'], 'Plan ceiling mismatch')
        require(sha(directory / 'evidence.json') == entry['evidence_sha256'] and sha(directory / 'verdict.json') == entry['verdict_sha256'], 'Evidence/verdict hash mismatch')
        expected_content = handoff['cases'][fact]['edit_content_sha256']
        for adapter in (directory.parent / 'edit-min', directory / 'edit-source', directory / 'edit-saved'):
            backend, _ = edit(adapter)
            require(backend['content_sha256'] == expected_content, 'Certification adapter changed')
        generations, boundaries, actual = [], [], []
        for service in plan['services']:
            folder = directory / 'services' / str(service['index'])
            status, record = read(folder / 'status.json'), read(folder / 'records.json')
            require(status['status'] == 'complete' and sha(folder / 'records.json') == status['records_sha256'], 'Incomplete/changed service')
            require(record['plan_sha256'] == plan['plan_sha256'] and record['service'] == service['index'] and record['phase'] == service['phase'], 'Service identity mismatch')
            require(record['instrument'] == 'hf-gpt2xl' and not record['synthetic'] and record['kv_write_deadline'] == 47, 'Instrument mismatch')
            require(status['forwards'] == record['forwards'] == sum(len(g['ids']) for g in record['generations']) + len(record['boundaries']), 'Actual forward accounting mismatch')
            require(0 <= record['forwards'] <= service['forwards'] <= plan['cap'] == 400, 'Service above ceiling')
            require(set(record['content_sha256'].values()) == {expected_content}, 'Service adapter identity mismatch')
            require(MODEL_SHA + '  model.safetensors' in record['fingerprint']['weights_sha256'], 'Model identity mismatch')
            for g in record['generations']:
                require(any(i['kind'] == 'generate' and i['question_id'] == g['question_id'] and i['condition'] == g['condition'] for i in service['items']), 'Unplanned generation')
                require(g['ids'] and tok.decode([i for i in g['ids'] if i != tok.eos_token_id]) == g['text'], 'Generation text/ID mismatch')
                require(not any(i in stops for i in g['ids'][:-1]), 'Generation passed a stop token')
                require(g['eos'] == (g['ids'][-1] in stops) and g['capped'] == (not g['eos'] and len(g['ids']) == spec['max_new_tokens']), 'Stop/cap mismatch')
            generations.extend(record['generations']); boundaries.extend(record['boundaries']); actual.append(record['forwards'])
        evidence = read(directory / 'evidence.json')
        require(generations == evidence['generations'] and boundaries == evidence['boundaries'], 'Evidence differs from records')
        own = replay(spec, evidence, backend, plan)
        peer = read(directory / 'verdict.json'); reported = {c['id']: c['status'] for c in peer['controls']}
        require(reported == entry['statuses'] and peer['verdict'] == entry['verdict'], 'Index/verdict inconsistency')
        for control, status in own['controls'].items():
            if status != reported[control]:
                result['differences'].append(dict(fact=fact, control=control, codex=status, claude=reported[control]))
        if own['verdict'] != peer['verdict']:
            result['differences'].append(dict(fact=fact, control='verdict', codex=own['verdict'], claude=peer['verdict']))
        for q in own['questions']:
            peer_q = peer['qualification'][q['id']]['level'] != 'excluded'
            if q['qualified'] != peer_q:
                result['differences'].append(dict(fact=fact, control='qualification', question=q['id'], codex=q['qualified'], claude=peer_q))
        result['certification_forwards'] += sum(actual); result['certification_services'] += len(actual)
        # Probe item selection is kept exactly as committed, without reselecting after outcomes.
        probe_plan = frozen_probe['facts'][fact]
        require(probe_plan['spec_sha256'] == sha(directory / 'spec.json'), 'Probe spec mismatch')
        items = {i['qid']: i for i in probe_plan['items'] if 'excluded' not in i}
        margins_file = root / 'probe' / fact / 'margins.json'; p = read(margins_file)
        require(p['fact'] == fact and set(p['margins']) == {'native', 'edit', 'random'}, 'Probe identity mismatch')
        require(p['forwards'] == probe_plan['forwards'] == 6 * len(items) <= 54, 'Probe count mismatch')
        calculated = {}
        for condition, values in p['margins'].items():
            require(set(values) == set(items), 'Probe inclusion differs from frozen plan')
            calculated[condition] = {}
            for qid, values_ in values.items():
                it = items[qid]; prompt_ids = tok.encode(it['prompt'], add_special_tokens=False)
                a = prompt_ids + tok.encode(it['prefix'] + it['true'], add_special_tokens=False)
                b = prompt_ids + tok.encode(it['prefix'] + it['new'], add_special_tokens=False)
                divergence = next(i for i in range(min(len(a), len(b))) if a[i] != b[i])
                require(divergence >= len(prompt_ids) and len(values_['true']) == len(a) - divergence and len(values_['new']) == len(b) - divergence, 'Probe suffix length mismatch')
                require(all(math.isfinite(v) and v <= 0 for key in ('new', 'true') for v in values_[key]), 'Invalid token log probability')
                margin = sum(values_['new']) - sum(values_['true'])
                calculated[condition][qid] = margin
                result['probe_margin_comparisons'].append(dict(fact=fact, condition=condition, question=qid,
                    recalculated=margin, archived=values_['margin'], equal_exactly=margin == values_['margin'], absolute_difference=abs(margin - values_['margin'])))
                if condition in pooled:
                    pooled[condition][fact + ':' + qid] = (qid, margin)
        valid = all(v < 0 for q, v in calculated['native'].items() if q == 'S1' or q.startswith('E'))
        result['probe_valid'] &= valid; result['probe_forwards'] += p['forwards']
        per_fact = {condition: rates(values) for condition, values in calculated.items() if condition != 'native'}
        result['cases'][fact] = dict(controls=own['controls'], verdict=own['verdict'], reported_controls=reported,
            actual_forwards=actual, planned_ceiling=plan['total_forwards'], random_diagnostics=own['random_diagnostics'],
            questions=[dict(id=q['id'], qualified=q['qualified'], status=q['status'], control=q['control'],
                            native=q['native']['text'] if q['native'] else None, edit=q['edit']['text'] if q['edit'] else None,
                            edit_capped=q['edit']['capped'] if q['edit'] else None) for q in own['questions']],
            probe=dict(valid=valid, forwards=p['forwards'], margins=calculated, rates=per_fact,
                       same_relation_neighbours=[q for q in items if q in ('O1','O2','O3')],
                       posthoc_per_fact_threshold=paper_threshold(per_fact['edit']),
                       excluded_items=[i for i in probe_plan['items'] if 'excluded' in i]))
    result['pooled_rates'] = {}
    for condition, values in pooled.items():
        grouped = {k: [] for k in ('ES','PS','NS')}
        for _, (qid, margin) in values.items():
            grouped['ES' if qid == 'S1' else 'PS' if qid.startswith('E') else 'NS'].append(margin < 0 if qid.startswith('O') else margin > 0)
        result['pooled_rates'][condition] = {k: dict(successes=sum(v), count=len(v), rate=sum(v)/len(v) if v else None) for k,v in grouped.items()}
    result['like_paper_pooled'] = paper_threshold(result['pooled_rates']['edit']) if result['probe_valid'] else None
    result['certified'] = sum(c['verdict'] == 'CERTIFIED' for c in result['cases'].values())
    result['all_probe_margins_equal_exactly'] = all(c['equal_exactly'] for c in result['probe_margin_comparisons'])
    result['total_forwards_since_v2'] = 9030 + result['certification_forwards'] + result['probe_forwards']
    result['reading'] = 'No frozen global row covers this mixed outcome. Keep B2 unvalidated/suspended for causal attribution; retain all measured B2/P1 negatives. Per-fact probability summaries are posthoc descriptions, not a replacement global gate.'
    result['limitations'] = ['No model probabilities independently reproduced: probe arithmetic uses archived per-token log probabilities; C4 uses recorded boundary margins without full logits.',
        'Probe eligibility is the committed item plan, not a newly optimized selection. Shakespeare has no S1 and no included same-relation neighbour; its NS is unrelated O4 only.',
        'An unknown or capped attribute control does not establish collateral preservation or collateral failure. Do not equate it with the explicit BMW/Gretzky attribute changes.']
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('--out', type=Path, required=True); args=p.parse_args()
    result = run(ROOT, ARCHIVE)
    with args.out.open('x') as f: f.write(json.dumps(result, indent=2, sort_keys=True, allow_nan=False)+'\n')
    print(json.dumps({k:result[k] for k in ('certified','certification_forwards','certification_services','probe_forwards','probe_valid','pooled_rates','like_paper_pooled','differences','all_probe_margins_equal_exactly','total_forwards_since_v2')}))
    print(json.dumps({f:c['probe']['rates']['edit'] for f,c in result['cases'].items()}))


if __name__ == '__main__': main()
