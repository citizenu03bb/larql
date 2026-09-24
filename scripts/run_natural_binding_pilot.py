"""Frozen NBCP-v1 controller. Inference only via the checksummed bounded Rust worker."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import natural_binding_pilot as ref


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def write_json(path, value):
    with Path(path).open('x') as f:
        json.dump(value, f, indent=2, allow_nan=False)
        f.write('\n')
        f.flush()
        os.fsync(f.fileno())


def read(path):
    return json.loads(Path(path).read_text())


def ledger_valid(jobs, events):
    starts = [e for e in events if e['event'] == 'attempt']
    completions = [e for e in events if e['event'] == 'complete']
    if len(starts) != len(completions) or len(starts) > 184:
        return False
    if len(events) != 2 * len(starts):
        return False
    for i, (attempt, done) in enumerate(zip(starts, completions)):
        if events[2*i] != attempt or events[2*i+1] != done:
            return False
        for event in (attempt, done):
            if event['ordinal'] != i or event['recipient'] != jobs[i]['recipient'] or event['arm'] != jobs[i]['arm']:
                return False
    return True


def analyze(stage, rows, tokens, jobs, records, events, direction_exists):
    """Pure analysis: missing data can never turn into a successful gate."""
    base = dict(stage=stage, confirmation_allowed=False, completed_forwards=len(records))
    if not ledger_valid(jobs, events) or len(records) != len(events)//2:
        return dict(base, status='infrastructure_stop', reason='attempt ledger incomplete or inconsistent')
    if len(records) < 40:
        return dict(base, status='mechanical_stop', reason='incomplete baseline/replay phase')
    by_job = {(r['recipient'], r['arm']): r for r in records}
    expected_prefix = {(j['recipient'],j['arm']) for j in jobs[:len(records)]}
    if len(by_job) != len(records) or set(by_job) != expected_prefix:
        return dict(base, status='mechanical_stop', reason='records differ from ledger')
    baselines, failures = {}, []
    for row in rows:
        b = by_job[row['id'], 'baseline_capture']
        n = by_job[row['id'], 'hook_disabled_replay']
        if b['digest'] != n['digest'] or not b['finite'] or not n['finite']:
            return dict(base, status='mechanical_stop', reason='nonfinite or replay digest mismatch')
        baselines[row['id']] = b
        if row['entity'] is not None and b['top1_id'] != tokens[row['id']]['answer_id']:
            failures.append(dict(id=row['id'], expected_id=tokens[row['id']]['answer_id'],
                                 observed_id=b['top1_id'], expected_logit=b['answer_logit'],
                                 alternate_logit=b['alternative_logit'], top1_logit=b['top1_logit']))
    base['eligible_factual_prompts'] = 16 - len(failures)
    base['factual_prompts'] = 16
    base['eligibility_failures'] = failures
    if failures:
        if len(records) != 40:
            return dict(base, status='mechanical_stop', reason='interventions occurred despite failed eligibility')
        return dict(base, status='eligibility_stop', reason='unedited model did not rank every complete answer top-1')
    if len(records) == 40:
        return dict(base, status='baseline_pass', reason='eligible; intervention phase pending')
    core = [r for r in rows if r['entity'] is not None]
    results = []
    for row in core:
        arms = {a: by_job.get((row['id'],a)) for a in ref.ARMS}
        if any(v is None for v in arms.values()):
            return dict(base, status='mechanical_stop', reason='incomplete intervention row')
        for a, record in arms.items():
            if not record['finite'] or record['applied'] != [[21,tokens[row['id']]['final_position']]] or record['rejected']:
                return dict(base, status='mechanical_stop', reason='invalid intervention accounting')
            if a == 'self' and record['max_logit_drift'] > 1e-4:
                return dict(base, status='mechanical_stop', reason='self patch changed full logits')
        if any(abs(arms['self'][m]) > 1e-4 for m in ('delta_gap','delta_target')):
            return dict(base, status='mechanical_stop', reason='self patch changed target metric')
        effects = {a: {m: arms[a][m] for m in ('delta_gap','delta_target')} for a in ref.ARMS}
        results.append(dict(id=row['id'], passes=ref.row_passes(effects), effects=effects))
        # Early stop is only legal at the end of the first failing complete row.
        if not results[-1]['passes']:
            expected_count = 40 + len(results) * len(ref.ARMS)
            if len(records) != expected_count:
                return dict(base, status='mechanical_stop', reason='continued after failed row or invalid count')
            return dict(base, status='causal_negative', reason='frozen row gate failed', rows=results)
    if len(records) != 184 or not direction_exists:
        return dict(base, status='mechanical_stop', reason='missing frozen direction or wrong full-stage count')
    return dict(base, status='pass', reason='every frozen gate passed', rows=results,
                confirmation_allowed=(stage == 'discovery'))


def verify_manifest(path):
    m = read(path)
    if m['schema'] != 'larql.nbcp.execution.v1' or m['max_forwards'] != 368:
        raise ValueError('wrong manifest schema or budget')
    if np.__version__ != m['numpy_version']:
        raise ValueError('NumPy version changed')
    for file, digest in m['files'].items():
        if sha(file) != digest:
            raise ValueError(f'frozen file changed: {file}')
    if sha(m['binary']) != m['binary_sha256'] or sha(sys.executable) != m['python_sha256']:
        raise ValueError('runtime binary changed')
    rows, jobs = read(m['corpus']), read(m['ledger'])
    if rows != ref.corpus() or jobs != ref.ledger(rows):
        raise ValueError('corpus/ledger does not match approved reference')
    return m


def confirmation_gate(m, manifest_hash):
    directory = Path(m['run_root'])/'discovery'
    verdict = read(directory/'verdict.json')
    if verdict.get('manifest_sha256') != manifest_hash or verdict.get('status') != 'pass' or not verdict.get('confirmation_allowed'):
        raise ValueError('discovery did not authorize confirmation')
    if verdict['completed_forwards'] != 184:
        raise ValueError('discovery forward count wrong')
    if sha(directory/'direction.json') != verdict['direction_sha256']:
        raise ValueError('frozen discovery direction changed')
    if sha(directory/'events.jsonl') != verdict['events_sha256'] or sha(directory/'records.json') != verdict['records_sha256']:
        raise ValueError('discovery evidence changed')
    rows = [r for r in read(m['corpus']) if r['split']=='discovery']
    tokens = {r['id']:r for r in read(m['tokens'])['rows']}
    jobs = [j for j in read(m['ledger']) if j['split']=='discovery']
    events = [json.loads(line) for line in (directory/'events.jsonl').read_text().splitlines()]
    records = read(directory/'records.json')
    for record in records:
        if sha(directory/record['artifact']) != record['artifact_sha256']:
            raise ValueError('discovery forward artifact changed')
    reevaluated = analyze('discovery',rows,tokens,jobs,records,events,True)
    if reevaluated['status'] != 'pass':
        raise ValueError('discovery reanalysis failed')
    return read(directory/'direction.json')


def worker_args(m, mode, stage):
    return [m['binary'],mode,stage,m['vindex'],m['corpus'],m['tokens'],m['ledger']]


def receive(process, noise):
    while True:
        line = process.stdout.readline()
        if not line:
            raise RuntimeError(f'worker exited before response: {process.poll()}')
        if line.startswith('NBCP\t'):
            return json.loads(line[5:])
        noise.write(line)
        noise.flush()


def run_stage(m, stage, manifest_hash, direction):
    root = Path(m['run_root'])
    root.mkdir(parents=True, exist_ok=True)
    out = root/stage
    out.mkdir()  # Exclusive reservation: no restart, even after a crash.
    rows = [r for r in read(m['corpus']) if r['split']==stage]
    row_by_id = {r['id']:r for r in rows}
    tokens = {r['id']:r for r in read(m['tokens'])['rows']}
    jobs = [j for j in read(m['ledger']) if j['split']==stage]
    records, events, captures, baseline_logits = [], [], {}, {}
    process, verdict = None, None
    write_json(out/'started.json',dict(stage=stage,manifest_sha256=manifest_hash,source_commit=m['source_commit']))
    try:
        with (out/'worker-noise.log').open('x') as noise, (out/'events.jsonl').open('x') as journal:
            process = subprocess.Popen(worker_args(m,'serve',stage),stdin=subprocess.PIPE,stdout=subprocess.PIPE,text=True,bufsize=1)
            if receive(process,noise)['type'] != 'preflight' or receive(process,noise)['type'] != 'ready':
                raise RuntimeError('unexpected worker initialization')
            u = None
            for ordinal,job in enumerate(jobs):
                if ordinal == 40:
                    verdict = analyze(stage,rows,tokens,jobs,records,events,False)
                    if verdict['status'] != 'baseline_pass':
                        break
                    try:
                        if stage == 'discovery':
                            u = ref.fit_rank_one(rows,captures)
                            direction = dict(u=u.tolist(), stage='discovery', manifest_sha256=manifest_hash)
                        else:
                            u = np.asarray(direction['u'],dtype=np.float64)
                        if u.shape != (2560,) or not np.isfinite(u).all() or abs(np.dot(u,u)-1)>1e-10:
                            raise ValueError('invalid frozen direction')
                    except ValueError as e:
                        verdict = dict(stage=stage,status='numerical_stop',reason=str(e),confirmation_allowed=False,
                                       completed_forwards=len(records))
                        break
                    write_json(out/'direction.json',direction)
                recipient,arm = job['recipient'],job['arm']
                replacement = None
                if arm in ref.ARMS:
                    r,d = captures[recipient],captures[job['donor']]
                    if arm.startswith('random_'):
                        sign = 1 if arm.endswith('_plus') else -1
                        replacement = ref.random_control(r,d,u,recipient,sign)
                    else:
                        replacement = ref.project(r,d,u)
                request = dict(ordinal=ordinal,recipient=recipient,arm=arm,
                               replacement=None if replacement is None else replacement.tolist())
                attempt = dict(event='attempt',ordinal=ordinal,recipient=recipient,arm=arm)
                events.append(attempt)
                journal.write(json.dumps(attempt)+'\n'); journal.flush(); os.fsync(journal.fileno())
                process.stdin.write(json.dumps(request,allow_nan=False)+'\n'); process.stdin.flush()
                result = receive(process,noise)
                if result['type']!='forward' or any(result[k]!=request[k] for k in ('ordinal','recipient','arm')):
                    raise RuntimeError('response mismatch')
                logits = np.asarray(result['logits'],dtype='<f4')
                if logits.shape!=(262144,) or not np.isfinite(logits).all():
                    raise RuntimeError('invalid logits')
                artifact = f'{ordinal:03d}-{arm}.npz'
                arrays = dict(logits=logits)
                if result['capture'] is not None:
                    h = np.asarray(result['capture'],dtype='<f4')
                    if h.shape!=(2560,) or not np.isfinite(h).all():
                        raise RuntimeError('invalid capture')
                    captures[recipient] = h.astype(np.float64)
                    baseline_logits[recipient] = logits
                    arrays['capture'] = h
                if replacement is not None:
                    exact = np.asarray(result['replacement_f32'],dtype='<f4')
                    np.testing.assert_array_equal(exact,np.asarray(replacement,dtype='<f4'))
                    arrays['replacement_f32'] = exact
                with (out/artifact).open('xb') as f:
                    np.savez_compressed(f,**arrays); f.flush(); os.fsync(f.fileno())
                digest = hashlib.sha256(logits.tobytes()).hexdigest()
                rec = dict(ordinal=ordinal,recipient=recipient,arm=arm,digest=digest,finite=True,
                           top1_id=int(np.argmax(logits)),top1_logit=float(np.max(logits)),
                           artifact=artifact,artifact_sha256=sha(out/artifact),applied=result['applied'],rejected=result['rejected'])
                if row_by_id[recipient]['entity'] is not None:
                    a,b = tokens[recipient]['answer_id'],tokens[recipient]['alternative_id']
                    rec.update(answer_logit=float(logits[a]),alternative_logit=float(logits[b]))
                    if arm in ref.ARMS:
                        base_logits = baseline_logits[recipient]
                        # Cast logits to f64 before differences, matching analyzer arithmetic.
                        rec.update(delta_target=float(logits[b])-float(base_logits[b]),
                                   delta_gap=(float(logits[b])-float(logits[a]))-(float(base_logits[b])-float(base_logits[a])),
                                   max_logit_drift=float(np.max(np.abs(logits.astype(np.float64)-base_logits.astype(np.float64)))))
                records.append(rec)
                complete = dict(attempt,event='complete',artifact=artifact,artifact_sha256=rec['artifact_sha256'])
                events.append(complete)
                journal.write(json.dumps(complete)+'\n'); journal.flush(); os.fsync(journal.fileno())
                print(f'{stage}: {ordinal+1}/184 {recipient} {arm} top1={rec["top1_id"]}',flush=True)
                if arm=='hook_disabled_replay' and digest!=hashlib.sha256(baseline_logits[recipient].tobytes()).hexdigest():
                    verdict = dict(stage=stage,status='mechanical_stop',reason='replay mismatch',confirmation_allowed=False,completed_forwards=len(records))
                    break
                if ordinal>=40 and (ordinal-39)%9==0:
                    # Analyze the completed recipient without requiring future rows.
                    current = records[-9:]
                    if current[-5]['arm']!='self':
                        raise RuntimeError('unexpected arm ordering')
                    effects = {r['arm']:{k:r[k] for k in ('delta_gap','delta_target')} for r in current}
                    self_rec = next(r for r in current if r['arm']=='self')
                    if self_rec['max_logit_drift']>1e-4 or any(abs(self_rec[k])>1e-4 for k in ('delta_gap','delta_target')):
                        verdict=dict(stage=stage,status='mechanical_stop',reason='self patch drift',confirmation_allowed=False,completed_forwards=len(records))
                        break
                    if not ref.row_passes(effects):
                        verdict=analyze(stage,rows,tokens,jobs,records,events,True)
                        break
            else:
                verdict=analyze(stage,rows,tokens,jobs,records,events,True)
            process.stdin.close()
            code=process.wait(timeout=60)
            if code!=0:
                raise RuntimeError(f'worker exit={code}')
    except Exception as e:
        verdict=dict(stage=stage,status='infrastructure_stop',reason=f'{type(e).__name__}: {e}',
                     confirmation_allowed=False,completed_forwards=len(records))
        if process is not None and process.poll() is None:
            process.kill(); process.wait(timeout=30)
    finally:
        write_json(out/'records.json',records)
        if verdict is None:
            verdict=dict(stage=stage,status='infrastructure_stop',reason='no verdict',confirmation_allowed=False,completed_forwards=len(records))
        verdict.update(manifest_sha256=manifest_hash,attempted_forwards=sum(e['event']=='attempt' for e in events),
                       records_sha256=sha(out/'records.json'),events_sha256=sha(out/'events.jsonl'))
        if (out/'direction.json').exists():
            verdict['direction_sha256']=sha(out/'direction.json')
        write_json(out/'verdict.json',verdict)
        print(json.dumps({k:v for k,v in verdict.items() if k not in ('rows','eligibility_failures')},indent=2),flush=True)
    return 0 if verdict['status'] in ('pass','eligibility_stop','causal_negative','numerical_stop') else 2


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest',type=Path,required=True)
    p.add_argument('--stage',choices=['discovery','confirmation'],required=True)
    p.add_argument('--preflight-only',action='store_true')
    a=p.parse_args()
    m=verify_manifest(a.manifest)
    manifest_hash=sha(a.manifest)
    direction=confirmation_gate(m,manifest_hash) if a.stage=='confirmation' else None
    # No model load until all file hashes and (for confirmation) discovery gates pass.
    if a.preflight_only:
        subprocess.run(worker_args(m,'preflight',a.stage),check=True)
        return 0
    return run_stage(m,a.stage,manifest_hash,direction)

if __name__=='__main__':
    sys.exit(main())
