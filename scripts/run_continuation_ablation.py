"""Cross emitted symbol with retained residual intervention; reuse fixed-probe traces."""
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main(previous, destination, binary, vindex):
    previous, root, binary = Path(previous), Path(destination), Path(binary)
    repo = Path(__file__).resolve().parents[1]
    spec = read(previous / 'fixed-intervention.json')
    binding = read(previous / 'parameter-binding.json')
    assert hashlib.sha256(json.dumps(spec, sort_keys=True, separators=(',', ':')).encode()).hexdigest() == binding['spec_sha256']
    assert sha(binary) == read(previous / 'session.json')['binary_sha256']
    auto = read(previous / 'results/006-auto-long-sentence.json')
    base = read(previous / 'results/007-base-long-sentence.json')
    prompt = read(previous / 'jobs/006-auto-long-sentence.json')['prompt_ids']
    fire = next(s for s in auto['steps'] if s['triggered'])
    step = fire['step']
    assert step == 6 and fire['applied_position'] == len(prompt) + step - 1
    assert base['generated_ids'][:step] == auto['generated_ids'][:step]
    assert base['generated_ids'][step] != auto['generated_ids'][step]
    intervention = {k: spec[k] for k in ('layer', 'gain', 'delta')}
    intervention['step'] = step
    jobs = {
        '000-he-no-residual': dict(prompt_ids=prompt, seed_ids=auto['generated_ids'][:step+1], max_new_tokens=8),
        '001-h-with-residual': dict(prompt_ids=prompt, seed_ids=base['generated_ids'][:step+1], max_new_tokens=8, intervention=intervention),
        '002-replay-h-no-residual': dict(prompt_ids=prompt, seed_ids=base['generated_ids'][:12], max_new_tokens=1),
        '003-replay-he-with-residual': dict(prompt_ids=prompt, seed_ids=auto['generated_ids'][:12], max_new_tokens=1, intervention=intervention),
    }
    assert sum(j['max_new_tokens'] for j in jobs.values()) == 18
    (root / 'jobs').mkdir(parents=True, exist_ok=False)
    (root / 'references').mkdir()
    for name, job in jobs.items():
        (root / 'jobs' / (name + '.json')).write_text(json.dumps(job, indent=2) + '\n')
    refs = {}
    for name in ('006-auto-long-sentence', '007-base-long-sentence'):
        source = previous / 'results' / (name + '.json')
        shutil.copy2(source, root / 'references' / source.name)
        s = next(s for s in read(source)['steps'] if s['step'] == 12)
        artifact = previous / 'results' / s['artifact']
        shutil.copy2(artifact, root / 'references' / artifact.name)
        refs[name] = dict(result_sha256=sha(source), artifact=artifact.name, artifact_sha256=sha(artifact))
    session = dict(mode='exploratory continuation ablation', max_forwards=24, planned_forwards=18,
                   max_runtime_seconds=1200, source_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=repo, text=True).strip(),
                   previous=str(previous), binary_sha256=sha(binary), parameter_sha256=binding['spec_sha256'],
                   reused_references=refs, start=time.time(), note='Diagnostic seeded branches; no detector retraining or new from-prompt success claim.')
    (root / 'session.json').write_text(json.dumps(session, indent=2) + '\n')
    child = subprocess.Popen([sys.executable, '-u', str(repo / 'scripts/explore_natural_fact_edit.py'), str(root), str(binary), vindex, '24'])
    try:
        deadline = time.monotonic() + 1150
        while not all((root / 'results' / (name + '.json')).exists() for name in jobs):
            if child.poll() is not None:
                raise RuntimeError(f'worker runner exited early: {child.returncode}')
            if time.monotonic() >= deadline:
                raise TimeoutError('continuation ablation deadline')
            time.sleep(1)
        summary = {name: read(root / 'results' / (name + '.json'))['text'] for name in jobs}
        (root / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
        print(json.dumps(summary), flush=True)
    finally:
        (root / 'STOP').touch()
        try:
            child.wait(timeout=35)
        except subprocess.TimeoutExpired:
            child.terminate()
            child.wait(timeout=20)
    if child.returncode:
        raise RuntimeError(f'worker runner failed: {child.returncode}')


if __name__ == '__main__':
    main(*sys.argv[1:])
