"""Create a new immutable NBCP execution manifest before any model inference."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
from run_natural_binding_pilot import read, sha, write_json

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--binary',type=Path,required=True)
p.add_argument('--vindex',type=Path,required=True)
p.add_argument('--run-root',type=Path,required=True)
a=p.parse_args()
repo=Path(__file__).resolve().parents[1]
data=repo/'docs/research-data/natural-binding-causal-pilot-v1'
source_commit=subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD'],text=True).strip()
if subprocess.check_output(['git','-C',str(repo),'status','--porcelain'],text=True).strip():
    raise SystemExit('commit the complete instrument before freezing')
for rel,digest in read(data/'design-manifest.json')['files'].items():
    if sha(repo/rel)!=digest:
        raise SystemExit(f'approved design changed: {rel}')
a.run_root.mkdir(parents=True,exist_ok=False)
files={str(repo/rel):sha(repo/rel) for rel in read(data/'design-manifest.json')['files']}
for rel in ('scripts/run_natural_binding_pilot.py','scripts/test_run_natural_binding_pilot.py',
            'scripts/freeze_natural_binding_run.py','crates/larql-lql/examples/natural_binding_causal_pilot.rs',
            'Cargo.lock','docs/research-natural-binding-causal-pilot-v1-execution.md'):
    files[str(repo/rel)]=sha(repo/rel)
# Bind every file in the immutable local vindex, including the actual weight planes.
# Streaming hashes use 1 MiB buffers; this does not instantiate model tensors.
for f in sorted(a.vindex.iterdir()):
    if f.is_file():
        print(f'hashing {f.name}',flush=True)
        files[str(f.resolve())]=sha(f)
manifest=dict(schema='larql.nbcp.execution.v1',source_commit=source_commit,repo=str(repo),
              vindex=str(a.vindex.resolve()),binary=str(a.binary.resolve()),binary_sha256=sha(a.binary),
              python=str(Path(sys.executable).resolve()),python_sha256=sha(sys.executable),
              numpy_version=np.__version__,controller=str(repo/'scripts/run_natural_binding_pilot.py'),
              run_root=str(a.run_root.resolve()),corpus=str(data/'corpus.json'),tokens=str(data/'tokenizer-preflight.json'),
              ledger=str(data/'forward-ledger.json'),files=files,max_forwards=368,max_per_stage=184,
              runtime_max_sec_per_stage=5400,approval='User approved the model-free protocol on 2026-09-24')
write_json(a.run_root/'execution-manifest.json',manifest)
print(json.dumps(dict(manifest=str(a.run_root/'execution-manifest.json'),sha256=sha(a.run_root/'execution-manifest.json'),
                      source_commit=source_commit,files_hashed=len(files)),indent=2))
