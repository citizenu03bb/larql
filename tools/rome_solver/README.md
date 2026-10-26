# Phase B ROME method

This package is separate from `tools/edit_cert`: Codex owns the method;
Claude Code owns certification and independently checks the rank-one solve.
No harness source is changed. There is no pretrained-model loader here.

`RomeSolver(model, layers, tokenizer, label="E2B", kv_deadline=13)` accepts
the model, layers and fast tokenizer exposed by the existing NF4 instrument.
It deliberately calls the model with autograd enabled instead of calling
`Nf4Instrument.forward`, which uses inference mode. Using a real model still
requires the approved protocol, synthetic rehearsal, shared lock and bounded
service described in `docs/phase-b-b1-b2-protocol.md`.

The method extracts a mean `down_proj` input key at the last subject token,
optimizes one absolute raw projection-output vector with frozen parameters,
and solves:

```text
q = solve(C, k*)                 # q = k* for the identity ablation
U = (v* - W k*)[:, None]
V = (q / (k*.T @ q))[:, None]
W' = W + U @ V.T
```

The SPD solve uses float64 without explicitly inverting C. Adapter arrays
are float32. The mathematical constraint is on the mean key. It does not
assert exact replacement at every context key. Optimization replaces the
raw output at the subject row, before post-FFN normalization; the exported
weight adapter is active at every token, with no subject or answer trigger.
This is a fixed Gemma adaptation of
[ROME](https://rome.baulab.info/), not a byte-identical port of its authors'
implementation. The KL direction follows
[`compute_v.py`](https://github.com/kmeng01/rome/blob/main/rome/compute_v.py).

`solution.save(new_directory)` exports the existing harness's `low_rank`
format (`backend.json`, `U0.npy`, `V0.npy`) plus `k_star.npy`, `v_star.npy`,
`w_k_star.npy`, `C.npy`, context witnesses and hashed `solver.json`. Identity
C is saved explicitly. NF4's effective dequantized projection is read only
to calculate `Wk*`; it is never requantized or overwritten. Float16 runtime
rounding remains a real-instrument issue for B2 to measure.

`solution.with_covariance()` constructs the identity ablation without a
model call, reusing exactly the same `k*`, `v*` and `Wk*`. It records zero
new forwards/updates and hashes of its source witnesses. It cannot rescue
or retune a failed primary layer search.

`seen_phrasings` includes S1, prefixes, their composed questions, rendered
chat inputs, teacher-forced inputs/targets, and the essence prompt. The
public dev adapter selects only an explicitly seen efficacy S1 and a
declared subject/target; it never trains on the other question texts.

Model-free commands, from the repository root:

```sh
/mnt/models/finetune/venv/bin/python -m unittest discover \
  -s tools/rome_solver/tests -p 'test_*.py' -v
/mnt/models/finetune/venv/bin/python tools/rome_solver/synthetic_dry_run.py \
  --out /tmp/phase-b0-synthetic-NEW
```

`prepare_b0.py --repo <checkout> --out <new-directory>` reads only
safetensors headers, installed modeling source, local tokenizer files,
English ADRs and the two explicitly named cached training Arrow files.
It fixes the covariance token windows before B1. It loads no model and
downloads nothing. The blind test directory is never accessed.
