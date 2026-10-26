# edit_cert: a certification harness for persistent model edits

`edit_cert` takes **one edit** of a language model and **one target fact**, and returns a verdict:

```
CERTIFIED | REFUSED (reasons)
```

It writes the verdict as `verdict.json` and `verdict.md`. The harness does not depend on the edit method. An edit is a *backend*: an activation hook, a low-rank weight delta (ROME-style rank 1, or LoRA-style rank r), or a planted test edit. The harness never edits anything itself.

It exists because a fact edit is easy to *seem* to make and hard to *own*. In the activation-patching series that preceded it, a delta keyed on the entity:
- flipped hydrogen's symbol to He on the prompts it was tuned on;
- left the atomic number alone at the boundary where it was measured;
- turned the atomic number into "three" once it was applied persistently to every mention.

The controls below are the ones that series needed, written down before any edit method is tried.

Design and decisions: [`docs/edit-certification-design.md`](../../docs/edit-certification-design.md).

## Scope of a certificate

A certificate covers *this edit, this fact, this question set, this instrument, greedy decoding, one trial per question*. It is not a claim that the method generalises across facts.

## Evidence rule

Two kinds of evidence:

- **Generation**: free greedy generation from an empty chat answer, up to EOS or a cap. Only a generation can make a control **PASS**. A cap hit is unreadable, not a pass.
- **Resumed-boundary diagnostic**: one teacher-forced forward on the *native* answer prefix, recording the top-1 and a margin. It can make a control **FAIL**, never **PASS**. When a generation exists for the same question, the generation decides.

Two more rules follow the same asymmetry:
- A phrasing used to build or calibrate the edit (`seen`) can refuse the edit but never certify it.
- A question whose *native* answer is wrong or truncated is excluded and listed. A control left without enough qualified questions is `NOT_EVALUATED`; it is never passed by default.

## Controls

| # | Control | PASS rule | Gating |
|---|---|---|---|
| C0 | Persistent form | No answer-time trigger, no donor pass, no use of the native proposal; one stored artifact applied at every forward; writes no later than the instrument's K/V write deadline (E4B block 22, E2B block 13) | yes |
| C1 | Efficacy on unseen phrasings | ≥ 2 unseen phrasings, both subject orders: new value stated, old value absent, no swap marker, EOS | yes |
| C2 | Other attributes of the entity | At least one qualified attribute in each subject order; every attribute question keeps its native value and states none of the rejected values (design §12) | yes |
| C3 | Mentions before and after the relation | C1 and C2 hold in each subject order separately; an answer-only mention (e.g. "What is water made of?") stays clean; row-split diagnostics count by the mention they act on | yes |
| C4 | Random direction of equal norm | The random counterpart never states the new value, and the edit's boundary gain exceeds 2 × max(random gain, 0) | yes |
| C5 | Donor ceiling | Reported. A phrasing whose donor ceiling cannot flip is removed from C1 | no |
| C6 | Off-target questions | Greedy answers token-identical to native. For trigger-keyed hooks, the first forward is bit-identical when no trigger token is present | yes |
| C7 | Persistence after reload | The harness's own saved copy, loaded in a later service, gives token-identical C1 answers, with the same content hash | yes |
| C8 | Propagation (RippleEdits-style) | Reported only in v1 | no |

The verdict is `CERTIFIED` if and only if every gating control is `PASS`. Each reason says whether it is a `FAIL` or `NOT_EVALUATED`, and on which evidence.

## Layout

```
edit_cert/
  spec.py          fact spec (JSON) and validation
  evidence.py      Generation / Boundary records, readings
  controls.py      C0..C8, pure functions over evidence
  verdict.py       the rule, JSON + markdown
  plan.py          items, worst-case forward budget, packing into services of <= 24 forwards
  runner.py        prepare / run one service / collect / certify
  instruments/     fake.py (tests), nf4.py (transformers + bitsandbytes NF4 bench)
  backends/        activation.py (static, one-shot), lowrank.py, fake.py
  replay/          archived runs R1..R5, the exit criterion
facts/             fact specs (hydrogen-symbol-He.json: a draft live question set, not run)
tests/             model-free tests
run_service.sh     one real-model service under the shared lock and memory limits
```

## Use

Everything except `run` on a real instrument is model-free.

```bash
cd tools/edit_cert
python3 -m edit_cert selftest                 # planted selective edit on the fake instrument -> CERTIFIED
python3 -m edit_cert replay --out /tmp/replay # archived cases R1..R5 against their expected verdicts
python3 -m edit_cert plan --run RUN --fact facts/hydrogen-symbol-He.json --edit EDIT_DIR --instrument nf4-e4b
python3 -m edit_cert verdict --run RUN
```

Real-model services go through `run_service.sh`. It refuses to start unless:
- the protocol file is committed and unchanged;
- `--i-have-approval` is passed;
- the run has a plan.

It then takes the shared heavy-model `flock` and runs the service in a `systemd-run` scope (MemoryHigh 22G, MemoryMax 24G, MemorySwapMax 8G, RuntimeMaxSec 1800). Each service is capped at 24 forwards by default, and services must run in order. `plan --cap 400` is accepted only for the NF4 instruments, under Phase B decision Q1; the cap and its authority are recorded in the hashed plan, and any other value is refused.

For the draft hydrogen question set, the plan is 21 services and at most 426 forwards. That is a worst case: every generation is budgeted at the 12-token cap. At about 0.3 s per E4B forward on the bench, plus about 7 s per load, this comes to roughly 4 to 5 minutes.

## Edits on disk

An edit is a directory holding `backend.json` (kind, label, form, provenance, params, content hash) and `.npy` arrays.

- `seen_phrasings` lists the texts used to build or calibrate the edit. `plan` refuses a fact spec that marks one of them as unseen.
- `load` refuses an edit whose arrays no longer match the stored content hash.

| Kind | What it does | Random counterpart |
|---|---|---|
| `activation_static` | adds `gain * vector` to block `layer`'s output on every row whose token is in `trigger_ids`, at every forward | Gaussian vector with the same float16 norm, same trigger rule |
| `activation_oneshot` | the archived one-shot injection; replay-only, refused by C0 | same, one-shot |
| `low_rank` | `y += scale * U (V^T x)` on named linear modules, in float32, without requantising the NF4 weight | random U, V with the same column norms |

## Exit criterion: replay of archived cases

`replay` reads the archived runs of the activation-patching series. For each case it:
- checks every artifact hash;
- recomputes every margin and top-1 from the saved logits;
- requires exact equality with the archived numbers;
- runs the same controls.

Archived runs hold resumed-boundary diagnostics only, so replay can refuse but never certify.

| Case | Edit | Verdict | Decisive reasons |
|---|---|---|---|
| R1 | one-shot L21 entity delta at row 5 (NF4) | REFUSED | C0 FAIL (one-shot, needs the native proposal); C1 FAIL (the symbol rises by +19.7 but H stays top-1 on the phrasing it was built from) |
| R2 | static L21 delta on every hydrogen token, g* = 1.0 (NF4) | REFUSED | C2 FAIL (number top-1 "three"); C3 FAIL (the answer mention alone, after the relation, gives "three"; the prompt mention alone does not) |
| R3 | random direction of the R1 norm (NF4) | REFUSED | C0 FAIL; C1 FAIL (no flip) |
| R4 | full helium subject substitution (NF4) | REFUSED | C0 FAIL (substitutes the subject); C2 and C3 FAIL (number becomes "two") |
| R5 | one-shot L21 delta × 2.585 (LARQL Q4K) | REFUSED | C0 FAIL; number +0.76 and mass +5.57 reproduced ("4" rises from 48th to 4th) |

Two coverage amendments followed an independent comparison by Codex: off-target frontiers in R3, and the symbol donor in R1, reported under C5. Neither changes a verdict or a gating status; see design §11. The private archives are not in this repository. The adapters are also tested on synthetic archives with the same layout, in `tests/replay_fixtures.py`. **No real edit has been certified yet.** The only `CERTIFIED` verdict comes from the synthetic self-test. A method that can earn one on a real model is the next step.

## Tests

```bash
python3 -m pytest -q tests                    # core, pipeline, replay; torch tests skip without torch
/mnt/models/finetune/venv/bin/python -m unittest discover -s tests -p 'test_backends_torch.py'   # hooks on CPU
```

## Limits

- Greedy decoding, one trial per question; no sampling statistics.
- Readings are token ids or whole-word surface matches. They can miss a paraphrased value; that shows up as `FAIL` or `NOT_EVALUATED`, never as a pass.
- The NF4 instrument wrapper is not exercised on the real model by the tests: that would be a model forward.
- Native generations are not yet cached across runs.

## Attribution

Designed, implemented and tested by Claude Code (an AI coding agent) under the direction of Patrick Kappeler, who set the mandate and approved the design decisions. The NF4 bench `setup()` it wraps was written by Codex. The independent verification of the replay is assigned to Codex.
