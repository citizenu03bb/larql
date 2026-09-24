# Natural Binding Causal Pilot v1 — eligibility stop

**Status:** closed at the frozen baseline eligibility gate, 2026-09-24.
**This is not a causal negative:** no subspace was fit and no intervention ran.

The unedited Gemma 4 E4B native-Q4K artifact ranked the required complete,
bare single-token answer top-1 on **1/16** discovery factual prompts. The
protocol required **16/16**, so the controller correctly stopped after the
fixed baseline/replay phase. The only passing prompt was `The atomic number
of hydrogen is:\n`, with answer `1`.

## Budget and validity

| Item | Observed |
|---|---:|
| Baseline/capture forwards | 20 |
| Disabled-hook replay forwards | 20 |
| Exact full-logit replay matches | 20/20 |
| Total attempted/completed forwards | 40/40 |
| Planned maximum across both stages | 368 |
| Factual discovery prompts eligible | 1/16 |
| Fitted directions | 0 |
| Intervention forwards | 0 |
| Confirmation forwards | 0 |

All stored logits and captures are finite. A separate audit recomputed every
artifact hash, full-logit digest, top-1 ID, recorded answer logit and replay
comparison from the saved f32 arrays. It reproduced the frozen analyzer's
`eligibility_stop`. The actual confirmation gate rejects this verdict before
worker startup; there is no confirmation directory or direction artifact.
The unused 328-forward allowance does not authorize another configuration.

## Factual results

Every prompt ends with a newline. T0 is `The {relation} of {entity} is:\n`;
T1 is `{entity} has {relation}:\n`. The observed column contains the literal
first tokenizer piece, not a decoded multi-token answer. Target rank is a
post-run descriptive calculation from stored logits; it is not a replacement
metric and cannot change eligibility.

| Element | Construction | Relation | Required | Observed top token | Required-token rank | Gate |
|---|---|---|---|---|---:|---|
| hydrogen | T0 | symbol | `H` | `A` | 15 | fail |
| hydrogen | T0 | atomic number | `1` | `1` | 1 | pass |
| hydrogen | T1 | symbol | `H` | `The` | 20 | fail |
| hydrogen | T1 | atomic number | `1` | `The` | 3 | fail |
| helium | T0 | symbol | `He` | `$$\` | 37 | fail |
| helium | T0 | atomic number | `2` | `$\` | 41 | fail |
| helium | T1 | symbol | `He` | `$\` | 16 | fail |
| helium | T1 | atomic number | `2` | `What` | 31 | fail |
| lithium | T0 | symbol | `Li` | `$$\` | 195 | fail |
| lithium | T0 | atomic number | `3` | `A` | 83 | fail |
| lithium | T1 | symbol | `Li` | `A` | 19 | fail |
| lithium | T1 | atomic number | `3` | `A` | 33 | fail |
| beryllium | T0 | symbol | `Be` | `$$\` | 84 | fail |
| beryllium | T0 | atomic number | `4` | `1` | 236 | fail |
| beryllium | T1 | symbol | `Be` | `a` | 78 | fail |
| beryllium | T1 | atomic number | `4` | `a` | 34 | fail |

Several top tokens are prose or LaTeX prefixes; other prompts produce an
incorrect bare token. This shows that these frozen completion prompts do not
reliably elicit the required answer form on this artifact. It does **not** show
that the model lacks the underlying chemical knowledge: no continuation was
generated to determine what would follow those prefixes. Nor does it test or
refute the rank-one causal hypothesis, since eligibility failed first.

The raw completion format was specified in advance, with architecture-correct
BOS and no chat wrapper. Rust matched all 40 authored prompt/answer encodings
against the model-free tokenization record before any weights were loaded.
Changing the wrapper, stripping prefixes, accepting answer aliases, selecting
new facts, or continuing generation would be a new protocol; none was done.

## Execution and provenance

- Design commit: `7c307eda11b50fcb74ebdcb3592bb270e3acb3a8`.
- Frozen instrument: `dc0aafecd24cbb491fbeffa098fe0a608c152aef`.
- Unit: `larql-nbcp-v1-discovery-dc0aafec-20260924.service`.
- Invocation: `b30875a12b0040cfbc1a592f8bb151d3`.
- Start: 2026-09-24 21:48:53 Europe/Paris.
- End: 2026-09-24 22:01:18 Europe/Paris.
- Service wall time: 12min 24.666s; CPU time: 1h 46min 1.753s.
- Reported peak memory: 21G; peak swap: 3.3G.
- Exit: `0/SUCCESS`; both controller and worker were verified gone afterward.
- Limits: MemoryHigh 22G, MemoryMax 24G, MemorySwapMax 8G, RuntimeMaxSec 90min.
- Pre-run checks: 16 Python tests, 3 actual-hook Rust tests, Rust/Python
  tokenization parity, and streaming hashes of every local vindex file.

The first preflight launch attempt was blocked by an automatic approval-review
usage limit and executed nothing. After the user's `continue`, the same frozen
preflight passed and this single discovery run began. It was not a scientific
retry and consumed no prior model forwards.

Artifact binding remains
`sha256-manifest:c53cc460e0809529289375db6a90d84d038586d7cc005f19e401c8264066987a`.
Here unedited means the instruction-tuned checkpoint's vindex without a LARQL
overlay. The canonical binding covers the index's recorded checksum manifest;
the new execution manifest additionally hashes all actual vindex files.

Evidence directory:
`/home/pat/Projects/larql-research/natural-binding-causal-pilot-v1/run-20260924`.
The run retains 40 compressed raw f32 artifacts, the append-only attempted /
completed ledger, frozen execution manifest, record summaries, verdict and
independent verification. The machine-readable repository summary is
`docs/research-data/natural-binding-causal-pilot-v1/result-summary.json`.

| Evidence | SHA-256 |
|---|---|
| `execution-manifest.json` | `91b6c928da9d9fbf13692c2099606a9a9acbab1cb9875c240a4c826c758b4d2c` |
| `discovery/verdict.json` | `0f6fd9f2d68b16ca2460662712e479b1f4c3e55430dfec7b9bf42a4015a5f7d3` |
| `discovery/records.json` | `0a276a82114247888ac08a4fef792b2aa7c3a5ca571f841dbb4f04248a4ec6fb` |
| `discovery/events.jsonl` | `1d05a5e7af58293601b411fcb1840fc6005d11d122f8e84bb59f4eefb62f0367` |
| `independent-verification.json` | `c7872f0d368b7754435a763fb4cea4acd7a83e435bd6623ae3bdf9c6b9b30e2c` |
| `confirmation-refusal.json` | `33365271ad87816f0fa40e3839c7341bdba44514b4805a947fcae5c5253de340` |

The approved pilot is closed at eligibility. The previous L11/L13 causal
negative remains unchanged, and all previous sealed holdouts remain untouched.
No new run, prompt repair or hypothesis extension is implied by this result.
