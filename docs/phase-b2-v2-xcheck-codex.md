# Independent B2 v2 cross-check — 2026-10-27

**The fixed gate confirms STOP:** 0/5 at L4, L8 and L12, below 2/5.
All 15 complete verdicts independently recalculate to **REFUSED**. The
native feasibility gate is **4/5**, unlike v1's 1/5. There is **one open
control-status interpretation difference**, on Bolt L8 C7; it changes
neither a verdict nor the development gate. No model forward, new solve,
identity-C ablation, test-set access or push was performed for this check.

## Basis and independence

Inputs are the raw v2 run, the public specifications and written protocol
at Claude's result commit `2ee2ee4f`, and design sections 3, 10, 12 and 13.
`tools/rome_solver/xcheck_b2_v2.py` imports no harness judges. It reconstructs
native qualification and C0--C8 from saved generations, declared form,
plans, raw diagnostic values and reload records. The expected headline
results were supplied in Pat's brief before the script was written; this
is explicitly **not** another blind freeze. Six model-free logical tests
cover a synthetic positive, truncation, entity substitution, value-based
off-target readings, random controls and reload mismatch. All pass.

The [independent replay](research-data/phase-b2-v2-xcheck-codex.json) pins
the script and inputs. Its full first output, including raw text/IDs, is
preserved at `phase-b-rome-20261026/b2-v2-independent-replay-first-codex.json`;
the versioned copy is compact. No v1 verifier, output, report or uncommitted
v1 file was changed. Neither side was corrected to make results agree.

## Recomputed results

| Check | Independent result |
|---|---|
| Native feasibility | Beethoven, Bolt, Eiffel, Toyota: feasible; Kilimanjaro: not feasible |
| Native question qualification | 195/195 agreements |
| Full verdicts | 15/15 REFUSED, 15/15 agreements |
| C0--C8 statuses | 134/135 agreements; C7 difference below |
| C1 and C2 statuses used for selection | 30/30 agreements |
| Real v2 forwards | 7,593, exactly recounted |
| Gate | L4 0/5, L8 0/5, L12 0/5; STOP_BELOW_2_OF_5 |

C1 passes for Beethoven L8/L12, Toyota L8 and Bolt L12. Their C2 outcomes
are respectively NOT_EVALUATED, NOT_EVALUATED, NOT_EVALUATED and FAIL.
The original five-fact denominator is retained. No case is replaced or
dropped to improve the score; NOT_EVALUATED never counts as PASS.

## Answer to the quorum's qualification

**Yes: the 0/5 must be reported partly as non-evaluation, rather than
converted into evidence that all attribute values were corrupted.** In
particular, at L8 the two efficacy positives have a correctly qualified
native A1 and an edited A1 truncated at 64 tokens:

| L8 fact | Native A1 | Edited A1 | Other attribute | C2 |
|---|---|---|---|---|
| Beethoven | 55 tokens, EOS | 64 tokens, no EOS | nine symphonies preserved | NOT_EVALUATED |
| Toyota | 36 tokens, EOS | 64 tokens, no EOS | Corolla preserved | NOT_EVALUATED |

The factual gate count is zero **established** dual passes. If both unknown
C2 outcomes were hypothetically PASS, L8 would count 2/5. That counterfactual
does not establish preservation, predict what later tokens would say, or
authorize extending these generations. The layer-wise sensitivity counts,
assuming C2 preservation only where C1 already passes and C2 is unknown,
are L4 **0 to 0**, L8 **0 to 2**, L12 **0 to 1**. They are not success
probabilities. The operational stop follows from the fixed requirement
for observed PASS, not from resolving those unknowns as FAIL.

Beethoven L12 is a useful caution: its capped A1 already says hearing
loss/deafness, yet still cannot PASS under the EOS rule. A blanket claim
that truncation demonstrates lost factual knowledge would be too strong.
The valid conclusion is failure to demonstrate the required combination
on this question set, instrument and decoding budget.

**The selectivity failures are independently observed.** Every C1-positive
edit has C6 FAIL on completed answers. Beethoven L8 makes Mozart and
Chopin Italian; Beethoven L12 additionally makes Bach Italian. Toyota L8
makes Honda German. Bolt L12 answers `swimmer` for Federer and corrupts
the subject's attributes. The answer-only readings also catch Giuseppe
Verdi replacing Beethoven at L12 and Volkswagen replacing Toyota at L8.
These remain counterexamples to full certification even if the unknown
C2 outcomes were hypothetically favourable. In particular, opening the
C1+C2 development gate in that counterfactual would not make these edits
CERTIFIED.

## Open C7 interpretation difference: Bolt L8

For unseen question E2, both the in-session and fresh-reload generations
have **64 identical IDs, capped=true, eos=false**. The other three unseen
answers are readable and identical, and artifact bytes match throughout.

The harness reports C7 PASS: `reloaded answer identical`. The independent
reading applies the design's all-controls rule, `A cap hit is unreadable,
not a pass`, and reports NOT_EVALUATED. This is a disagreement about
whether C7 may certify reproducibility of an identical **capped prefix**,
or requires a complete answer. It is not a disagreement about hashes,
IDs or process separation. Both readings are preserved for Pat's
arbitration; no harness or checker repair was made. Bolt L8 is REFUSED
under both readings, and C7 is outside the C1+C2 layer score.

## Integrity, A1 and limits of the evidence

The existing v1 record auditor ran unchanged on v2. It verifies all 510
generations against the pinned local tokenizer, EOS/cap flags, fingerprints,
counter receipts and exact source/saved adapter bytes. The **same 15
adapters** are used, with no new optimization. The raw 7,353 generated
tokens plus 240 diagnostic forwards sum to **7,593**. Native IDs are
identical across the three layers for each fact.

The driver log contains **105 distinct successful services**, seven per
case, with no duplicate service launch. The first 15 native services cover
the five L4 feasibility cases; the feasibility event precedes every edited
service. Three additional native services for Beethoven L8 run between
that event and the first edited service, as allowed. The A1 repair is
recorded; all five initially skipped L4 edited/reload portions are now
present, without repeating their native service pairs. The raw index is
byte-identical to the versioned copy at `2ee2ee4f`.

The zip's SHA-256 is
`7e1b9eab1c416729f6d0a7837346d2e7c914d8c625f69cf68c468525f326563b`.
All **442 payload file hashes** match its manifest; all **437 live run
files** match their archived copies. The five additional payload files are
the session's driver, two stdout logs, protocol and results report. The
CRC check also passes. This is an independent archive/byte check, not a
claim to have replayed model inference.

C4 ratios are independently recomputed from the saved native/edit/random
margins; declared efficacy token IDs match bare or spaced first tokens
on all 180 relevant boundary entries. Raw diagnostic prefixes, full logits
and the actual random adapter arrays are not archived, so this check does
not reproduce their model computation or independently measure the used
random adapter's norm. The peer fingerprints also omit full checkpoint
hashes; pinned metadata/source hashes and the separate B1 preflight remain
the available linkage. These limitations do not erase the free-generation
C2 non-evaluations or completed C6 counterexamples.

Receipts: [records](research-data/phase-b2-v2-records-audit-codex.json),
[driver](research-data/phase-b2-v2-driver-audit-codex.json),
[archive](research-data/phase-b2-v2-archive-audit-codex.json),
[boundary tokens](research-data/phase-b2-v2-boundary-token-audit-codex.json).

## Reproduction

Use a new output filename; this does not load a model:

```sh
python3 tools/rome_solver/xcheck_b2_v2.py \
  --root ~/Projects/larql-research/phase-b-certification-v2-20261026 \
  --out /tmp/b2-v2-independent-NEW.json
```

The stop is confirmed without amending its rule. The identity-C ablation
is not reached, and no B3 method freeze or unblinding is authorized by
this check. Any subsequent experiment needs its own decision; longer
answers cannot be silently substituted into the closed v2 result.
