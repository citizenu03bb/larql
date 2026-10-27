# Phase B, D1: diagnostic of the 15 B2 edits, frozen before any D1 forward

*Claude Code, 2026-10-27. Requires Pat's explicit approval before the first service. The Wikipedia arm also requires a separate download approval (§3). Development facts only; the sealed test set is not opened.*

**Mandate line.** B2 refused 15 edits but did not say why. D1 measures, without generation or a new solve, whether each edit fires at the subject, whether its value moves the answer, and whether the dose that reaches neighbours predicts the observed bleed under three covariances. The public negative can then name its cause instead of "cause not established".

## 1. What is held fixed

- **The edits.** The 15 B2 edits: Codex's solves, unchanged, at `~/Projects/larql-research/phase-b-rome-20261026/solves/<fact>-L<k>/edit`. Their algebra was cross-checked model-free, 15/15.
- **The comparison material.** The v2 specs, and the v2 native generations and C6 outcomes, read from `~/Projects/larql-research/phase-b-certification-v2-20261026/`.
- **The instrument.** Gemma 4 E2B-it NF4, through Codex's frozen `setup()`: deterministic, offline, no cache.
- **What a covariance arm changes.** Only V. For a covariance M, V_M = M⁻¹k\* / (k\*ᵀM⁻¹k\*), so V_Mᵀk\* = 1 in every arm. U = v\* − Wk\* is the B2 value, unchanged; v\* does not depend on the covariance in ROME's solve. The three arms are:
  - **C** (B1, local text): the B2 edits as certified.
  - **I** (identity): V = k\* / ‖k\*‖².
  - **W** (Wikipedia): a covariance estimated with Codex's unchanged B1 code on WikiText-2, same size and settings as B1 (§3).

## 2. Measurements

### M1, dose (capture only)

For each fact, one native forward per prompt, with the down_proj input captured at blocks 4, 8 and 12 in the same forward. The prompts are S1, E1–E4, O1–O4 and W1, in the chat template, exactly as certified.

The dose of an edit at a position with key k is **a = Vᵀk**. It is the fraction of the subject's update the edit adds there: the output changes by a·U, and a = 1 at k\*. For each prompt we report:
- **the subject dose**, at the last subtoken of the subject surface. For E and S prompts the subject is the fact's subject. For O prompts it is the neighbour named in `tools/edit_cert/facts/phase-b-d1-neighbours.json`, a table fixed in this commit;
- **the maximum dose** over all prompt positions, and where it occurs.

Doses are computed model-free from the saved keys, for each arm.

### M2, probability probe (teacher-forced, no generation)

For each prompt among S1, E1–E4 and O1–O4:
- **The prefix.** From the recorded v2 native answer, the prefix is the answer text up to the first word-bounded occurrence of a native true value: the fact's old value for S and E, the question's accepted value for O. The search starts after the answer's own mention of the subject, or of the neighbour, so that "Mount Kenya is located in Kenya" reads the second "Kenya".
- **The two continuations.**
  - **true**: the value found in the native answer.
  - **new**: when the true list and the fact's new list have the same length, the new surface at the same position as the true value found (Japan with Germany, Japanese with German); otherwise the first new surface. Its first letter's case follows the true value. The pairing is not always grammatical ("South Korean" against "South Germany"). Δm, which compares the same pair across conditions, is the main quantity, not the absolute m of off-target items.
- **The margin.** m = log P(new | prompt + prefix) − log P(true | prompt + prefix). Both are summed over the tokens from the first token where the two sequences differ.
- **Exclusions.** A prompt whose native answer does not contain a true value is excluded and reported. So is an O prompt whose true values include the fact's new value (Phelps, the Colosseum, Volkswagen), since it cannot show bleed.

Conditions, all on the same prefixes:
- native;
- each edit under arms C, I and W (W only if §3 is approved);
- the random rank-1 counterpart of each arm-C edit, with equal column norms and the seed of the spec.

The shift is Δm = m_condition − m_native.

### M3, served-module check (module calls, not model forwards)

For each edit and arm, the loaded NF4 down_proj with the adapter hook is applied to k\*. The checks are:
- the served output without the adapter matches the saved Wk\*, within a relative error of 2% of ‖Wk\*‖;
- the served output with the adapter equals Wk\* + U(V_Mᵀk\*) = v\*, within 5% of ‖v\* − Wk\*‖.

A failure is reported as an implementation finding, and the readings that depend on it are suspended.

## 3. The Wikipedia covariance (arm W), only with a separate download approval

- **The file to download.** `wikitext-2-raw-v1.zip`, 4 721 645 bytes, from `https://huggingface.co/datasets/ggml-org/ci/resolve/main/wikitext-2-raw-v1.zip`. Its LFS SHA-256, published by the host, is `ef7edb566e3e2b2d31b29c1fdb0c89a4cc683597484c3dc2517919c615435a11`.
  - The download is refused if that hash differs.
  - The archive is unpacked into its own directory, `~/Projects/larql-research/datasets/wikitext-2-raw-v1/`, and only `wiki.train.raw` is read, as text.
  - The licence is CC BY-SA. The corpus is not redistributed.
- **The sample.**
  - Records: the training file split into articles at its top-level headings (lines ` = Title = `), keyed by order.
  - Windows: Codex's `fixed_windows` with the quota `{wiki: 200}`, 512 tokens per window and seed 20261026. This is the same number of windows as B1, so the domain of the text is the only change.
  - The tokens and their hashes are fixed before any forward.
- **The covariance.** Codex's unchanged `collect_model_covariances` (commit `5f199533` of `research/phase-b-rome`): blocks 4, 8 and 12, the B1 positions, ridge 1e-4. This is one service of 200 forwards; B1 took 861 s.
- **A known limit.** 200 × 64 = 12 800 keys for a 6144-dimensional second moment, as in B1. ROME used about 100 000 Wikipedia samples. Arm W tests the domain of the text, not the sample size.
- **Overlap with the facts.** Whether the sampled windows mention a dev subject or neighbour is checked by string search and reported.

## 4. Budget, services, limits

| Service | Content | Forwards |
|---|---|---|
| 0 | W covariance, only if §3 is approved | 200 (512 tokens each) |
| 1–5 | one per fact: M1 captures (10) + M2 for native, 3 edits × 3 arms and 3 random counterparts (13 conditions × ≤ 9 prompts × 2 continuations) + M3 module calls | ≤ 244 each |

- **Total:** at most 1 420 forwards, all short except service 0.
- **Limits:** every service runs under `tools/edit_cert/run_d1_service.sh`. It applies the shared `flock`, `systemd-run`, 22G/24G/8G and 1800 s, deterministic cuBLAS and offline mode. It refuses to start without `--i-have-approval` or with an uncommitted protocol.
- **Order:** service 0 runs first, so that the W arm's V can be built before services 1–5. Without §3, the W conditions are dropped and services 1–5 run at most 190 forwards each.
- **Stop rule:**
  - any failed service, nonfinite value or budget overrun stops the series, with no retry and no parameter change;
  - any change of prompt, prefix or threshold after a D1 forward is a protocol amendment, recorded as such.

## 5. Readings, fixed now

The thresholds are conventions chosen before the data. They are reported with the values, never tuned.

**Validity of the probe.**
- **On the four edits with C1 PASS in B2** (Beethoven L8 and L12, Toyota L8, Bolt L12), arm C must give m > 0 on at least 3 of the 4 unseen E prompts. Otherwise the probe is declared invalid, and only M1 and M3 are reported.
- **Random counterparts:** the median |Δm| over the E prompts must stay below 1 nat. Otherwise the probe noise is reported next to every Δm.

**Efficacy class of each arm-C edit, from the unseen E prompts.** Let *dose* be the median subject dose, *shift* the median Δm, and *flip* the number of E prompts with m > 0 in each subject order.

| Class | Condition | Meaning |
|---|---|---|
| K, key | dose < 0.5 | the edit does not fire at the subject in unseen phrasings |
| V, value | dose ≥ 0.5 and shift < 2 nats | it fires; its value does not move the answer |
| S, silent | shift ≥ 2, flip ≥ 1 in each order, C1 FAIL in B2 | probability moved; greedy generation did not state it |
| P, pass | C1 PASS in B2 and flip ≥ 1 in each order | consistent with B2 |
| U, other | anything else | reported as is |

**Does dose predict bleed (arm C)?**
- **The pairs.** Every (edit, O1–O3 prompt) that can show bleed is one pair. It is labelled bleed when its B2 C6 outcome states the edited value.
- **The statistic.** The AUC of the neighbour subject dose for that label, computed only if there are at least 3 pairs of each label.
- **The reading.** AUC ≥ 0.8: the dose predicts the bleed. AUC ≤ 0.6: it does not. Anything between is inconclusive.
- **Also reported:** the Spearman correlation between the neighbour dose and the neighbour Δm.

**Does the covariance matter (arms I and W against C)?**
- **The selectivity index of an edit under an arm** is the median neighbour subject dose (O1–O3, bleed-possible) divided by the median E subject dose.
- **An arm improves selectivity** if its index is at most 0.67 times arm C's on at least 3 of the 4 C1 PASS edits, while it keeps flip ≥ 1 in each order on those edits.
- **An arm is worse** if its index is at least 1.5 times arm C's on at least 3 of the 4.
- **Otherwise the arm is neutral.**

**What follows, decided now, each step needing Pat's approval.**
- **W or I improves selectivity:** D2 re-certifies those edits with that arm's V in the full harness. There is no new v\* and no new layer. The harness rules and the B2 gate are unchanged.
- **No arm improves selectivity and the dose predicts the bleed:** the bleed is a property of the subject and neighbour keys at these blocks, which a different covariance does not separate. The write-up says so. The next method question would be a constraint on several keys (MEMIT or AlphaEdit over a range), only after P1.
- **Mostly K or V classes on the C1 FAIL edits:** efficacy fails in the edit itself, not in the reading of the bench.
- **Mostly S classes:** efficacy is hidden by greedy generation. The bench's generation criterion is stricter than the probability criterion of the literature, and the write-up says so. The verdicts are not changed.
- **M3 fails:** the implementation is in question. Every reading above is suspended until it is explained.

## 6. Prediction (not a reading rule)

- **Probe:** valid.
- **Efficacy classes:** the L4 edits and Eiffel are mostly class V (they fire, but the value is too weak), not K.
- **Dose and bleed:** the dose predicts the bleed (AUC ≥ 0.8), with neighbours of the same type at a dose of 0.3–0.8 and unrelated O4 subjects below 0.2.
- **Covariance arms:** arm W is neutral or improves selectivity slightly; arm I is worse than C.

## 7. Outputs and verification

- **Outputs:** `~/Projects/larql-research/phase-b-d1-20261027/`, with the plan (hashed), the service records (captured keys in float32, per-token log-probabilities, M3 errors), the analysis and an index.
- **Archive:** a report, an index copy in `docs/research-data/`, and an archive with a SHA-256 manifest and a restore check.
- **Verification:**
  - The analysis is model-free and runs from the saved records. Claude wrote both the analysis and the run.
  - Codex is asked to recompute the doses, margins, classes, AUC and arm comparisons from the records with its own code.
