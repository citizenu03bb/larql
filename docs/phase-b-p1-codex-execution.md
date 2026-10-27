# Codex P1 execution: approved A1, covariance first

Pat's brief 40 approves P1-2 and Codex's five solves, including the solve
cap of 500 sequence-forwards and 25 Adam updates. The primary A1 protocol
on `research/edit-certification` fixes the inputs. No Claude independent
solve source is read or imported. The sealed test set remains unopened.

## Native gate cross-check

The independent native qualifier reproduces 5/10 feasible facts and 478
actual forwards, with no qualification or feasibility difference. The
five are BMW, Germany, Gretzky, Nintendo and Shakespeare. The negation
reader uses the frozen limit of two intervening words. Shakespeare E2
therefore qualifies mechanically on `not known to the British government`.
Excluding that questionable E2 still leaves two qualified unseen phrasings
in both orders and both attributes: Shakespeare remains feasible, and
the gate remains 5/10. This sensitivity does not silently amend a reading.

## P1-2 frozen execution details

- Module input: `transformer.h.17.mlp.c_proj`, width 6400; GPT-2 XL fp16,
  local model files verified against their published SHA256SUMS.
- Training text: the two verified WikiText-103 parquet shards, in filename
  order. Each row is one raw line; existing final newlines are retained,
  otherwise a newline is appended, including blank rows. Top-level `= Title =`
  headings begin articles; section headings do not. Empty preamble is
  discarded. Articles have string ordinal IDs, starting at zero.
- Codex's unchanged `fixed_windows`: `{wiki: 1200}`, width 1024, seed
  20261026. Its existing lexical-ID order, record shuffle and disjoint
  full-window selection are retained. All window hashes are written before
  any model forward. Every one of 1,228,800 token positions is counted.
- Three fresh model services, exactly 400 sequence-forwards each. Each
  uses the same model loading configuration as the completion instrument;
  the actual attention implementation is recorded. Capture occurs before
  c_proj, in one transformer call without cache or generation.
- Each part accumulates the unchanged `SecondMoment` float64 sum with
  **no ridge**. Combine all three sums in float64, divide by the total
  count, and add **one** ridge, `1e-4 * trace(moment) / 6400`. Store float32
  `covariance/C17.npy`, validate SPD, then atomically publish status.json
  with the matrix hash, 1200 forwards and sample details. No partial matrix
  is presented as the shared covariance.
- Preparation and model workers run through the shared flock and bounded
  user systemd service (22G/24G/8G, 1800 s). Preparation has zero model
  forwards. Every real covariance service has a fresh synthetic Conv1D
  rehearsal checking 3×400 counters, capture count, partial-sum aggregation,
  single ridge and refusal of a 401st sequence.
- Stop on an actual failed/partial service, hash drift, nonfinite value,
  shape, time or budget error. No automatic retry. Existing completed
  parts may be verified and reused; an existing failed part is refused.

The first synthetic Conv1D fixture exposed a float64/float32 test-input
mismatch; the fixture was corrected before any real model call. Fresh
successful rehearsal output is preserved separately from that attempt.

Covariance publication is the interface for Claude's independent solves.
Codex's port and its own five solves are a separate committed implementation
and will consume exactly this covariance and the already generated 21
templates. Their SHA-256 is
`b7dce8e1fa21e362e63e4ea45fc5a25030cce7a61781dcdd55f0c0216afebf15`.
No templates are regenerated. No push is authorized.

Published sources read as reference, never executed:
[GPT-2 XL parameters](https://github.com/kmeng01/rome/blob/main/hparams/ROME/gpt2-xl.json),
[value objective](https://github.com/kmeng01/rome/blob/main/rome/compute_v.py),
[key direction](https://github.com/kmeng01/rome/blob/main/rome/compute_u.py).
