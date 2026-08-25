# Alder-v2 conditional terminal-residual preregistration

**Status:** closed at the phase A detector gate. No detector was frozen, phase
B did not run, and the final holdout remained sealed. See
[the result](research-alder-v2-terminal-residual-results.md).

## Question and claim boundary

Can the terminal residual mechanism developed on spent Galadriel material
certify the pre-existing synthetic association `Project Alder` / custodian ->
`Ada Lovelace` under a new prospective passive/agentless development suite and
the still-sealed Alder holdout?

This is one model-bound replication on one exact Gemma 4 E4B Q4K artifact. A
pass certifies one additional conditional behaviour patch; it does not measure
arbitrary edit compilation, cross-model transfer, or knowledge acquisition.

## Relationship to earlier work

Alder version 1 is closed as a failure. Its one-row payload selected target ID
45158 on 7/8 development positives, and its unchanged LOSO detector selector
classified 31/32 captures. The same passive prompt failed both stages. Its
holdout remained sealed.

The terminal residual adapter version 2 subsequently passed on spent
Galadriel development material. That result established a runtime seam and a
fixed deterministic compiler, not an Alder result. Alder-v2 applies that
already-fixed mechanism prospectively. It does not repair, overwrite, or
reinterpret version 1.

## Fixed artifact and candidate

- Base artifact:
  `/mnt/models/larql-vindexes/gemma-4-e4b-upstream-20260811-q4k.vindex`.
- Base binding:
  `sha256-manifest:c53cc460e0809529289375db6a90d84d038586d7cc005f19e401c8264066987a`.
- Payload: the existing single permitted Alder `COMPOSE` result at L31/F1625.
- Entity: `Project Alder`.
- Relation class: custodian / keeper / steward / caretaker.
- Target: `Ada Lovelace`.
- Target IDs: `[45158, 37937, 217262]`.
- Holdout: the original `multitoken_replication_alder_holdout.json`, whose
  prompts remain unread by this version until every development gate passes.

No candidate, target, payload row, target tokenisation, or base artifact may
be replaced after freeze.

## Frozen development material

Dev1, dev2, and dev3 are the 32 spent version 1 captures. Alder-v2 recaptures
them rather than reusing their numeric residuals. Dev4 is a new 16-prompt
prospective stress suite frozen before its first model run:

- four `11` positives, including passive `is overseen by`, care-of, and
  agentless `oversight ... falls to` surfaces;
- four `10` entity-only controls;
- four `01` relation-only controls using the same new surfaces;
- four `00` neither-factor controls.

The exact file is
`multitoken_replication_alder_dev4.json`. The deterministic union
`multitoken_replication_alder_development_v2.json` contains 12 positives and
36 controls and exists only to let the terminal compiler and development gate
reuse one resident E4B weight set. The four separate files remain the source
for LOSO suite boundaries.

No development prompt may be changed after the first capture starts. No
holdout prompt may be used for capture, detector fitting, compiler synthesis,
hard-negative mining, threshold selection, or implementation choices.

## Phase A: fresh residual captures and detector gate

Apply the exact version 1 payload without a contextual detector and capture all
prompt-position residuals for dev1 through dev4. Each suite receives exactly
one sequential, bounded E4B process; no processes overlap. These captures are
fresh and occur after the payload already exists.

Run `analyze_contextual_positions.py --deployable-only` on the four complete
suites. The diagnostic report may score the historical 42-layer,
ten-representation, four-family search space, but certification is restricted
before observation to candidates exactly representable by the current ABI:

- representation: `mean_no_bos`;
- methods: `raw_conjunction`, `unit_conjunction`, or
  `unit_factor_prototypes`, the latter serialized exactly as the difference
  between present and absent prototypes with zero thresholds.

Acceptance requires all **48/48** leave-one-entire-suite-out decisions correct
and a strictly positive worst signed margin. The highest accepted candidate is
selected by the unchanged lexicographic ordering: total correct, weakest suite
accuracy, decision margin, then ranking margin. If none is accepted, Alder-v2
fails and no terminal adapter or holdout run occurs.

Fit the selected candidate on all 48 development captures, serialize only its
runtime-equivalent two-factor detector, attach it to L31/F1625, reparse the
patch, validate dimensions and checksum binding, and hash it before phase B.

## Phase B: fixed terminal adapter and development gate

Use the unchanged softcap-aware terminal adapter compiler that passed the
Galadriel version 2 spike. With the contextual source patch loaded, capture
post-L41 last-position residuals for the 12 development positives. Every
positive must receive exactly one `ALLOW`; a detector rejection ends Alder-v2.

The compiler is fixed:

1. choose the positive with the lowest exact target-versus-runner margin;
2. differentiate that margin through final RMSNorm, logit scaling, and
   `30 * tanh(raw / 30)`;
3. project the gradient orthogonal to the unit payload down vector and require
   retained norm ratio at least `0.10`;
4. set `alpha_max` to five percent of the minimum synthesis residual norm;
5. require the fixed 65-point score grid to be non-decreasing within `1e-5`
   and to reach a minimum margin of at least `+0.75` at `alpha_max`;
6. use deterministic bisection to select the smallest alpha reaching `+0.75`
   within `1e-4 * alpha_max`;
7. serialize, reparse, checksum-check, and hash exactly one derived patch.

There is no alpha, layer, direction, scale, prompt, or threshold sweep.

Evaluate all 48 development prompts first. Activation passes only if:

- all 12 positives remain detector-allowed, apply the adapter exactly once,
  choose ID 45158, and have target margin at least `+0.5`;
- adapter norm is at most five percent of every positive reference residual;
- all 36 controls remain detector-rejected, apply no adapter, and retain
  complete f32 logits bit-identical to the contextual source path;
- every detector decision and both factor margins are bit-identical between
  reference and adapter paths.

Only after activation passes, cached development generation must emit the
exact three IDs on all 12 positives, arm once, and release after the span. All
36 controls must reject, never arm, and not emit ID 45158 first. Any failure
keeps the final holdout sealed.

## Phase C: one final holdout access

The dedicated `--alder-v2-holdout` mode first reparses the source and derived
patches, checks the base binding and vector dimensions, and hashes the frozen
adapter. Only then may it read the holdout file.

In one resident E4B process, the four positives and twelve controls must pass
the same activation invariants and then the cached-generation contract. The
candidate is certified only if all **4/4 positives and 12/12 controls** pass.
Report the fact-level outcome before the 16-prompt aggregate.

## Resource and evidence contract

Every full E4B invocation is sequential and runs in a unique foreground
`systemd --user` unit with `--wait --pipe --collect`, after checking the host
process list. Fixed limits are:

- `MemoryHigh=22G`;
- `MemoryMax=24G`;
- `MemorySwapMax=8G`;
- `OOMPolicy=stop`.

Capture units use `RuntimeMaxSec=1800`, development uses `5400`, and the final
holdout uses `3600`. Completion is established from the actual unit status and
process list, never from a detached wrapper notification.

Before phase A, a manifest records hashes of this document, source commit,
release binaries, zero-weight preflight output, payload, vindex manifest, the
four development suites, aggregate development suite, holdout, and scripts.
Captures, selected detector, contextual source patch, derived patch, evidence,
journals, and provenance remain under the private research directory. The
technical result document in this repository is the public source of truth;
career planning receives only a verdict and link.

## Stop rules

- Any phase failure stops all later phases.
- A cgroup kill, timeout, or runtime error is infrastructure-inconclusive for
  Alder-v2; it does not authorize relaxed limits or a repeat.
- Each declared E4B phase runs once. No rerun, candidate substitution, prompt
  edit, threshold movement, or alternative adapter is allowed in this version.
- The holdout is accessed by exactly one process and is never opened after a
  failed development gate.
- The actual outcome is reported as pass, scientific failure, or
  infrastructure-inconclusive. Alder-v2 then closes.
