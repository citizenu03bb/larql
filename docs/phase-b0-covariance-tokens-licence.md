# B0 covariance-token redistribution review — Codex, 2026-10-27

This resolves the authorized tracked-file action in Pat's briefs 50/51.
It does **not** authorize a push, public release, rewrite, or deletion of
private research history. The later brief 51 says the other branch was
pushed; brief 50 is the earlier readiness snapshot, not new push approval.

## Exact inputs verified

The 819328-byte NPY introduced in **70f0cd6c** has shape (200,512), int64:
40 ADR windows, 80 Emotion windows and 80 Banking77 windows. Its SHA-256 is
`ef9ee0423bbc5534eca7d450f6cb29bb4eba0802f4421165b844f97ec3bf3cdd`;
Git blob **740ee39a76a010aaa23bd227e9bba2c02e968440**. Token IDs can be
decoded to source text, so the file is treated as redistributed input
content, not anonymized statistics.

| Source actually used | Cache/Hub revision | Input Arrow SHA-256 |
|---|---|---|
| dair-ai/emotion, config split, 0.0.0, train, 16000 rows | cab853a1dbdf4c42c2b3ef2173804746df8825fe | d362b3fb7dabba0613533182e3e0dce2340972e5f90d9ee02783de28bdf8e095 |
| legacy-datasets/banking77, config default, 0.0.0, train, 10003 rows | f54121560de48f2852f90be299010d1d6dc612ec | 565fe069366117758d60332fbef73dd22a4c32969c0c9be7bf9869eac533b6f0 |

Both Arrow file hashes match the original source manifest. The cache's
Hub refs and snapshot directories confirm those revisions; these are
not inferred merely from the prepared Arrow directory names. Both local
dataset_info.json licence fields are empty, but the corresponding primary
dataset cards are present in those exact snapshots and were read directly.

## Licensing findings

**Emotion: no general public-redistribution permission established.** The
[exact cached card](https://huggingface.co/datasets/dair-ai/emotion/blob/cab853a1dbdf4c42c2b3ef2173804746df8825fe/README.md)
has `license: other`, and its licensing section says: “The dataset should
be used for educational and research purposes only.” Card SHA-256:
`7b10e5670f3e5b0a12274e90aad8f85fc48d8e25eccb4e4c16933cd8b7dfc373`.
The current [primary card](https://huggingface.co/datasets/dair-ai/emotion/raw/main/README.md)
and [origin README](https://github.com/dair-ai/emotion_dataset#usage)
give the same usage limitation. I found no broader redistribution grant
there and do not infer one from accessibility or from a citation request.
This is an unresolved permission question, not a claim that every possible
research quotation or copyright exception is prohibited.

The web reader could not retrieve the revision-specific HF raw URLs;
historical card contents above were therefore verified from the local
Hub snapshots, while current owner sources were checked online. No dataset
or executable code was downloaded to perform this review.

**Banking77: CC BY 4.0 confirmed.** Its exact cached card declares
`cc-by-4.0` (card SHA-256
`a3515557cb51062eff114ddf8e15554e134094a8ef6dfd482654df606af4be1b`).
The [original PolyAI source licence](https://github.com/PolyAI-LDN/task-specific-datasets/blob/master/LICENSE)
and [current dataset card](https://huggingface.co/datasets/PolyAI/banking77)
agree. CC BY 4.0 permits sharing subject to its conditions; the
[licence](https://creativecommons.org/licenses/by/4.0/)
requires attribution, a licence link, and an indication of modifications.
The adjacent DATA-SOURCES notice now identifies the creators/source,
licence and token-sampling transformation. This does not clear the Emotion
part of the mixed array.

## Tracked-tree action and reproducibility

Following the explicit unclear-permission condition in brief 51, remove
`docs/research-data/phase-b0-preflight/covariance-tokens.npy` in a dedicated
follow-up commit. Keep the original source/sample manifest, original
B0 manifest, protocols and their historical hashes **unchanged**.

The [adjacent source/attribution note](research-data/phase-b0-preflight/DATA-SOURCES.md)
explains the public datasets, exact inputs and private regeneration.
`tools/rome_solver/regenerate_b0_tokens.py` reproduces the array using the
frozen sampling helper and tokenizer metadata, without loading pretrained
weights or doing a model forward. A private copy was regenerated at
`$HOME/Projects/larql-research/phase-b0-private-inputs-20261027/covariance-tokens.npy`.
All 200 windows, stream hashes, raw token hash and complete NPY hash match.
The [receipt](research-data/phase-b0-tokens-regeneration-codex.json) contains
hashes/metadata only, not dataset text or token IDs.

The NPY was tracked despite the repository's existing `*.npy` ignore
rule. No ignore-rule change is needed.

## Consumers and deliberate compatibility impact

- `prepare_b0.py` originally writes the file and its receipt. It refuses
  an existing output directory; do not run it over the frozen receipts.
- `run_phase_b.py:verify_b0` traverses phase-b0-manifest.json, which records
  this file. It is expected to fail while the private input is absent.
- `make_preflight` hashes it, and real B1 loads it via numpy.load. A clean
  clone cannot run those original real steps until the privately obtained,
  hash-matching input is restored at their expected location.
- Existing unit tests use synthetic windows, not the removed tracked
  array. The later P1-5 offline replay and token/log-probability audits
  do not read this file. Historical measured results are not changed.

The frozen reader and manifest are not edited to hide the missing input.
This records withdrawal of a reconstructible data artifact, not a new
research run or a change to a measurement.

Validation after withdrawal: **16 synthetic CPU ROME tests pass** without
the tracked NPY. Calling the frozen read-only verify_b0 function fails on
the missing token file as expected, before any model construction or
forward. Private regeneration passed all original byte/hash checks.

## History is still a publication blocker

Deleting a file at the tip does **not** erase it from earlier commits.
The blob remains reachable from 70f0cd6c through this branch. Pushing the
ordinary branch would expose it despite the deletion. Attribution for
Banking77 does not settle the separate Emotion permission uncertainty.

Accordingly, **research/phase-b-rome is not cleared for push** by this
commit. Resolve that exposure either with suitable redistribution
permission or an explicitly approved publication history/export that
excludes the blob. A rewrite or new sanitized publication history needs
Pat's decision under brief 51; neither was performed. This is not an
assertion that a licence requires deletion of private historical records.
The original private evidence and this branch's ancestry are preserved.

Claude's secret scan still needs to be rerun before any later authorized
publication. No claim is made here that the two agents' branches must be
merged, or that existing frozen absolute paths must be rewritten.
