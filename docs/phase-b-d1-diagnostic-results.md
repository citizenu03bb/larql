# Phase B D1: diagnostic of the 15 B2 edits — results

*Claude Code, 2026-10-27. Protocol `docs/phase-b-d1-diagnostic-protocol.md`, approved by Pat with the WikiText-2 download. Plan `50668f20…d9e1`. Development facts only; the sealed test set was not opened. Claude wrote both the run and the analysis. Codex's model-free recomputation is pending.*

## Summary

- **The run.** It went as planned: 6 services and 1 342 forwards, each service at exactly its planned count, with no failure, no retry and no change after the plan.
- **The probe (M2) is invalid by its own pre-registered rule.** No efficacy class is reported, and no reading uses the margins. On Bolt L12, which passed C1 in generation, the teacher-forced margin is positive on only 1 of the 4 unseen prompts, and the rule required 3.
  - A likely cause, not established: the probe compares fixed surface forms ("track", "sprinting" against "swimming") at the native answer's prefix, while the edited generation reaches the value through other words.
  - The margins stay in the records for Codex's recomputation.
- **M3 passes on every edit and arm.** Applied to k\*, the served NF4 module reproduces Wk\* within 0.05% and v\* within 0.13% of ‖v\* − Wk\*‖. The implementation applies the update it was given.
- **M1, the edit fires at the subject in every unseen phrasing.** The median dose at the subject over the four unseen prompts lies between 0.73 and 1.20 on all 15 edits. No edit is in class K. Where B2's efficacy failed, the cause was not a key mismatch between the seen and unseen phrasings.
- **M1, the edit fires almost as strongly at same-type neighbours.** Under arm C, the median neighbour dose is 0.71 (range 0.03–0.91). Mozart, Bach and Chopin receive 0.71–0.88 of Beethoven's dose, and Honda 0.83–0.91 of Toyota's. Unrelated subjects (the O4 capitals) receive a median of 0.23.
- **The dose does not separate the neighbours that bleed from those that do not.** The AUC is 0.69 on 13 bleed pairs against 23 kept, which is inconclusive under the protocol's thresholds. High neighbour doses occur with and without bleed. At L4, for instance, Beethoven's neighbours get 0.77–0.88 and none bleeds, since that edit does not flip Beethoven either.
- **The covariance does not change the picture.** Arms I (identity) and W (WikiText-2) are both **neutral** against arm C. No selectivity index falls to 0.67 times arm C's on any of the four C1 PASS edits. The median neighbour dose is 0.71 under C, 0.76 under I and 0.66 under W.

## Reading, as fixed in the protocol

- **No arm improves selectivity.** By the protocol, **no D2 re-certification follows**. A different covariance, including an in-domain one, does not separate Beethoven from Mozart at these blocks.
- **The bleed branch.** The branch "the dose predicts the bleed" is not reached, since the AUC is inconclusive. What the doses do show, read directly from M1:
  - at L8 and L12, a rank-1 key at the subject's last token sends 49–91% of the edit to same-type neighbours, for every fact. At L4 the share ranges from 3% (the Louvre, for the Eiffel edit) to 88%;
  - so whether a neighbour bleeds depends mostly on whether the edit is strong enough to flip anything, not on the key.

  This is an observation, not a pre-registered reading.
- **The efficacy classes V and S cannot be assessed**, because the probe is invalid. The one thing M1 settles is that efficacy did not fail through K.
- **No reading is suspended by M3.**

## Prediction against outcome

| Prediction (written before the run) | Outcome |
|---|---|
| Probe valid | **Wrong**: invalid, because of Bolt L12 |
| L4 and Eiffel mostly class V, not K | Not K (right); V cannot be assessed |
| Dose predicts bleed, AUC ≥ 0.8 | **Wrong**: 0.69, inconclusive |
| Same-type neighbours 0.3–0.8, unrelated below 0.2 | Neighbours higher than predicted (median 0.71, up to 0.91); unrelated median 0.23 |
| W neutral or slightly better; I worse than C | W neutral (right); I neutral (**wrong**) |

## What this changes for the write-up

The negative can name a cause that is now measured: **at blocks 8 and 12 of Gemma 4 E2B NF4, where the edits flip, the subject key of a rank-1 ROME edit is shared to about 50–90% with same-type entities, whatever the covariance among the three tested.**
- It is consistent with the C6 failure of every edit that flips.
- It does not, by itself, explain the efficacy failures. Those are not a key mismatch, and their value-side cause could not be measured with this probe.
- It is a statement about these blocks, this instrument and these five facts.

## Inputs and integrity

- **The W covariance.** It comes from WikiText-2 train: 629 articles, 2 485 830 tokens, 200 windows of 512, seed 20261026. It was computed with Codex's unchanged `collect_model_covariances`, whose file hash is pinned in the plan.
  - The windows mention Toyota and Honda once each, Spain once, Italy 4 times, Canada 7 times, Japan 23 times and Egypt 28 times.
  - The service took 948 s, with a memory peak of 10.1 G.
- **Limits.** Every service ran under the shared lock with 22G/24G/8G and 1800 s; the peak was 15.2 G.
- **Files.**
  - The outputs are in `~/Projects/larql-research/phase-b-d1-20261027/`: the plan, the records, the keys, the covariance, `analysis.json` and the logs.
  - The versioned analysis is `docs/research-data/phase-b-d1-analysis-claude.json`.
  - The archive is `phase-b-d1-20261027.zip`, with a SHA-256 manifest and a restore check.
