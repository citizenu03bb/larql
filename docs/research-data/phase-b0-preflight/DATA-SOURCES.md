# B0 covariance inputs: sources, attribution and private regeneration

`covariance-tokens.npy` is withdrawn from the tracked tree after the
2026-10-27 licence review. The array is reversible dataset content, not
just numerical measurement output. The original `covariance-sample.json`
and `phase-b0-manifest.json` remain unchanged as historical evidence.

**This removal does not remove the blob from Git history or clear the
branch for publication.** The file remains reachable in 70f0cd6c until
Pat authorizes a separate resolution. Do not infer an Emotion redistribution
licence from the Banking77 notice below or from the repository code licence.

## Banking77 attribution

**BANKING77**, by Iñigo Casanueva, Tadas Temcinas, Daniela Gerz, Matthew
Henderson and Ivan Vulic (2020), *Efficient Intent Detection with Dual
Sentence Encoders*, Proceedings of the 2nd Workshop on NLP for ConvAI.
[Original dataset and citation](https://github.com/PolyAI-LDN/task-specific-datasets#banking),
[paper](https://arxiv.org/abs/2003.04807).

Banking77 is supplied under
[Creative Commons Attribution 4.0 International](https://creativecommons.org/licenses/by/4.0/),
as confirmed by the
[upstream licence](https://github.com/PolyAI-LDN/task-specific-datasets/blob/master/LICENSE)
and the exact cached
[dataset card](https://huggingface.co/datasets/legacy-datasets/banking77/blob/f54121560de48f2852f90be299010d1d6dc612ec/README.md).
No warranties are given by that licence, and no endorsement is implied.

The research transformation selected training texts, sorted their record
IDs, applied seed 20261026 permutations, tokenized with Gemma's tokenizer,
inserted EOS separators and selected **80 disjoint 512-token windows**.
This notice identifies that contribution; it does not license the entire
mixed-source array under CC BY 4.0.

## Emotion attribution and unresolved redistribution

The actual source is **dair-ai/emotion**, config `split`, dataset version
`0.0.0`, `train` (16000 rows), Hub snapshot
`cab853a1dbdf4c42c2b3ef2173804746df8825fe`.
The [card for that snapshot](https://huggingface.co/datasets/dair-ai/emotion/blob/cab853a1dbdf4c42c2b3ef2173804746df8825fe/README.md)
labels the licence `other` and provides a research/education-use notice.
The [origin repository](https://github.com/dair-ai/emotion_dataset#usage)
gives the same usage limitation. A general permission to redistribute
reconstructible text publicly has **not been established** in this review.

Research attribution: Elvis Saravia, Hsien-Chi Toby Liu, Yen-Hao Huang,
Junlin Wu and Yi-Shin Chen (2018), *CARER: Contextualized Affect
Representations for Emotion Recognition*, EMNLP, pp. 3687–3697.
[Primary paper](https://aclanthology.org/D18-1404/).
The private sample contains **80 disjoint 512-token windows** from this
training split. Citation is not treated as a redistribution grant.

## Reproduce privately without a model forward

Obtain the datasets through their public sources at the revisions above,
under their applicable terms. The verified strict path uses the cached
Arrow training files named in `covariance-sample.json`:

- `dair-ai___emotion/split/0.0.0/cab853a1dbdf4c42c2b3ef2173804746df8825fe/emotion-train.arrow`;
- `legacy-datasets___banking77/default/0.0.0/f54121560de48f2852f90be299010d1d6dc612ec/banking77-train.arrow`.

The public dataset IDs, configs and revisions can be passed to a local
Hugging Face datasets installation to prepare those caches. Acquiring
inputs is a separate step: the reproducer performs **no download**.
Different datasets/Arrow versions can serialize the same text differently;
the strict reproducer checks the recorded Arrow byte hashes and rejects
such a mismatch rather than changing the frozen source receipt.

From the repository root, with the local cached Gemma tokenizer and the
Python environment containing NumPy, PyArrow, torch and transformers:

```bash
mkdir -p "$HOME/Projects/larql-research/phase-b0-private-inputs-20261027"
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  /mnt/models/finetune/venv/bin/python tools/rome_solver/regenerate_b0_tokens.py \
  --datasets /mnt/models/distill-course/hf/datasets \
  --tokenizer-root /mnt/models/hf-models/gemma-4-E2B-it \
  --out "$HOME/Projects/larql-research/phase-b0-private-inputs-20261027/covariance-tokens.npy"
```

The output must not already exist and must be outside this worktree. If
the hash-listed ADRs have changed, add `--docs-revision 70f0cd6c` while
that original private history is available, or supply the exact historical
source revision. Only the 27 hash-listed ADRs are read. The RNG source
order is **docs, emotion, banking77**; the alphabetized JSON quota order
must not replace it.

The script verifies all source and tokenizer metadata hashes, all streams,
all **200 windows**, raw token bytes and complete NPY bytes before writing.
Expected NPY SHA-256:
`ef9ee0423bbc5534eca7d450f6cb29bb4eba0802f4421165b844f97ec3bf3cdd`.
It loads no pretrained weights and performs no inference. A private
regeneration was verified byte-for-byte in the
[reproduction receipt](../phase-b0-tokens-regeneration-codex.json).

The frozen `run_phase_b.py` still expects this file at its old worktree
path: B0 verification, real preflight and real B1 cannot run from a clean
checkout until a private input is supplied. They are not weakened to skip
the hash check. Restoring that input locally is not authorization to run
another model experiment. The existing `*.npy` ignore rule prevents an
ordinary git add from tracking a regenerated local input again.

See the [licence review](../../phase-b0-covariance-tokens-licence.md) for
source versions, exact hashes and the remaining history/publication issue.
