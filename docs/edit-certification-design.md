# Edit certification harness — design (Phase A)

*Status: design draft for review, 2026-10-26. No code, no model forward. Written by Claude Code; to be approved by the author before implementation.*

**Mandate line.** A method-agnostic certifier is the showcase piece that stays valid whatever Phase B finds, and it is the gate an agent must pass before installing a persistent edit.

## 1. What it does

Input: one **edit** (a backend) and one **fact spec**. Output: a verdict, `CERTIFIED` or `REFUSED`, with reasons, as `verdict.json` and `verdict.md`.

A certificate is narrow by construction: *this edit, this fact, this question set, this instrument, greedy decoding*. It is never a claim of generalisation across facts.

Non-goals for v1:
- no edit method (that is Phase B);
- no LARQL/Q4K live instrument (45 s per forward); LARQL artifacts enter only through replay (§7);
- no sampling-based statistics: greedy decoding, one trial per question, as in every series so far.

## 2. Interfaces

Three pieces, all plain Python. Only the NF4 instrument and the torch backends import torch, lazily, so the core and its tests run without a model.

### 2.1 Instrument

```python
class Instrument(Protocol):
    name: str                         # "nf4-e4b", "nf4-e2b", "fake"
    def chat_prompt(self, question: str) -> list[int]: ...
    def forward(self, ids: list[int]) -> Logits: ...        # last-position logits, full prefix, no cache
    def decode(self, ids: list[int]) -> str: ...
    def token_ids(self, surface: str) -> list[int]: ...
    eos_ids: frozenset[int]
    layers: Sequence[Module] | None   # hook point for activation backends
    def fingerprint(self) -> dict: ... # model files, versions, quantization, commit
```

- `Nf4Instrument` wraps `setup()` from `scripts/run_nf4_local_bench.py` (Codex's frozen NF4 bench: double-quant NF4, CPU PLE lookup, eager attention, deterministic settings). It counts forwards against the service cap (`Limit`).
- `FakeInstrument` is a tiny deterministic model with a planted fact table, for tests only. Its outputs carry `synthetic: true` and can never become evidence.

### 2.2 Edit backend

```python
class EditBackend(Protocol):
    kind: str                          # "activation_static", "activation_oneshot", "rank_one", "lora", "larql_slot"
    def apply(self, inst: Instrument) -> ContextManager: ...   # active edit inside the block, removed on exit
    def save(self, path: Path) -> dict: ...                    # artifact + sha256
    @classmethod
    def load(cls, path: Path) -> "EditBackend": ...
    def random_counterpart(self, seed: int) -> "EditBackend": ...  # same kind, same placement, same norm
    def form(self) -> dict: ...        # declared properties, checked model-free (control C0)
```

`form()` declares, for example: the layers it writes; whether it needs an answer-time trigger, a donor pass or knowledge of the native proposal; its norm; and its artifact hash.

Backends planned for v1:

| Kind | Source | Random counterpart |
|---|---|---|
| `activation_static` | Phase 3 hook: add V·g at a layer to every row whose token is in a trigger set, at every forward | Gaussian vector, same float16 norm, same trigger rule |
| `activation_oneshot` | the archived L21 injection at the native proposal, then release | same, one-shot |
| `rank_one` | Phase B: a bf16 (u, v) adapter added to one NF4 linear at run time, without requantising | random u, v with the same Frobenius norm of u·vᵀ |
| `lora` | a PEFT adapter directory | random A, B with the same ‖BA‖_F |

`larql_slot` is listed for the interface only. In v1 it is replay-only.

### 2.3 Fact spec (JSON)

```json
{
  "fact_id": "hydrogen-symbol-He",
  "subject": {"surface": ["hydrogen", "Hydrogen"], "token_ids_by_instrument": {"nf4-e4b": [13076, 50604, 70599, 154624]}},
  "relation": "chemical symbol",
  "old": {"accept": ["H"], "token_ids": [236814, 640]},
  "new": {"accept": ["He"], "token_ids": [2209, 1293]},
  "swap_markers": ["helium"],
  "questions": [
    {"id": "E1", "control": "efficacy", "order": "subject_after_relation", "text": "What is the chemical symbol for hydrogen?", "seen": false},
    {"id": "E2", "control": "efficacy", "order": "subject_before_relation", "text": "...", "seen": false},
    {"id": "A1", "control": "attribute", "order": "subject_after_relation", "text": "What is the atomic number of hydrogen?", "expect": {"accept": ["1", "one"], "reject": ["2", "two", "3", "three"]}}
  ],
  "donor": {"surface": "helium"}
}
```

- `seen: true` marks a phrasing used to build or calibrate the edit. Efficacy counts only `seen: false` questions; the backend's provenance must list the seen phrasings, and the harness refuses an efficacy question whose text appears there.
- Every question declares the **order** of subject and relation, because Phase 3b showed that this order decides what a mention carries.

## 3. Controls

Each control returns `PASS`, `FAIL` or `NOT_EVALUATED`, with its evidence (token ids, decoded text, margins, artifact hashes).

**Native qualification comes first.** Every question is generated natively, from an empty answer, to EOS. A question whose native answer is wrong, or truncated at the cap, is excluded and listed. A control with too few qualified questions is `NOT_EVALUATED`; it is never silently passed.

**Generation rule (all controls).** Free greedy generation from an empty chat answer, up to EOS or a cap (12 new tokens by default). A cap hit is `unreadable`, not a pass. Readings use token ids where possible, with decoded text for swap markers. Resumed-prefix margins are recorded only as **diagnostics**, labelled as such, and never decide efficacy.

| # | Control | What is measured | PASS rule (fixed before any run) | Mandatory |
|---|---|---|---|---|
| C0 | Persistent form | `form()` of the backend, model-free | no answer-time trigger, no donor pass, no use of the native proposal; one stored artifact, applied identically at every forward | yes |
| C1 | Efficacy, unseen phrasings | ≥ 2 unseen phrasings, at least one per subject order | every qualified question emits a `new` token, no `old` token, no swap marker, EOS | yes |
| C2 | Other attributes of the same entity | e.g. atomic number, atomic mass, in both orders | each answer still contains its native value and none of its reject values, no swap marker | yes |
| C3 | Mentions before and after the relation | C1 and C2 split by `order`, plus one question where the subject appears only in the answer (e.g. "What is water made of?") | C1 and C2 pass in **each** order separately; the answer-only question has no swap marker | yes |
| C4 | Random direction, same norm | `random_counterpart(seed)` on the C1 questions, seed fixed in the spec | (a) no C1 question emits `new`; (b) the diagnostic margin gain of the real edit at the native value boundary is > 2 × max(random gain, 0) | yes |
| C5 | Donor or ceiling | when a donor exists: full-subject substitution at the same boundaries | reported as a fraction of the ceiling. A phrasing whose ceiling does not flip is marked *unsuitable* and leaves C1 (H1 lesson: an unflippable ceiling empties the normalisation) | qualification only |
| C6 | Off-target questions | same relation on other subjects (lithium, oxygen), the donor's own facts (helium's symbol and number), one unrelated question | greedy answers token-identical to native. Hook backends: the first forward with no trigger token must be bit-identical to native. First-token KL is reported | yes |
| C7 | Persistence after reload | `save()`, then a **new service** (new process, fresh model load), `load()`, rerun C1 | token-identical to the in-session C1 answers; artifact hash unchanged | yes |
| C8 | Propagation (RippleEdits style) | e.g. "Which element has atomic number 1? Give its symbol." | reported in v1, not gating (see decision Q2) | no |

**Verdict rule.** `CERTIFIED` if and only if every mandatory control is `PASS`. Otherwise `REFUSED`, with one reason per control that is `FAIL` or `NOT_EVALUATED`, ordered C0 to C7. The scope line (fact, questions, instrument, decoding) is printed with every verdict.

## 4. Execution

1. **`plan`** (model-free): fact spec plus backend give the list of generations and the forward budget, split into services of at most **24 forwards**. The plan is written and hashed before any forward. A service never starts a generation it cannot finish within its cap.
2. **`run --service k`**: one service under the shared lock, through a committed wrapper:
   `flock -w 3600 ~/Projects/larql-research/exploratory-natural-fact-edit/.heavy-model.lock systemd-run --user --wait -p MemoryHigh=22G -p MemoryMax=24G -p MemorySwapMax=8G -p RuntimeMaxSec=1800 …`
   Native generations are cached per instrument fingerprint and reused across backends; they are never re-run under an edit.
3. **`verdict --run DIR`** (model-free): recomputes every control and the verdict from saved ids, logits and hashes. It refuses on any hash mismatch.
4. Archive: zip plus SHA-256 manifest in `~/Documents/Codex/2026-09-24/our/outputs/`, a report, a commit.

**Cost estimate for one full certification of H → He** on E4B NF4:
- about 12 questions × ≤ 12 tokens, native and edited: about 290 forwards;
- random control on C1: about 24;
- persistence: about 24;
- in total about 340 forwards, or **about 15 services**.

At about 0.3 s per forward and 7 s per model load, that is under 5 minutes of service time. The native half is paid once per instrument.

## 5. Package layout

```
tools/edit_cert/
  README.md                  # English, public
  edit_cert/
    spec.py                  # fact spec + validation
    controls.py              # C0..C8, pure functions over saved evidence
    verdict.py               # rule, JSON + markdown rendering
    plan.py                  # budget and sharding
    instruments/fake.py, nf4.py
    backends/activation.py, rank_one.py, lora.py
    replay/                  # adapters for archived runs (§7)
  facts/hydrogen-symbol-He.json
  tests/                     # pytest, model-free
  run_service.sh             # the flock + systemd-run wrapper
```

## 6. Tests without a model

- **Verdict logic:** each control on hand-built evidence for PASS, FAIL and NOT_EVALUATED, including a cap hit, a swap marker, a disqualified native question, and an empty order group.
- **Fake instrument with planted edits:**
  - a selective edit → `CERTIFIED`;
  - an edit that also moves the second attribute → `REFUSED(C2)`;
  - an edit that fires only when the subject precedes the relation → `REFUSED(C1, C3)`;
  - a same-norm random edit → `REFUSED(C1)`;
  - a backend that loses its effect on reload → `REFUSED(C7)`;
  - a one-shot backend → `REFUSED(C0)`.
- **Plan:** budget arithmetic; no service over 24 forwards; a seen phrasing used as efficacy is rejected.
- **Hooks:** the activation backends on a synthetic torch stack (as in the existing dry runs): exact application, rows outside the trigger set untouched, removal on exit.
- **Replay adapters:** each adapter on a tiny fixture copied from the real archive layout.

## 7. Exit criterion: replay of archived cases

The harness passes Phase A when its **replay** mode, which reads archived artifacts and runs no model, gives the established verdicts below. Replay feeds the same control functions as a live run. Controls without archived evidence come out `NOT_EVALUATED`, which is itself part of the expected verdict.

| Case | Archive | Expected verdict and decisive reasons |
|---|---|---|
| R1. One-shot L21 entity delta (NF4, row 5, at the native boundary) | `nf4-asymmetry-20261026`, `nf4-phase3b-rows-20261026` | `REFUSED`: C0 FAIL (one-shot, uses the native proposal); C3 FAIL (the same vector on the answer mention gives "three": +15.6); C1 NOT_EVALUATED (resumed boundary only; on NF4 the symbol margin rises +19.7 but H stays top-1) |
| R2. Static L21 patch, g ∈ {1 … 3} | `nf4-phase3-static-20261026` (calibration) | `REFUSED`: C2 FAIL at g* = 1.0 (number top-1 "three", margin −0.5); C1, C4, C6, C7 NOT_EVALUATED (stopped by rule) |
| R3. Random direction, same norm | `nf4-asymmetry-20261026` (random), `entity-first-attribute-swap-20261025` (Q4K, three seeds) | `REFUSED`: C1 FAIL (no flip; symbol gain +1.06 on NF4) |
| R4. Full helium donor taken as an "edit" | `nf4-asymmetry-20261026` (donor) | `REFUSED`: C0 FAIL (substitutes the subject token); C2 FAIL (number becomes "two"): the ceiling is an entity swap, not an edit |
| R5. Q4K L21 delta, gain 2.585 | `entity-first-attribute-swap-20261025` | `REFUSED`: C0 FAIL (one-shot); boundary diagnostics reproduced (symbol +16.5; number +0.76; mass 48th → 4th) |

Plus the synthetic positive from §6, so the harness is shown able to say `CERTIFIED`. **There is no real positive case yet; Phase B is meant to supply one.**

Replay must reproduce the archived numbers exactly (same ids, same margins to the float), or the case fails.

## 8. Attribution and verification

- Claude Code designs and writes the harness and its tests. The replay of R1 to R5 is checked by an **independent verifier written by Codex** from the archives and this document, without reading the harness code.
- No live certification run is part of Phase A's exit. A first live run, for example the static L21 backend through the full C0–C7 battery, gets its own committed protocol and Pat's go-ahead.

## 9. Decisions requested from Pat

- **Q1. Verdict states.** Recommended: keep the two verdicts, `CERTIFIED | REFUSED`, and split reasons into `FAIL` and `NOT_EVALUATED`. The alternative is a third state, `INCONCLUSIVE`.
- **Q2. Propagation (C8).** Recommended: reported, not gating, in v1. RippleEdits-style probes are where published methods fail most, and H → He is a poor test of it, because He already belongs to helium. Make it gating once B has facts with unowned targets.
- **Q3. Forward cap.** Recommended: keep 24 forwards per service and shard (about 15 services, under 5 minutes). The alternative is a higher cap for the NF4 bench only.
- **Q4. Exit criterion.** Recommended: model-free replay of R1–R5 plus the synthetic positive. A live run comes later, under its own protocol.
- **Q5. Where it lives.** `tools/edit_cert/` on `research/edit-certification`, extractable later as its own public repository.
