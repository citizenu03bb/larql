"""Model-free reference and forward ledger; never imports an inference engine."""
import hashlib
import json
from pathlib import Path

import numpy as np

ARMS = (
    "matched", "cross_construction", "wrong_entity", "entity_absent", "self",
    "random_matched_plus", "random_matched_minus",
    "random_cross_plus", "random_cross_minus",
)
FACTS = [
    ("hydrogen", "H", "1"), ("helium", "He", "2"),
    ("lithium", "Li", "3"), ("beryllium", "Be", "4"),
    ("boron", "B", "5"), ("carbon", "C", "6"),
    ("nitrogen", "N", "7"), ("oxygen", "O", "8"),
]
TEMPLATES = [
    "The {relation} of {entity} is:\n",
    "{entity} has {relation}:\n",
    "For {entity}, the {relation} is:\n",
    "Regarding {entity}, give its {relation}:\n",
]
RELATIONS = ("chemical symbol", "atomic number")


def corpus():
    rows = []
    for split, entities, templates in (
        ("discovery", range(4), range(2)),
        ("confirmation", range(4, 8), range(2, 4)),
    ):
        for e in entities:
            for t in templates:
                for r in range(2):
                    name, symbol, number = FACTS[e]
                    rows.append(dict(id=f"{split}-e{e}-t{t}-r{r}", split=split,
                                     entity=e, pair=e // 2, template=t, relation=r,
                                     prompt=TEMPLATES[t].format(entity=name, relation=RELATIONS[r]),
                                     answer=(symbol, number)[r], alternative=(symbol, number)[1-r]))
        for t in templates:
            for r in range(2):
                rows.append(dict(id=f"{split}-absent-t{t}-r{r}", split=split,
                                 entity=None, pair=None, template=t, relation=r,
                                 prompt=TEMPLATES[t].format(entity="this element", relation=RELATIONS[r])))
    return rows


def ledger(rows):
    jobs = []
    for split in ("discovery", "confirmation"):
        stage = [r for r in rows if r["split"] == split]
        by_key = {(r["entity"], r["template"], r["relation"]): r["id"] for r in stage}
        for row in stage:
            for kind in ("baseline_capture", "hook_disabled_replay"):
                jobs.append(dict(split=split, recipient=row["id"], arm=kind))
        for row in stage:
            if row["entity"] is None:
                continue
            e, t, r = (row[k] for k in ("entity", "template", "relation"))
            donors = dict(
                matched=by_key[e, t, 1-r], cross_construction=by_key[e, t ^ 1, 1-r],
                wrong_entity=by_key[e ^ 1, t, 1-r], entity_absent=by_key[None, t, 1-r],
                self=row["id"],
            )
            for arm in ARMS:
                donor = donors.get(arm)
                if arm.startswith("random_matched"):
                    donor = donors["matched"]
                elif arm.startswith("random_cross"):
                    donor = donors["cross_construction"]
                jobs.append(dict(split=split, recipient=row["id"], donor=donor, arm=arm))
    return jobs


def fit_rank_one(rows, captures):
    """Fit only discovery entity-by-relation interactions, never logits."""
    if any(r["split"] != "discovery" for r in rows):
        raise ValueError("confirmation rows cannot enter fitting")
    core = [r for r in rows if r["entity"] is not None]
    keys = {(r["entity"], r["template"], r["relation"]): r["id"] for r in core}
    contrasts = []
    for pair in sorted({r["pair"] for r in core}):
        for template in sorted({r["template"] for r in core}):
            a, b = 2 * pair, 2 * pair + 1
            contrasts.append(captures[keys[a, template, 0]] - captures[keys[a, template, 1]]
                             - captures[keys[b, template, 0]] + captures[keys[b, template, 1]])
    x = np.asarray(contrasts, dtype=np.float64)
    if not np.isfinite(x).all():
        raise ValueError("nonfinite captures")
    _, s, vt = np.linalg.svd(x, full_matrices=False)
    if s[0] <= 1e-8 or (len(s) > 1 and s[0] - s[1] <= 1e-8 * s[0]):
        raise ValueError("zero or numerically ambiguous leading direction")
    u = vt[0]
    if u[np.argmax(np.abs(u))] < 0:
        u = -u
    return u


def project(recipient, donor, u):
    recipient, donor, u = (np.asarray(x, dtype=np.float64) for x in (recipient, donor, u))
    if recipient.ndim != 1 or recipient.shape != donor.shape or u.shape != recipient.shape:
        raise ValueError("shape mismatch")
    if not all(np.isfinite(x).all() for x in (recipient, donor, u)):
        raise ValueError("nonfinite vector")
    if abs(np.dot(u, u) - 1.0) > 1e-10:
        raise ValueError("nonunit basis")
    return recipient + u * np.dot(u, donor - recipient)


def random_control(recipient, donor, u, row_id, sign):
    """Deterministic orthogonal direction, same perturbation norm, both signs."""
    if sign not in (-1, 1):
        raise ValueError("invalid sign")
    v = np.array([int.from_bytes(hashlib.sha256(
        f"NBCP-v1|{row_id}|{i}".encode()).digest()[:8], "big") / 2**64 - 0.5
                  for i in range(len(u))], dtype=np.float64)
    v -= u * np.dot(u, v)
    norm = np.linalg.norm(v)
    if norm <= 1e-12:
        raise ValueError("degenerate random control; no reseeding")
    magnitude = np.linalg.norm(project(recipient, donor, u) - recipient)
    return recipient + sign * magnitude * v / norm


def row_passes(effects):
    """Effects contain delta_gap and delta_target for every fixed arm."""
    if set(effects) != set(ARMS):
        raise ValueError("incomplete or extra arms")
    if not all(np.isfinite(effects[a][m]) for a in ARMS for m in ("delta_gap", "delta_target")):
        raise ValueError("nonfinite metric")
    if any(abs(effects["self"][m]) > 1e-4 for m in ("delta_gap", "delta_target")):
        return False
    controls = [a for a in ARMS if a not in ("matched", "cross_construction", "self")]
    return all(
        min(effects[a][m] for a in ("matched", "cross_construction")) > 0.1
        and min(effects[a][m] for a in ("matched", "cross_construction"))
        - max(abs(effects[a][m]) for a in controls) > 0.001
        for m in ("delta_gap", "delta_target")
    )


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = corpus()
    jobs = ledger(rows)
    for name, value in (("corpus.json", rows), ("forward-ledger.json", jobs)):
        (args.output_dir / name).write_text(json.dumps(value, indent=2) + "\n")
    print(json.dumps({"prompts": len(rows), "max_forwards": len(jobs),
                      "per_stage": {s: sum(j["split"] == s for j in jobs)
                                    for s in ("discovery", "confirmation")},
                      "model_forwards_executed": 0}, indent=2))
