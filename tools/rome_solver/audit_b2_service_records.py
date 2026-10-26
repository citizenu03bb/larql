#!/usr/bin/env python3
"""Model-free ID/text, counter, fingerprint and index audit of peer records.

Loads only the pinned local tokenizer, never a model. Does not judge edits.
The full checkpoint was hashed by B1 preflight; peer fingerprints themselves
do not contain its hash, and this limitation is kept explicit in the output.
"""
import argparse
import hashlib
import json
from pathlib import Path


def load(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def require(test, why):
    if not test:
        raise ValueError(why)


def audit(root, preflight, method_root):
    from transformers import AutoTokenizer
    raw = (root / "index.json").read_bytes()
    index = json.loads(raw)
    model_path = Path(preflight["model_root"]) / "gemma-4-E2B-it"
    for name, expected in preflight["metadata_sha256"].items():
        require(sha(model_path / name) == expected, f"Live metadata hash mismatch: {name}")
    stat = (model_path / "model.safetensors").stat()
    require((stat.st_size, stat.st_mtime_ns) == (preflight["checkpoint_bytes"], preflight["checkpoint_mtime_ns"]),
            "Checkpoint stat differs from B1 full-hash preflight")
    tok = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    stops = {tok.eos_token_id, tok.convert_tokens_to_ids("<turn|>")}
    require(None not in stops, "Missing EOS token")
    cases = {}
    for case, entry in sorted(index["cases"].items()):
        if entry["state"] != "certified":
            cases[case] = {"state": entry["state"]}
            continue
        run = root / case / "run"
        spec = load(run / "spec.json")
        plan = load(run / "plan.json")
        verdict = load(run / "verdict.json")
        source = method_root / "solves" / case / "edit"
        original = load(source / "backend.json")
        require(original["content_sha256"] == plan["backend"]["content_sha256"], f"Plan uses another edit: {case}")
        for copy_name in ("edit-source", "edit-saved"):
            copy = run / copy_name
            for name in ("backend.json", "U0.npy", "V0.npy"):
                require(sha(copy / name) == sha(source / name), f"Copied/reloaded edit differs: {case} {copy_name}/{name}")
        algebra = load(root / case / "algebra-xcheck-claude.json")
        require(algebra["ok"] and entry["algebra_ok"] and algebra["solve"] == str(source.parent), f"Independent algebra receipt mismatch: {case}")
        require(sha(run / "verdict.json") == entry["verdict_sha256"], f"Verdict hash mismatch: {case}")
        require(sha(run / "evidence.json") == entry["evidence_sha256"], f"Evidence hash mismatch: {case}")
        statuses = {c["id"]: c["status"] for c in verdict["controls"]}
        require(statuses == entry["statuses"] and verdict["verdict"] == entry["verdict"], f"Index verdict mismatch: {case}")
        fingerprints, generations, tokens, boundaries, counters = [], 0, 0, 0, []
        for service, expected in zip(plan["services"], entry["records_sha256"]):
            p = run / "services" / str(service["index"])
            require(sha(p / "records.json") == expected, f"Record hash mismatch: {p}")
            record = load(p / "records.json")
            require(all(digest == original["content_sha256"] for digest in record["content_sha256"].values()), f"Service edit differs: {p}")
            status = load(p / "status.json")
            require(status == dict(status="complete", forwards=record["forwards"], records_sha256=expected), f"Status mismatch: {p}")
            fingerprint = record["fingerprint"]
            require(fingerprint["metadata_sha256"] == preflight["metadata_sha256"], f"Fingerprint metadata mismatch: {p}")
            require(fingerprint["transformers_modeling_source_sha256"] == preflight["modeling_source_sha256"], f"Modeling source mismatch: {p}")
            require(fingerprint["instrument"] == "nf4-e2b" and not fingerprint["synthetic"], f"Wrong instrument: {p}")
            require(fingerprint["quantization"]["bnb_4bit_quant_type"] == "nf4"
                    and fingerprint["quantization"]["bnb_4bit_use_double_quant"], f"Wrong quantization: {p}")
            local_tokens = 0
            for g in record["generations"]:
                ids = g["ids"]
                require(ids and all(type(i) is int and 0 <= i < len(tok) for i in ids), f"Invalid IDs: {p}")
                require(not any(i in stops for i in ids[:-1]), f"Tokens after EOS: {p}")
                require(tok.decode([i for i in ids if i not in stops]) == g["text"], f"Text/IDs mismatch: {p} {g['question_id']}")
                eos = ids[-1] in stops
                require(g["eos"] == eos and g["capped"] == (not eos and len(ids) == spec["max_new_tokens"]), f"EOS/cap mismatch: {p}")
                require(len(ids) <= spec["max_new_tokens"], f"Token cap exceeded: {p}")
                local_tokens += len(ids)
            require(record["forwards"] == local_tokens + len(record["boundaries"]), f"Raw forward recount mismatch: {p}")
            generations += len(record["generations"])
            tokens += local_tokens
            boundaries += len(record["boundaries"])
            counters.append(record["forwards"])
            fingerprints.append(hashlib.sha256(json.dumps(fingerprint, sort_keys=True).encode()).hexdigest())
        require(counters == entry["forwards"] and sum(counters) == entry["forwards_total"], f"Index count mismatch: {case}")
        require(len(set(fingerprints)) == 1, f"Fingerprint changed between services: {case}")
        cases[case] = dict(generations=generations, generated_tokens=tokens, diagnostic_forwards=boundaries,
                           forwards=sum(counters), fingerprint_sha256=fingerprints[0], index_consistent=True,
                           ids_text_eos_cap_consistent=True, primary_edit_and_reload_byte_identical=True,
                           primary_edit_content_sha256=original["content_sha256"],
                           independent_algebra_receipt_sha256=sha(root / case / "algebra-xcheck-claude.json"))
    return dict(author="Codex", model_forwards=0, full_certification=False, cases=cases,
                complete=index.get("complete", False), input_index_sha256=hashlib.sha256(raw).hexdigest(),
                checkpoint_stat_matches_B1=True, tokenizer_metadata_hashes_verified=True,
                limitation="Peer fingerprints contain metadata/source/quantization but no full checkpoint hash; B1 preflight full hash and unchanged live checkpoint stat are separate evidence.")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--preflight", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    report = audit(args.root, load(args.preflight), args.preflight.parent)
    report["script_sha256"] = sha(__file__)
    report["preflight_sha256"] = sha(args.preflight)
    with args.out.open("x") as f:
        f.write(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(dict(cases=len(report["cases"]), complete=report["complete"],
                          model_forwards=0, total_peer_forwards=sum(c.get("forwards", 0) for c in report["cases"].values()))))


if __name__ == "__main__":
    main()
