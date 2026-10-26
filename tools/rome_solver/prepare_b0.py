#!/usr/bin/env python3
"""B0 metadata/text preparation ONLY: no model constructor or model forward.

Reads safetensors headers, installed Python source, local tokenizer and local
English corpus. Refuses existing output. Tensor payload bytes read: zero.
"""
import argparse
import ast
import hashlib
import importlib.util
import json
from pathlib import Path
import struct

import numpy as np

from rome_solver.covariance import fixed_windows, local_records
from rome_solver.solver import CONTEXT_PREFIXES, prepare, request_from_spec


def sha(data):
    return hashlib.sha256(data).hexdigest()


def architecture(model_root):
    root = Path(model_root)
    with (root / "model.safetensors").open("rb") as f:
        size = struct.unpack("<Q", f.read(8))[0]
        if not 0 < size < 64 * 1024 * 1024:
            raise ValueError("Invalid safetensors header size")
        raw = f.read(size)
        if len(raw) != size:
            raise ValueError("Truncated header")
    header = json.loads(raw)
    transformer_root = Path(importlib.util.find_spec("transformers").origin).parent
    source_path = transformer_root / "models/gemma4/modeling_gemma4.py"
    source = source_path.read_bytes()
    tree = ast.parse(source)
    decoder = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "Gemma4TextDecoderLayer")
    forward = next(n for n in decoder.body if isinstance(n, ast.FunctionDef) and n.name == "forward")
    body = ast.unparse(forward)
    sequence = ["hidden_states = self.mlp(hidden_states)",
                "hidden_states = self.post_feedforward_layernorm(hidden_states)",
                "hidden_states = residual + hidden_states", "if self.hidden_size_per_layer_input:",
                "hidden_states *= self.layer_scalar"]
    offsets, cursor = [], 0
    for fragment in sequence:
        cursor = body.index(fragment, cursor) + len(fragment)
        offsets.append(cursor)
    witnesses, all_shapes = {}, {}
    for layer in range(35):
        prefix = f"model.language_model.layers.{layer}."
        expected = {"mlp.down_proj.weight": [1536, 6144 if layer < 15 else 12288],
                    "post_feedforward_layernorm.weight": [1536],
                    "layer_scalar": [1], "per_layer_projection.weight": [1536, 256]}
        for suffix, shape in expected.items():
            entry = header[prefix + suffix]
            if entry["shape"] != shape:
                raise ValueError(f"Unexpected header shape: {prefix + suffix}")
            if layer in (4, 8, 12):
                witnesses[prefix + suffix] = dict(shape=shape, dtype=entry["dtype"])
        all_shapes[str(layer)] = header[prefix + "mlp.down_proj.weight"]["shape"]
    config = json.loads((root / "config.json").read_text())["text_config"]
    if (config["hidden_size"], config["intermediate_size"], config["num_hidden_layers"]) != (1536, 6144, 35):
        raise ValueError("E2B configuration differs")
    return dict(model="Gemma-4-E2B-it", tensor_payload_bytes_read=0,
                safetensors_header_length=size, safetensors_header_sha256=sha(raw),
                metadata_sha256={name: sha((root / name).read_bytes()) for name in
                                 ("config.json", "tokenizer.json", "tokenizer_config.json", "chat_template.jinja")},
                modeling_source="transformers/models/gemma4/modeling_gemma4.py",
                modeling_source_sha256=sha(source), decoder_forward_line=forward.lineno,
                all_35_projection_shapes_verified=True, all_down_proj_shapes=all_shapes,
                candidate_header_entries=witnesses,
                pipeline="down_proj -> post_feedforward_layernorm -> FFN residual -> PLE residual -> layer_scalar",
                brief_discrepancy="layer_scalar is after the residual and PLE, not before them; raw down_proj site is unchanged")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--model-root", type=Path, default=Path("/mnt/models/hf-models/gemma-4-E2B-it"))
    parser.add_argument("--datasets", type=Path, default=Path("/mnt/models/distill-course/hf/datasets"))
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    proof = architecture(args.model_root)
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model_root, local_files_only=True)
    records, sources = local_records(args.repo, args.datasets)
    tokens, manifest = fixed_windows(records, tok)
    dev = []
    for p in sorted((args.repo / "tools/edit_cert/facts/phase-b-dev").glob("*.json")):
        spec = json.loads(p.read_text())
        request = request_from_spec(spec)
        examples = [prepare(tok, prefix + request.prompt, request.subject, request.target)
                    for prefix in CONTEXT_PREFIXES]
        essence = prepare(tok, request.essence_template.format(subject=request.subject), request.subject)
        dev.append(dict(fact_id=spec["fact_id"], spec_sha256=sha(p.read_bytes()),
                        subject=request.subject, prompt=request.prompt, target=request.target,
                        target_ids=[e.target_ids for e in examples],
                        input_lengths=[len(e.ids) for e in examples],
                        subject_positions=[e.position for e in examples], essence_position=essence.position,
                        last_subject_tokens=[tok.decode([e.ids[e.position]]) for e in examples]))
    if len(dev) != 5:
        raise ValueError("Expected the five public dev facts")
    manifest.update(sources=sources, tokenizer_metadata_sha256=proof["metadata_sha256"],
                    synthetic=False, model_forwards=0, prepared_before_covariance=True)
    args.out.mkdir(parents=True)
    np.save(args.out / "covariance-tokens.npy", tokens, allow_pickle=False)
    manifest["token_file_sha256"] = sha((args.out / "covariance-tokens.npy").read_bytes())
    for name, obj in (("architecture.json", proof), ("covariance-sample.json", manifest),
                      ("dev-tokenization.json", dict(model_forwards=0, contexts=list(CONTEXT_PREFIXES), facts=dev))):
        (args.out / name).write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n")
    print(json.dumps(dict(model_forwards=0, tensor_payload_bytes_read=0,
                          shape=list(tokens.shape), source_counts=manifest["quotas"],
                          token_file_sha256=manifest["token_file_sha256"])))


if __name__ == "__main__":
    main()
