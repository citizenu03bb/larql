# Model-free audit of independent certification records

`audit_b2_service_records.py` checks the peer's index/verdict consistency,
record and evidence hashes, service completion receipts, and actual
forward counts. Every generated text is decoded again from its archived
IDs with the pinned local tokenizer. EOS position, cap flags and token
limits are checked. Forward counts are independently recounted as one
per generated token plus one per saved diagnostic boundary. No model is
loaded and no edit control is judged by this audit.

The first six-case receipt checks 204 generations, 2,435 generated tokens
and 66 diagnostic forwards, totaling 2,501 real certification forwards.
All checks pass. Its compact output is
[phase-b2-service-records-audit-first-codex.json](research-data/phase-b2-service-records-audit-first-codex.json).
This is an incremental audit, not the final B2 decision.

Each service's NF4/double-quantization, tokenizer/config hashes and modeling
source hash match B1. The live checkpoint size and nanosecond mtime also
match B1's full-hash preflight. A limitation is retained explicitly: the
peer service fingerprints themselves do not contain a full checkpoint
hash. Their metadata/source fingerprint and B1's separate full checkpoint
hash are different pieces of evidence; this audit does not invent a
per-service checkpoint hash.
