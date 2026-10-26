# Phase B interface request to the harness author

For Claude Code, before B2 certification. This is a request, not an
implemented harness change; `tools/edit_cert` remains untouched here.

Pat approved up to 400 NF4 forwards per Phase B service. At base commit
`2fdc844f`, `plan.build(..., cap=400)` and `pack(..., cap=400)` support that
packing, but public `runner.prepare` and the prepare CLI still use the
default 24 and expose no cap parameter. Please expose an explicit cap
through prepare/CLI and record it in the hashed plan. Preserve 24 as the
default outside the approved Phase B NF4 scope; reject unsupported limits.

The existing separation of native/edit/reload phases makes a full dev
fact cost four services at cap 400: 312 + 400 + 24 + 96 = 832 forwards.
The B0 protocol and cost sheet use this actual packing. No cap escalation
or manual plan-JSON rewriting should be needed at execution time.

No method-side interface is missing for the raw output hook or export:
the existing instrument exposes `model`, `layers`, `tok` and its K/V
deadline; the existing backend accepts the required projection path and
float32 U0/V0 arrays. The CPU synthetic round trip verifies that contract.

Native cache, multi-fact certification and any evidence additions remain
with the harness author under the decision brief's division of work.
