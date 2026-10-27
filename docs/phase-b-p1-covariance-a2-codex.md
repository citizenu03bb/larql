# P1 covariance A2 — approved, frozen before A2 execution

Pat explicitly approved **“Approuver A2 : 6 × 200”** in this chat on
2026-10-27, after reviewing the concrete proposal below. A2 waits for the
current A1 attempt to finish under its original bounds; its failed status
is a mandatory precondition, not an automatic retry permission.

The approved A1 requires three covariance services of 400 forwards, each
with a 1800-second systemd maximum and an internal 1740-second deadline.
On service 0, the completed-window receipts show 15 forwards at 96.677 s
and 27 at 169.972 s: about 6.108 s per window between these observations.
That steady interval implies about 2443 s for 400 windows, beyond the
fixed service limit. This is an execution feasibility issue, not a model
editing result.

**Approved amendment:** if A1 service 0 fails its existing limit,
close that attempt without retrying it. Start a separately recorded A2
with **six services of 200 windows**, each retaining the original
22G/24G/8G, 1800-second bounds and shared flock. At the observed rate a
200-window service takes about 1222 seconds plus loading overhead.

The exact existing 1200×1024 sample, window order, all-position capture,
model, layer, float64 second moment, total 1,228,800 keys, and single final
ridge remain unchanged. Only the service partition changes. No resampling,
new text or new target is introduced. A2 still costs exactly 1200 successful
covariance forwards. The failed A1 attempt is extra and remains explicitly
counted; the combined upper bound is **1600 sequence-forwards** (A1 at most
400 plus A2 1200). Its exact actual use comes from A1's final worker receipt.

The current A1 service is allowed to complete or stop under its existing
deadline, so its exact forward counter can be preserved. No live source,
limit or parameter is modified, and there is no silent interruption with
an ambiguous counter. Its partial directory and logs remain intact. A2
uses new part directories and a new committed worker/preflight/rehearsal.
No A2 model service starts without explicit approval. Any A2 service failure
stops A2, without automatic retry. The final `covariance/C17.npy` and
`covariance/status.json` are published only after all six valid parts.

The independent solver port, its tests and preparation can continue while
A1 runs. Neither the five solve requests nor their separately approved
500-sequence/25-update caps change. No push or blind-set access is proposed.
