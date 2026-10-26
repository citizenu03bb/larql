# B1 covariance complete

Pat approved B1/B2 with "go for B1 and B2". The frozen v1 method and
protocol were used without alteration. Execution wrapper: `2f5bc5f0`.

One real NF4 E2B service completed **200/200 forwards, 0 optimizer updates**.
The registered 200×512 token windows produced 12,800 keys at each of L4,
L8 and L12. All three float32 covariances passed float64 Cholesky validation.
No sample, ridge, layer, seed or hyperparameter was changed.

The service held the shared flock and enforced 22G/24G/8G and 1800 seconds.
systemd reported success, runtime **14min 23.499s**, memory peak **12.2G**,
swap **8.1M**. The model worker exited; only the pre-existing lightweight
router process remained on the GPU. The fresh synthetic rehearsal used
the same stage wrapper and exact 200-forward count.

The checkpoint's full SHA-256 is
`2db5482b20d746879bb3ef79b5203e9075a2e2b98f54ec7c2f281c1477ddc550`.
Matrix artifact SHA-256 values:

| Layer | SHA-256 |
|---|---|
| L4 | `e8cc456e4b245184dfb53bef91b067f53949947a6821bf5594e4470bd8e02d29` |
| L8 | `4e9d52c3aa4fc36728094fafa3bdef837f874fa77daeefefa3381c9c67f9398b` |
| L12 | `73c7a2f9bc2c9f62b9bc4c89ae2703e44c16565a0df275341105e7e7f749147a` |

The matrix hashes were rechecked after the service. Arrays and full logs
remain in `larql-research/phase-b-rome-20261026/b1/`, with monitor/journal
files in its parent. The [compact receipt](research-data/phase-b1-covariance-results.json)
records the model fingerprint, limits, counters, source and array hashes.

B1 establishes a fixed preconditioner for B2, not an edited fact or a
certification. B2 still requires the registered solves and Claude Code's
independent certifications before any layer selection. The blind set
remains unopened. No push was made.
