# Galadriel compiled-surface C0 result

**Run date:** 2026-08-28

**Frozen source:** `ed08f5b2aad58e3f2d162e927c5ad832ccce8a63`

**Outcome:** failed (`1/7` positives, `21/21` controls)

**Claim boundary:** product-only result on spent Galadriel development
material

## Verdict

C0 is closed as a clean product failure. The exact surface matcher and reject
path worked, but the terminal residual adapter is not an autonomous effecter.
Removing the conditionally visible L31/F1625 FFN row caused six of seven
supported positive surfaces to miss the first target ID.

No threshold, surface, target, adapter, or output was changed, and the run was
not repeated.

## Positive results

Every positive matched its frozen suffix and returned `fire`, so this is not a
matcher failure. Only `Lothlorien's queen is` emitted the exact target IDs
`[8833, 553, 21457]`, triggered L31/F1625, and released. The six misses emitted
these first IDs instead of `8833`:

| Prompt | First generated ID |
|---|---:|
| `The monarch of Lothlorien is` | 496 |
| `Lothlorien's monarch is` | 2876 |
| `Lothlorien is governed by` | 506 |
| `The queen of Lothlorien is` | 2876 |
| `Lothlorien is led by` | 2876 |
| `The leader governing Lothlorien is` | 6065 |

Because the first target ID was not selected, explicit `ActiveSpan`
authorization correctly remained inert on every miss.

## Control results

All twenty-one controls returned `compiled_reject`, never triggered a span,
and produced complete three-token generations bit-identical to a separate
source-Q4K path. This validates the important safety half of C0: an unsupported
surface has no residual adapter, FFN overlay, or continuation effect.

It does not estimate a population false-positive rate because all material is
spent and enumerated.

## Architectural consequence

The prior version-2 description said that the terminal adapter strengthened
the first target token. C0 shows that “strengthens” was load-bearing: the
adapter complemented the conditionally exposed FFN payload row; it did not
replace it.

The next product candidate, if implemented, is C1:

```text
exact compiled surface fire
    -> expose the frozen FFN effect row
    -> apply the frozen terminal residual delta
    -> natural first target ID
    -> explicitly authorized ActiveSpan
```

The FFN row is an effect component, not an eligibility signal. Rejection still
hides it and applies no delta. C1 is a new architecture and protocol, not a C0
retry. It must verify that the compiled manifest, linked patch slot, target
IDs, adapter vector, and base checksum agree exactly before generation.

## Evidence identities and resources

- preregistration SHA-256:
  `b02012b4315682b4bc82718f0d1604cf036030870afff88585f674a4ae8d2026`
- pre-run manifest SHA-256:
  `3771dd3010c1f63d01d97cf9fb9bf341bab24b57a37fba10de384823f2f9cd0d`
- compiled surface manifest SHA-256:
  `0aa1c4f29004f6efe8273c6566333c55d0b5c82e24488de3a6ec2006f2329cb3`
- release binary SHA-256:
  `a19c5355f17397b292ff124108b09671eacc0da6820fd7b7530d3ab2a1e7a301`
- result evidence SHA-256:
  `97220a7ebbc415f10835bffac67653a9feb10dd832c3ee918265edc3140f7ad1`
- systemd unit:
  `larql-product-surface-v1-20260828-ed08f5b2.service`
- invocation ID:
  `46892813c12b4374a32b5a52119b40f9`
- wall time: 30 minutes 21.043 seconds
- CPU time: 3 hours 47 minutes 28.144 seconds
- memory peak: 22G
- swap peak: 2.9G

