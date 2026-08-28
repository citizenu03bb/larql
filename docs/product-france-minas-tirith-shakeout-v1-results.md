# France -> Minas Tirith product shakeout v1 result

**Status:** implementation-inconclusive before target projection

The unique v1 effect-compilation unit exited after loading the frozen source
payload and model weights but before exposing terminal hidden state. No target
logit, target margin, compiled direction, adapter, generation, control, or
surface decision was produced.

The exact error was:

```text
contextual forward did not expose terminal hidden
```

The cause is structural and independent of `Minas Tirith`: the contextual
capture API intentionally returned terminal hidden only when its detector list
was non-empty. V1 passed an empty list because the raw INSERT source contains no
Context operation. The source row was applied, but the requested diagnostic was
therefore absent by API contract.

Evidence:

- unit: `larql-minas-effect-v1-20260828-f6c97eac.service`;
- invocation: `2b7e2f7cfb224df8867b1034bad96920`;
- source commit: `f6c97eac8567a485de7857a7e3f09209df741123`;
- release binary SHA-256:
  `68f675e7ca0a717a8f280d54b1dcfeaa3bee1e2ab5815aa648df77f9e8168185`;
- failure evidence SHA-256:
  `9ec615a3ff312e2887b76afc03d57e1117ef05d493f699fc163ec6b94eb4a9f5`;
- runtime: `1min 20.198s`;
- CPU: `3min 26.512s`;
- memory peak: `22G`;
- swap peak: `4.7G`.

V1 produced no effect patch. It does not update the preregistered 65% success
prediction because it observed none of the event being predicted.

V2 changes one implementation factor: it persists and applies the already
specified compilation-only always-allow Context shim before capture. Target,
prompt, token IDs, L31/F1625 row, compiler, margins, resource bounds, controls,
and runtime exact-surface rule remain unchanged. V2 uses a new commit, binary,
protocol ID, unit, pre-run evidence, contextual source, effect, and result file.
