# Lazy Acts Implementation Plan

Branch: `lazy-acts`
Worktree: `/tmp/probelab-lazy-acts`

## Objective

Implement a lazy, probe-focused activations system (`Acts`) optimized for high throughput and low memory on large datasets, while keeping API feel close to current `probelab` usage.

## Design Constraints

- Keep user-facing flow familiar: `collect -> select layer -> pool -> fit/predict`.
- Use tinygrad-inspired style for operations: chainable methods, explicit `dim` args, lazy until `realize()`/`save()`/`cache()`.
- Keep v1 scope narrow: no generic lazy tensor algebra.
- Prioritize out-of-core workloads (disk cache + streaming batches/layers).

## Tinygrad-Inspired API Shape

### Core object

```python
acts = pl.collect(model, tokens, layers=range(16))    # lazy source plan
acts = acts.select_layers(12).mean_pool(dim="s")      # lazy transform chain
x = acts.realize()                                     # executes plan
```

### Methods

- `select_layers(layer_or_layers)`
- `slice_batch(slice_or_indices)`
- `mean_pool(dim="s")`
- `max_pool(dim="s")`
- `last_pool(dim="s")`
- `sum_pool(dim="s")`
- `cast(dtype)`
- `to(device)`
- `cache(path=None)`
- `save(path)`
- `realize()`
- `iter_batches(batch_size)`
- `iter_layers()`

### Compatibility aliases (keep familiar feel)

- `collect_activations(...) -> collect(...)` alias
- `load_activations(...) -> load(...)` alias (if added)
- `Acts.mean_pool()` defaults to sequence axis as today
- Optional bridge alias: `select(layer=...)` mapped to `select_layers(...)`

## Implementation Phases

## Phase 0: Baseline and guardrails

Goal: lock current behavior and establish comparison harness.

Changes:

- Add baseline benchmark scripts in `perf_checks/`:
  - `perf_checks/layer_sweep_baseline.py`
  - `perf_checks/probe_sweep_baseline.py`
  - `perf_checks/token_level_baseline.py`
- Record baseline runtime + peak memory + IO volume in markdown notes.

Tests:

- Ensure existing tests are green before migration (`tests/`).

Exit criteria:

- Baseline metrics logged and reproducible.

## Phase 1: Introduce `Acts` data model and op IR

Goal: create lazy container + normalized operation representation.

New files:

- `probelab/processing/acts.py`
- `probelab/processing/ops.py`
- `probelab/processing/sources.py`

Core pieces:

- `Acts` class: source descriptor + immutable op list.
- `dims` contract restricted to `bh`, `bsh`, `blh`, `blsh`.
- `seq_mask` validation when `s` axis present.
- tinygrad-style method chaining returns new `Acts` object.

Tests:

- `tests/processing/test_acts_construction.py`
- `tests/processing/test_acts_api_dims.py`

Exit criteria:

- `Acts(...)` creation and transform chaining work without execution.

## Phase 2: Planner + pushdown optimizer

Goal: compile op list to executable plan with deterministic pushdown rules.

New files:

- `probelab/processing/planner.py`
- `probelab/processing/plan_types.py`

Planner responsibilities:

- Normalize op sequence.
- Push down `select_layers`, `slice_batch`, `*_pool(dim="s")` where legal.
- Validate unsupported plans early with clear errors.
- Preserve stable output semantics.

Tests:

- `tests/processing/test_planner_pushdown.py`
- `tests/processing/test_planner_failures.py`

Exit criteria:

- Pushdown behavior is deterministic and covered by tests.

## Phase 3: Executors (tensor source first, then collect/disk)

Goal: execute plans in streaming mode.

New files:

- `probelab/processing/executors/tensor_exec.py`
- `probelab/processing/executors/model_exec.py`
- `probelab/processing/executors/disk_exec.py`
- `probelab/processing/executors/common.py`

Execution behavior:

- `realize()` materializes final tensor.
- `iter_batches()` streams batch shards.
- `iter_layers()` streams one layer view at a time.
- No full `blsh` residency for layer sweep paths.

Tests:

- `tests/processing/test_exec_tensor_source.py`
- `tests/processing/test_exec_model_source.py`
- `tests/processing/test_exec_disk_source.py`

Exit criteria:

- End-to-end lazy execution works for model/tensor/disk sources.

## Phase 4: Disk format + cache/load

Goal: robust out-of-core storage for repeated sweeps.

Changes:

- Add storage module:
  - `probelab/processing/storage.py`
- Implement `Acts.save(path)`, `Acts.cache(path=None)`, `pl.load(path)`.
- Use chunked format (Zarr preferred) with attrs: `dims`, `dtype`, version, optional layer ids.

Tests:

- `tests/processing/test_storage_roundtrip.py`
- `tests/processing/test_storage_partial_read.py`

Exit criteria:

- Save/load round-trip preserves numeric parity and metadata.

## Phase 5: Public API wiring + compatibility bridge

Goal: expose new API while minimizing churn in user workflows.

Changes:

- Update exports:
  - `probelab/processing/__init__.py`
  - `probelab/__init__.py`
- Add:
  - `collect(...)`
  - `load(...)`
  - `Acts`
- Keep compatibility adapters:
  - `collect_activations(...)` calling `collect(...).realize()` or returning an `Activations` shim during transition.

Compatibility strategy:

- Keep old API entrypoints for one release window but implement via `Acts` backend.
- Mark old names as deprecated in docs/tests after parity is proven.

Tests:

- `tests/test_public_api_compat.py`

Exit criteria:

- Existing examples run with minimal edits.

## Phase 6: Probe migration to tensor/Acts-native interfaces

Goal: make probes consume `Acts` and stream efficiently.

Changes:

- `probelab/probes/base.py`:
  - support `fit(x, y, mask=None)` and `predict(x, mask=None)` where `x` can be `Acts` or `torch.Tensor`.
- Migrate probes:
  - `probelab/probes/logistic.py`
  - `probelab/probes/mlp.py`
  - `probelab/probes/attention.py`
  - `probelab/probes/multimax.py`
  - `probelab/probes/gated_bipolar.py`

Performance-specific behavior:

- Logistic/MLP token-level training uses batch streaming + mask flattening per batch.
- Layer sweeps consume `iter_layers()` without full materialization.

Tests:

- `tests/test_probes.py` migrated to `Acts` fixtures.
- Add memory-aware tests in `tests/probes/test_probe_streaming_paths.py`.

Exit criteria:

- Probe AUROC parity within tolerance vs eager baseline.

## Phase 7: Example/docs migration

Goal: make new flow the default user path.

Changes:

- Update README and examples:
  - `README.md`
  - `examples/01_basic_training.py`
  - `examples/04_layer_sweep.py`
  - `examples/06_token_level.py`
  - `examples/07_streaming.py`
- Add migration guide:
  - `docs/migration_acts.md`

Key messaging:

- Preferred API: `collect(...).mean_pool().cache(...)` + `iter_layers()`.
- Keep old examples available briefly with deprecation note.

Exit criteria:

- Docs and examples consistently use `Acts` API.

## Phase 8: Perf acceptance + cleanup

Goal: verify goals and remove legacy complexity.

Changes:

- Final perf run on baseline scripts.
- Remove/deprecate unused eager-only code paths in:
  - `probelab/processing/activations.py`
  - associated eager-only tests.

Acceptance targets:

- Peak memory reduction 5x+ on multi-layer sweeps.
- Sweep runtime reduction 2x+ when reusing cached pooled activations.
- Numerical parity with baseline within floating-point tolerance.

Exit criteria:

- Performance targets met and legacy fallback either removed or isolated behind deprecation.

## File-by-File Implementation Order

1. `probelab/processing/ops.py`
2. `probelab/processing/sources.py`
3. `probelab/processing/acts.py`
4. `probelab/processing/planner.py`
5. `probelab/processing/executors/common.py`
6. `probelab/processing/executors/tensor_exec.py`
7. `probelab/processing/executors/model_exec.py`
8. `probelab/processing/storage.py`
9. `probelab/processing/executors/disk_exec.py`
10. `probelab/processing/__init__.py`
11. `probelab/__init__.py`
12. `probelab/probes/base.py`
13. `probelab/probes/logistic.py`
14. Remaining probes and docs/examples

## Risks and Mitigations

- Planner complexity creep:
  - Mitigation: fixed op set and explicit non-goals; reject unsupported graphs.
- Disk format lock-in:
  - Mitigation: versioned metadata and adapter layer in `storage.py`.
- Probe migration regressions:
  - Mitigation: parity tests on AUROC and output shapes across sequence/token paths.
- Silent performance regressions:
  - Mitigation: CI perf smoke checks for key sweep workloads.

## Suggested Initial PR Stack

1. PR 1: `Acts` model + ops + planner skeleton + basic tests.
2. PR 2: model/tensor executors + `realize`/`iter_batches`.
3. PR 3: storage + `cache`/`load` + partial-read tests.
4. PR 4: API exports + compatibility bridge.
5. PR 5: Logistic + MLP migration.
6. PR 6: remaining probes + examples + docs + perf validation.

