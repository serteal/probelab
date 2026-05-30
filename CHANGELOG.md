# Changelog

## 0.1.1 - 2026-05-30

Added:

- `Activations.to(device=, dtype=)` now casts dtype in addition to moving device.
- Feature probe `predict`/`predict_logits` accept `batch_size=` and run inference
  in chunks to bound peak memory.
- `storage.stream_hdf5` accepts `layers=` (parity with memmap and `load`).

Changed:

- Storage `stream` is unified across backends: HDF5 gains `layers=`, both default
  `chunk_tokens=500_000`, and `stream_memmap(layers=...)` accepts an int or list.
- Feature probes now raise `ValueError` on an empty training set instead of
  returning an uninitialized, unusable probe.
- `MLP.predict` sets eval mode so dropout never leaks into inference.

Fixed:

- `pool` max pooling over flat activations now chunks like mean pooling, avoiding
  a large transient on big token sets.
- `storage.has_memmap` tolerates a corrupt/foreign `meta.json` (returns `False`)
  so the `format="auto"` dispatcher can fall back to HDF5.

Removed / internal:

- `VmapEnsemble` is now internal (`probelab.utils._vmap_ensemble`); it is no
  longer exported from `probelab.utils`.
- Dropped dead code: `types.AggregationMethod`, `Logistic.loss_on_batch`,
  `MassMean` unused `optimizer_fn`/`scheduler_fn`, an unused `_check_val_loss`
  parameter, and the unused `[tool.ty]` config.

## 0.1.0 - 2026-05-27

Initial PyPI release.

Includes:

- Backend-agnostic `Activations` containers for dense and flat+offsets sequence activations.
- Dataset registry and built-in dataset loaders for probe training and evaluation.
- Tokenization helpers, chat-template metadata, and composable token masks.
- Feature and sequence probe implementations with estimator-style and `torch.nn.Module` APIs.
- Pooling, batching, metrics, and optional activation storage helpers.
- Optional mirin-backed activation collection adapter.
