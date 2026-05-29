# Storage

`probelab.storage` persists `Activations` to disk. Two backends are available.

| Backend  | Path             | Best for                                   | Extra            |
|----------|------------------|--------------------------------------------|------------------|
| `hdf5`   | a `.h5` file     | any `dims`; portable single-file artifacts | `probelab[storage]` |
| `memmap` | a directory      | flat multilayer (`"blsh"`); zero-copy loads| (built in)       |

## Dispatcher (recommended)

`save` / `load` / `stream` pick a backend from `format` or the path
(`.h5`/`.hdf5` → hdf5, suffixless/directory → memmap):

```python
from probelab import storage

storage.save(acts, "acts.h5", dtype="bfloat16")     # -> hdf5
loaded = storage.load("acts.h5", device="cpu", cast="float32")

storage.save(multilayer_acts, "run/acts")           # -> memmap directory
acts8 = storage.load("run/acts", layers=8)          # zero-copy single layer
for chunk, idx in storage.stream("run/acts", layers=8, chunk_tokens=500_000):
    ...
```

Force a backend with `format="hdf5"` or `format="memmap"`.

## Backend functions

The explicit functions are also exported:
`save_hdf5` / `load_hdf5` / `stream_hdf5` and
`save_memmap` / `load_memmap` / `stream_memmap` / `has_memmap`.

Both backends share the same load/stream keyword arguments: `layers` (int or
list; `int` returns single-layer `"bsh"`), `device`, and `cast`. (`memmap`
accepts a deprecated `layer=` alias.)

```python
storage.save_hdf5(acts, "a.h5", dtype="bfloat16", compression="gzip")
storage.load_memmap("run/acts", layers=[8, 12], device="cuda", cast="float32")
```

Metadata must be JSON-serializable. The probelab version is recorded in saved
artifacts so future readers can migrate formats.
