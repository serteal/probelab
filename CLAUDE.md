# CLAUDE.md

This file provides guidance to Claude Code when working in this repository.

## Project

- **Name**: probelab
- **Version**: 0.1.0
- **Python**: >=3.11
- **Build system**: hatchling
- **License**: Apache-2.0

`probelab` is data/probe-first. The core package should be usable with
activations collected by any backend. Optional collection adapters live under
`probelab.collection`.

## Main API

```python
import probelab as pl

dataset = pl.datasets.load("circuit_breakers")

acts = pl.Activations.from_padded(
    hidden_states,                 # [batch, seq, hidden]
    attention_mask=attention_mask, # [batch, seq]
    dims="bsh",
    metadata={"model": "my-model", "site": "resid_post", "split": "train"},
)

prepared = acts.mean("s")
probe = pl.probes.Logistic().fit(prepared, dataset.labels)
scores = probe.predict(prepared)
auroc = pl.metrics.auroc(dataset.labels, scores)
```

For mirin-backed collection:

```python
from probelab.collection.mirin import collect_activations, stream_activations

tokens = pl.tokenize_dataset(dataset, tokenizer, mask=pl.masks.assistant())
acts = collect_activations(model, tokens, layers=[16])
```

Do not import collection adapters from the top-level package. `import probelab`
must not import `mirin` or `transformers`.

## Commands

```bash
uv sync --dev
uv run pytest tests/
uv run pytest tests/test_activations.py -v
uv run pytest tests/test_probes.py -v
uv run pytest tests/test_tokenization.py -v
uv run ruff check probelab/
uv run ruff format probelab/
```

## Package Structure

```text
probelab/
  __init__.py          # backend-agnostic public API
  activations.py       # Activations container and constructors
  tokenization.py      # Tokens and HuggingFace-tokenizer helpers
  chat_templates.py    # tokenizer template metadata
  masks.py             # token detection masks
  pool.py              # pooling/reduction functions
  metrics.py           # probe metrics
  datasets/            # dataset registry and dataset loaders
  probes/              # probe implementations
  collection/          # optional activation collection adapters
    types.py           # ActivationChunk
    mirin.py           # mirin adapter
  storage/             # explicit activation persistence helpers
  utils/               # validation and internal utilities
```

## Design Rules

1. Core code must not depend on `mirin` or `transformers` at import time.
2. Probes consume `Activations`; they should not know how activations were collected.
3. Collection adapters should return `Activations` or stream `ActivationChunk`.
4. Public activation constructors should be easy for external collectors:
   `from_tensor`, `from_padded`, and `from_flat`.
5. Keep persistence in `probelab.storage`, not on `Activations`.
6. Add tests for import behavior when changing dependencies or package exports.

## Activation Shapes

`Activations` supports four dimension strings:

- `bh`: `[batch, hidden]`
- `blh`: `[batch, layer, hidden]`
- `bsh`: flat token storage for logical `[batch, seq, hidden]`
- `blsh`: flat token storage for logical `[batch, layer, seq, hidden]`

For sequence layouts, the public fields are:

- `data`: flat token tensor
- `offsets`: cumulative per-sample token offsets
- `detection_mask`: flat boolean token mask
- `layers`: layer ids when `dims` contains `l`
- `metadata`: arbitrary provenance dict; JSON-compatible metadata is persisted
  by `probelab.storage`

## Common Workflows

Bring your own activations:

```python
acts = pl.Activations.from_flat(
    data=flat_hidden,
    offsets=offsets,
    detection_mask=detection_mask,
    dims="bsh",
)
probe = pl.probes.Logistic().fit(acts.mean("s"), labels)
```

Use probelab datasets and masks with an external collector:

```python
dataset = pl.datasets.load("wildguard_mix")
tokens = pl.tokenize_dataset(dataset, tokenizer, mask=pl.masks.assistant())
# collect however you want, then:
acts = pl.Activations.from_padded(hidden, attention_mask=attention, dims="bsh")
```

Stream mirin activations:

```python
for chunk in stream_activations(model, tokens, layers=[16]):
    pooled = pl.pool.mean(
        chunk.data[:, 0, :],
        chunk.detection_mask,
        offsets=chunk.offsets,
    )
```

## Adding Code

- New probes go in `probelab/probes/` and inherit `BaseProbe`.
- New datasets go in `probelab/datasets/` and register through the registry.
- New collection backends go in `probelab/collection/<backend>.py`.
- New chat-template rules go in `probelab/chat_templates.py`.
