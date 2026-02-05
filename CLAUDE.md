# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Information

- **Name**: probelab
- **Version**: 0.1.0
- **Author**: Alex Serrano
- **Python**: >=3.11
- **Build System**: hatchling (modern Python packaging)
- **License**: Apache-2.0

## Core Dependencies

- **PyTorch**: Deep learning framework (v2.7.1+)
- **Transformers**: HuggingFace models (v4.53.0+)
- **scikit-learn**: ML utilities (v1.7.0+)
- **jaxtyping**: Type annotations for arrays
- **einops**: Tensor operations
- **accelerate**: GPU/TPU optimization

## Main API Imports

```python
import probelab as pl

# Core types
from probelab import Label, Activations, Scores

# Activation collection
from probelab import collect_activations

# Probes
from probelab.probes import Logistic, MLP, Attention

# Datasets
from probelab.datasets import CircuitBreakersDataset, DolusChatDataset

# Metrics
from probelab.metrics import auroc, recall_at_fpr

# Masks for selective token processing
from probelab import masks
```

## Commands

### Development Setup

```bash
# Clone and install in development mode
git clone <repo_url>
cd probelab
uv sync --dev

# Verify installation
uv run python -c "import probelab; print(probelab.__version__)"
```

### Testing

```bash
# Run all tests
uv run pytest tests/

# Run tests with verbose output
uv run pytest tests/ -v

# Run specific test files
uv run pytest tests/test_activations.py -v
uv run pytest tests/test_probes.py -v
uv run pytest tests/test_scores.py -v
uv run pytest tests/test_masks.py -v
uv run pytest tests/test_metrics.py -v

# Run with coverage
uv run pytest tests/ --cov=probelab --cov-report=html --cov-report=term
```

### Code Quality

```bash
uv run ruff check probelab/
uv run ruff format probelab/
```

## Architecture Overview

probelab is a library for training classifiers (probes) on LLM activations. The design philosophy is **explicit over implicit** - operations are method calls on data objects, not hidden in pipeline abstractions.

### Core Concepts

1. **Activations**: Hidden states extracted from LLM layers, with axis-aware operations
2. **Scores**: Prediction outputs from probes, with aggregation methods
3. **Probes**: Simple classifiers that operate on Activations and return Scores
4. **Masks**: Control which tokens are used for training

### Module Structure

```
probelab/
├── __init__.py          # Public API exports
├── types.py             # Core type definitions (Label, Message, Dialogue)
├── datasets/            # Dataset handling
│   ├── base.py         # DialogueDataset base class
│   ├── deception.py    # Deception detection datasets
│   └── harmfulness.py  # Harmfulness detection datasets
├── models/              # Model interfaces
│   ├── architectures.py # Model-specific configurations
│   └── hooks.py        # PyTorch hook management
├── processing/          # Data processing
│   ├── activations.py  # Activations container (select, pool methods)
│   ├── scores.py       # Scores container (pool, ema, rolling methods)
│   └── tokenization.py # Dialogue tokenization
├── probes/              # Probe implementations
│   ├── base.py         # BaseProbe abstract class
│   ├── logistic.py     # Logistic regression probe
│   ├── mlp.py          # MLP probe
│   └── attention.py    # Attention-based probe
├── masks/               # Mask functions
│   ├── basic.py        # all, none, last_token
│   ├── role.py         # assistant, user, system
│   └── composite.py    # AndMask, OrMask, NotMask
├── utils/               # Utilities
│   ├── pooling.py      # masked_pool utility
│   └── validation.py   # check_activations, check_scores
├── metrics.py           # auroc, recall_at_fpr, etc.
└── logger.py           # Logging configuration
```

### Core API Pattern

The API follows an **Activations-centric** pattern - operations are methods on data objects:

```python
import probelab as pl

# 1. Collect activations
acts = pl.collect_activations(
    model=model,
    tokenizer=tokenizer,
    data=dataset,
    layers=[16],
    mask=pl.masks.assistant(),
    batch_size=32
)

# 2. Transform activations (method chaining)
prepared = acts.select(layer=16).pool("sequence", "mean")

# 3. Train probe
probe = pl.probes.Logistic(device="cuda").fit(prepared, labels)

# 4. Predict and get probabilities
scores = probe.predict(test_acts.select(layer=16).pool("sequence", "mean"))
probs = scores.scores  # [batch, 2] tensor
```

### Key Classes

**Activations** - Axis-aware container for hidden states:
```python
# Methods
acts.select(layer=16)           # Select single layer, removes LAYER axis
acts.select(layers=[8, 16])     # Select multiple layers, keeps LAYER axis
acts.pool("sequence", "mean")   # Pool over sequence dimension
acts.pool("layer", "mean")      # Pool over layer dimension
acts.to("cuda")                 # Move to device

# Properties
acts.shape                      # Tensor shape
acts.n_layers                   # Number of layers
acts.batch_size                 # Batch size
acts.seq_len                    # Sequence length
acts.d_model                    # Hidden dimension
```

**Scores** - Container for probe predictions:
```python
# Methods
scores.pool("sequence", "mean")  # Mean pooling over tokens
scores.ema(alpha=0.5)           # EMA pooling, then max
scores.rolling(window_size=10)  # Rolling window mean, then max
scores.to("cuda")               # Move to device

# Properties
scores.scores                   # Raw tensor [batch, 2] or [batch, seq, 2]
scores.shape                    # Tensor shape
scores.batch_size               # Batch size
```

**Probes** - Classifiers:
```python
# Logistic regression
probe = Logistic(C=1.0, device="cuda")
probe.fit(activations, labels)
scores = probe.predict(activations)  # Returns Scores object
probe.save("probe.pt")
probe = Logistic.load("probe.pt")

# MLP
probe = MLP(hidden_dim=64, dropout=0.1, device="cuda")

# Attention (handles sequences internally)
probe = Attention(hidden_dim=64, device="cuda")
```

### Common Workflows

**1. Basic Training**
```python
import probelab as pl

# Load data
dataset = pl.datasets.CircuitBreakersDataset()

# Collect activations
acts = pl.collect_activations(
    model=model,
    tokenizer=tokenizer,
    data=dataset,
    layers=[16],
    mask=pl.masks.assistant(),
    batch_size=32
)

# Prepare and train
prepared = acts.select(layer=16).pool("sequence", "mean")
probe = pl.probes.Logistic(device="cuda").fit(prepared, dataset.labels)

# Evaluate
test_acts = pl.collect_activations(...)
test_prepared = test_acts.select(layer=16).pool("sequence", "mean")
scores = probe.predict(test_prepared)
probs = scores.scores[:, 1].cpu().numpy()
print(f"AUROC: {pl.metrics.auroc(test_labels, probs):.3f}")
```

**2. Multi-Layer Analysis**
```python
# Collect from multiple layers
acts = pl.collect_activations(..., layers=[8, 12, 16, 20])

# Train probe on each layer
results = {}
for layer in [8, 12, 16, 20]:
    prepared = acts.select(layer=layer).pool("sequence", "mean")
    probe = pl.probes.Logistic().fit(prepared, labels)
    scores = probe.predict(test_prepared)
    results[layer] = pl.metrics.auroc(test_labels, scores.scores[:, 1])
```

**3. Token-Level with Score Aggregation**
```python
# Train on tokens
prepared = acts.select(layer=16)  # Keep SEQ axis
probe = pl.probes.Logistic().fit(prepared, labels)

# Predict and aggregate
scores = probe.predict(test_acts.select(layer=16))
aggregated = scores.pool("sequence", "mean")  # or .ema() or .rolling()
probs = aggregated.scores
```

**4. Using Masks**
```python
# Only detect on last assistant message
mask = pl.masks.AndMask(
    pl.masks.assistant(),
    pl.masks.nth_message(-1)
)

acts = pl.collect_activations(..., mask=mask)
```

### Adding New Probes

1. Create class inheriting from `BaseProbe` in `probes/`
2. Implement: `fit(X: Activations, y) -> self`, `predict(X: Activations) -> Scores`, `save()`, `load()`
3. Add tests in `tests/test_probes.py`

### Best Practices

1. **Memory**: Use streaming for large datasets, appropriate batch sizes
2. **Reproducibility**: Save probes with `probe.save()`, log hyperparameters
3. **Testing**: Test on both CPU and GPU, include edge cases

### Recent Changes (Breaking)

**Removed:**
- `Pipeline` class - Use direct method chaining on Activations/Scores
- `transforms/` module - Use `Activations.select()`, `Activations.pool()`, `Scores.pool()`, `Scores.ema()`, `Scores.rolling()`
- `coordination/` module - For multi-probe training, just use loops

**Migration:**
```python
# OLD (Pipeline-based)
from probelab import Pipeline
from probelab.transforms import pre, post

pipeline = Pipeline([
    ("select", pre.SelectLayer(16)),
    ("pool", pre.Pool(dim="sequence", method="mean")),
    ("probe", Logistic()),
])
pipeline.fit(acts, labels)
probs = pipeline.predict(test_acts)

# NEW (Direct API)
prepared = acts.select(layer=16).pool("sequence", "mean")
probe = Logistic().fit(prepared, labels)
test_prepared = test_acts.select(layer=16).pool("sequence", "mean")
scores = probe.predict(test_prepared)
probs = scores.scores
```
