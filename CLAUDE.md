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

# Processing (tokenization + activation collection)
from probelab.processing import tokenize_dataset, collect_activations, Tokens

# Probes
from probelab.probes import Logistic, MLP, Attention

# Datasets
from probelab.datasets import CircuitBreakersDataset, BenignInstructionsDataset

# Metrics
from probelab.metrics import auroc, recall_at_fpr

# Masks for selective token processing
from probelab import masks

# Utilities
from probelab import Normalize
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

# Run specific test modules
uv run pytest tests/test_activations.py -v   # Activations container
uv run pytest tests/test_probes.py -v        # Probe implementations
uv run pytest tests/test_scores.py -v        # Scores container
uv run pytest tests/test_masks.py -v         # Mask functions
uv run pytest tests/datasets/ -v             # Dataset classes
uv run pytest tests/utils/ -v                # Utilities

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
├── __init__.py          # Exports: Activations, Scores, Label, Normalize, logger
├── types.py             # Core types (Label, Message, Dialogue, Role)
├── masks.py             # Mask functions (all, assistant, user, nth_message, etc.)
├── metrics.py           # auroc, recall_at_fpr
├── logger.py            # Logging config
│
├── processing/          # Tokenization + activation collection
│   ├── tokenization.py     # Tokens, tokenize_dialogues, tokenize_dataset
│   ├── activations.py      # Activations, collect_activations, stream_activations
│   ├── scores.py           # Scores (pool, ema, rolling)
│   └── chat_templates.py   # TEMPLATES dict for tokenizer config
│
├── models/              # Model interfaces
│   ├── architectures.py    # ARCHITECTURES dict for model structure
│   └── hooks.py            # HookedModel context manager
│
├── probes/              # Probe implementations
│   ├── base.py             # BaseProbe abstract class
│   ├── logistic.py         # Logistic regression
│   ├── mlp.py              # MLP
│   └── attention.py        # Attention-based
│
├── datasets/            # Dataset classes (17 files)
│   ├── base.py             # Dataset base class
│   └── ...                 # Domain-specific datasets
│
└── utils/               # Utilities
    ├── normalize.py        # Normalize class
    ├── pooling.py          # masked_pool utility
    └── validation.py       # check_activations, check_scores
```

### Core API Pattern

The API follows a **two-step** pattern: tokenize first, then collect activations:

```python
import probelab as pl

# 1. Tokenize with mask (determines which tokens to extract)
tokens = pl.processing.tokenize_dataset(dataset, tokenizer, mask=pl.masks.assistant())

# 2. Collect activations
acts = pl.processing.collect_activations(model, tokens, layers=[16])

# 3. Transform activations (method chaining)
prepared = acts.select(layer=16).pool("sequence", "mean")

# 4. Train probe
probe = pl.probes.Logistic(device="cuda").fit(prepared, labels)

# 5. Predict and get probabilities
scores = probe.predict(test_prepared)
probs = scores.scores  # [batch, 2] tensor
```

### Key Classes

**Tokens** - Tokenized inputs ready for activation collection:
```python
# Created via tokenize_dataset or tokenize_dialogues
tokens = pl.processing.tokenize_dataset(dataset, tokenizer, mask=pl.masks.assistant())

# Properties
tokens.input_ids        # [batch, seq] token IDs
tokens.attention_mask   # [batch, seq] attention mask
tokens.detection_mask   # [batch, seq] which tokens to extract (from mask)
tokens.padding_side     # "left" or "right"

# Methods
tokens.to("cuda")       # Move to device
tokens[10:20]           # Slice
len(tokens)             # Batch size
```

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
acts.has_axis(Axis.LAYER)       # Check if axis exists
acts.axis_size(Axis.BATCH)      # Size of axis
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

# Load and split data
train_ds, test_ds = pl.datasets.CircuitBreakersDataset().split(0.8)

# Tokenize
train_tokens = pl.processing.tokenize_dataset(train_ds, tokenizer, mask=pl.masks.assistant())
test_tokens = pl.processing.tokenize_dataset(test_ds, tokenizer, mask=pl.masks.assistant())

# Collect activations
train_acts = pl.processing.collect_activations(model, train_tokens, layers=[16])
test_acts = pl.processing.collect_activations(model, test_tokens, layers=[16])

# Prepare and train
train_prepared = train_acts.select(layer=16).pool("sequence", "mean")
test_prepared = test_acts.select(layer=16).pool("sequence", "mean")
probe = pl.probes.Logistic(device="cuda").fit(train_prepared, train_ds.labels)

# Evaluate
scores = probe.predict(test_prepared)
probs = scores.scores[:, 1].cpu().numpy()
print(f"AUROC: {pl.metrics.auroc([l.value for l in test_ds.labels], probs):.3f}")
```

**2. Multi-Layer Analysis**
```python
# Collect from multiple layers
tokens = pl.processing.tokenize_dataset(dataset, tokenizer, mask=pl.masks.assistant())
acts = pl.processing.collect_activations(model, tokens, layers=[8, 12, 16, 20])

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
# Train on tokens (keep SEQ axis)
prepared = acts.select(layer=16)
probe = pl.probes.Logistic().fit(prepared, labels)

# Predict and aggregate
scores = probe.predict(test_acts.select(layer=16))
aggregated = scores.pool("sequence", "mean")  # or .ema() or .rolling()
probs = aggregated.scores
```

**4. Using Masks**
```python
# Only detect on last assistant message (compose with & operator)
mask = pl.masks.assistant() & pl.masks.nth_message(-1)

tokens = pl.processing.tokenize_dataset(dataset, tokenizer, mask=mask)
acts = pl.processing.collect_activations(model, tokens, layers=[16])
```

**5. Streaming for Large Datasets**
```python
# Stream activations to avoid OOM
tokens = pl.processing.tokenize_dataset(large_dataset, tokenizer, mask=pl.masks.all())

for acts_batch, indices, seq_len in pl.processing.stream_activations(model, tokens, layers=[16]):
    # Process each batch incrementally
    prepared = acts_batch.select(layer=16).pool("sequence", "mean")
    # ... accumulate results
```

### Adding New Probes

1. Create class inheriting from `BaseProbe` in `probes/`
2. Implement: `fit(X: Activations, y) -> self`, `predict(X: Activations) -> Scores`, `save()`, `load()`
3. Add tests in `tests/test_probes.py`

### Adding New Model Architectures

Model and tokenizer configs are separate:

1. **New model structure** → Edit `models/architectures.py`, add to `ARCHITECTURES` dict
2. **New chat template** → Edit `processing/chat_templates.py`, add to `TEMPLATES` dict

```python
# models/architectures.py
ARCHITECTURES["mistral"] = Arch(
    get_layers=lambda m: list(m.model.layers),
    get_layer=lambda m, i: m.model.layers[i],
    get_layernorm=lambda m, i: m.model.layers[i].input_layernorm,
    set_layers=lambda m, layers: setattr(m.model, "layers", ...),
    num_layers=_num_layers_from_config,
)

# processing/chat_templates.py
TEMPLATES["mistral"] = {
    "prefix_pattern": re.compile(r"..."),
    "fold_system": False,
    "token_padding": (0, 0),
}
```

### Best Practices

1. **Memory**: Use `stream_activations()` for large datasets
2. **Reproducibility**: Save probes with `probe.save()`, log hyperparameters
3. **Testing**: Test on both CPU and GPU, include edge cases
