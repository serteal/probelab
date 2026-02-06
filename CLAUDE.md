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

# Load data via registry
dataset = pl.datasets.load("circuit_breakers")
pl.datasets.list_datasets(category="deception")

# Tokenize and collect activations (single layer returns no LAYER axis)
tokens = pl.processing.tokenize_dataset(dataset, tokenizer, mask=pl.masks.assistant())
acts = pl.processing.collect_activations(model, tokens, layers=[16])

# Train probe (auto-detects device from input)
probe = pl.probes.Logistic().fit(acts.pool("sequence", "mean"), labels)

# Evaluate - predict() returns tensor directly
probs = probe.predict(test_acts)  # [batch]
score = pl.metrics.auroc(labels, probs)
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

probelab is a library for training classifiers (probes) on LLM activations. The design philosophy is **tensor-centric** - probes return raw tensors, not wrapper objects.

### Core Concepts

1. **Activations**: Hidden states extracted from LLM layers, with axis-aware operations
2. **Probes**: Classifiers that operate on Activations and return probability tensors
3. **Masks**: Control which tokens are used for training
4. **Utils**: Standalone pooling functions for aggregation

### Module Structure

```
probelab/
├── __init__.py          # Exports: Activations, Label
├── types.py             # Core types (Label, Message, Dialogue, Role)
├── masks.py             # Mask functions (all, assistant, user, nth_message, etc.)
├── metrics.py           # auroc, recall_at_fpr
├── logger.py            # Logging config
│
├── processing/          # Tokenization + activation collection
│   ├── tokenization.py     # Tokens, tokenize_dialogues, tokenize_dataset
│   ├── activations.py      # Activations, collect_activations, stream_activations
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
│   ├── attention.py        # Attention-based
│   ├── multimax.py         # Multi-head hard max
│   └── gated_bipolar.py    # AlphaEvolve gated bipolar
│
├── datasets/            # Dataset classes (17 files)
│   ├── base.py             # Dataset base class
│   └── ...                 # Domain-specific datasets
│
└── utils/               # Utilities
    ├── pooling.py          # pool, ema, rolling functions
    └── validation.py       # check_activations
```

### Core API Pattern

The API follows a **two-step** pattern: tokenize first, then collect activations:

```python
import probelab as pl

# 1. Tokenize with mask (determines which tokens to extract)
tokens = pl.processing.tokenize_dataset(dataset, tokenizer, mask=pl.masks.assistant())

# 2. Collect activations (single layer has no LAYER axis, multiple layers keep it)
acts = pl.processing.collect_activations(model, tokens, layers=[16])

# 3. Pool sequence dimension
prepared = acts.pool("sequence", "mean")

# 4. Train probe and predict (chained)
probs = pl.probes.Logistic().fit(prepared, labels).predict(test_prepared)
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
# collect_activations behavior:
# - Single layer (layers=[16]): returns [batch, seq, hidden] - no LAYER axis
# - Multiple layers (layers=[8, 16, 20]): returns [layer, batch, seq, hidden]

# Methods
acts.select(layer=16)           # Select single layer from multi-layer, removes LAYER axis
acts.select(layers=[8, 16])     # Select multiple layers, keeps LAYER axis
acts.pool("sequence", "mean")   # Pool over sequence dimension
acts.pool("layer", "mean")      # Pool over layer dimension
acts.to("cuda")                 # Move to device

# Properties
acts.shape                      # Tensor shape
acts.has_axis(Axis.LAYER)       # Check if axis exists
acts.axis_size(Axis.BATCH)      # Size of axis
```

**Probes** - Classifiers with two interfaces:
```python
# probe(x) - Differentiable forward pass, returns logits tensor
logits = probe(features)  # [batch] or [n_tokens]

# probe.predict(X) - Convenience method, returns probabilities tensor
probs = probe.predict(activations)  # [batch, 2] or [n_tokens, 2]

# Save/load
probe.save("probe.pt")
probe = pl.probes.Logistic.load("probe.pt", device="cuda")
```

**Utils** - Standalone pooling functions for token-level aggregation:
```python
# Pool token-level predictions to sequence-level
probs = pl.utils.pool(token_probs, mask, "mean")  # or "max", "last_token"
probs = pl.utils.ema(token_probs, mask, alpha=0.5)  # EMA + max
probs = pl.utils.rolling(token_probs, mask, window_size=10)  # rolling mean + max
```

### Common Workflows

**1. Basic Training**
```python
import probelab as pl

# Load and split data
train_ds, test_ds = pl.datasets.load("circuit_breakers").split(0.8)

# Tokenize
train_tokens = pl.processing.tokenize_dataset(train_ds, tokenizer, mask=pl.masks.assistant())
test_tokens = pl.processing.tokenize_dataset(test_ds, tokenizer, mask=pl.masks.assistant())

# Collect activations (single layer, no LAYER axis)
train_acts = pl.processing.collect_activations(model, train_tokens, layers=[16])
test_acts = pl.processing.collect_activations(model, test_tokens, layers=[16])

# Pool and train
train_prepared = train_acts.pool("sequence", "mean")
test_prepared = test_acts.pool("sequence", "mean")
probe = pl.probes.Logistic().fit(train_prepared, train_ds.labels)

# Evaluate
probs = probe.predict(test_prepared)
print(f"AUROC: {pl.metrics.auroc(test_ds.labels, probs):.3f}")
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
    probs = probe.predict(test_prepared)
    results[layer] = pl.metrics.auroc(test_labels, probs)
```

**3. Token-Level with Aggregation**
```python
# Collect single layer (no LAYER axis, keeps SEQ)
acts = pl.processing.collect_activations(model, tokens, layers=[16])

# Train on tokens (don't pool - keep SEQ axis)
probe = pl.probes.Logistic().fit(acts, labels)

# Predict token-level [batch, seq], then aggregate
token_probs = probe.predict(test_acts)
pooled = pl.utils.pool(token_probs, mask, "mean")  # [batch]
```

**4. Using Masks**
```python
# Only detect on last assistant message (compose with & operator)
mask = pl.masks.assistant() & pl.masks.nth_message(-1)

tokens = pl.processing.tokenize_dataset(dataset, tokenizer, mask=mask)
acts = pl.processing.collect_activations(model, tokens, layers=[16])
prepared = acts.pool("sequence", "mean")
```

**5. Streaming for Large Datasets**
```python
# Stream activations to avoid OOM
tokens = pl.processing.tokenize_dataset(large_dataset, tokenizer, mask=pl.masks.all())

for acts_batch, indices, seq_len in pl.processing.stream_activations(model, tokens, layers=[16]):
    # Process each batch incrementally (stream returns raw [layer, batch, seq, hidden])
    # ... accumulate results
```

**6. Differentiable Probe for Fine-tuning**
```python
# Use probe(x) for gradient-based training
hidden_states = model(..., output_hidden_states=True).hidden_states[layer]
logits = probe(hidden_states)  # Differentiable!
loss = some_loss_fn(logits, targets)
loss.backward()
```

### Adding New Probes

1. Create class inheriting from `BaseProbe` in `probes/`
2. Implement:
   - `__call__(x: Tensor) -> Tensor` - differentiable forward, returns logits
   - `predict(X: Activations) -> Tensor` - returns probabilities
   - `fit(X: Activations, y) -> self`
   - `save()`, `load()`
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
