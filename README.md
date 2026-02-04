# `probelab`

A library for training probes on LLM activations.

## Features

- Efficient extraction of activations from any layer of transformer models
- Train various probe architectures (logistic regression, MLPs, attention-based)
- Work with dialogue/message-based data using fine-grained detection masks
- 60+ built-in datasets across 12 categories with a registry API

## Installation

### From Source (Development)

```bash
uv add git+https://github.com/serteal/probelab.git
```

or:

```bash
git clone https://github.com/serteal/probelab.git
cd probelab
uv sync
```

## Quick Start

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

import probelab as pl

model = AutoModelForCausalLM.from_pretrained(
    model_name := "meta-llama/Llama-3.1-8B-Instruct",
    torch_dtype=torch.bfloat16,
    device_map="auto",
)
tokenizer = AutoTokenizer.from_pretrained(model_name)

train_ds, test_ds = pl.datasets.load("repe").split(0.8)

# Collect activations
train_acts = pl.collect_activations(
    model=model,
    tokenizer=tokenizer,
    data=train_ds,
    layers=[16],
    mask=pl.masks.assistant(),
    batch_size=32,
)

# Prepare activations and train probe
prepared = train_acts.select(layer=16).pool("sequence", "mean")
probe = pl.probes.Logistic(device="cuda").fit(prepared, train_ds.labels)

# Evaluate
test_acts = pl.collect_activations(
    model=model,
    tokenizer=tokenizer,
    data=test_ds,
    layers=[16],
    mask=pl.masks.assistant(),
    batch_size=32,
)
test_prepared = test_acts.select(layer=16).pool("sequence", "mean")
scores = probe.predict(test_prepared)

y_pred = scores.scores[:, 1].cpu().numpy()
y_true = [label.value for label in test_ds.labels]
print(f"AUROC: {pl.metrics.auroc(y_true, y_pred):.3f}")
```

## What does `probelab` allow?

### Multi-Probe Training and Evaluation

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

import probelab as pl

model = AutoModelForCausalLM.from_pretrained(
    model_name := "meta-llama/Llama-3.1-8B-Instruct",
    torch_dtype=torch.bfloat16,
    device_map="auto",
)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Combine datasets to build a balanced binary task
train_dataset, test_dataset = (
    pl.datasets.CircuitBreakersDataset() + pl.datasets.BenignInstructionsDataset()
).split(0.8)

# Compose a role-based mask and drop special tokens
mask = pl.masks.assistant() & ~pl.masks.special_tokens()

# Collect activations once
train_acts = pl.collect_activations(
    model=model,
    tokenizer=tokenizer,
    data=train_dataset,
    layers=[16],
    mask=mask,
    batch_size=32,
)

# Prepare activations
train_prepared = train_acts.select(layer=16).pool("sequence", "mean")

# Train multiple probes
probes = {
    "logistic": pl.probes.Logistic(device="cuda").fit(train_prepared, train_dataset.labels),
    "mlp": pl.probes.MLP(device="cuda").fit(train_prepared, train_dataset.labels),
}

# Evaluate
test_acts = pl.collect_activations(
    model=model,
    tokenizer=tokenizer,
    data=test_dataset,
    layers=[16],
    mask=mask,
    batch_size=32,
)
test_prepared = test_acts.select(layer=16).pool("sequence", "mean")

for name, probe in probes.items():
    scores = probe.predict(test_prepared)
    y_pred = scores.scores[:, 1].cpu().numpy()
    y_true = [label.value for label in test_dataset.labels]
    print(f"{name}: AUROC={pl.metrics.auroc(y_true, y_pred):.3f}")
```

### Token-Level Training with Score Aggregation

Train on individual tokens, then aggregate predictions at inference time:

```python
import probelab as pl

# Collect activations (keep sequence dimension)
acts = pl.collect_activations(
    model=model,
    tokenizer=tokenizer,
    data=dataset,
    layers=[16],
    mask=pl.masks.assistant(),
    batch_size=32,
)

# Train on tokens (no sequence pooling)
prepared = acts.select(layer=16)
probe = pl.probes.Logistic(device="cuda").fit(prepared, labels)

# Predict and aggregate scores
scores = probe.predict(test_acts.select(layer=16))

# Different aggregation methods
mean_scores = scores.pool("sequence", "mean")  # Simple mean
ema_scores = scores.ema(alpha=0.5)             # Exponential moving average
rolling_scores = scores.rolling(window_size=10) # Rolling window mean
```

### Multi-Layer Analysis

```python
import probelab as pl

# Collect from multiple layers
acts = pl.collect_activations(
    model=model,
    tokenizer=tokenizer,
    data=dataset,
    layers=[8, 12, 16, 20, 24],
    mask=pl.masks.assistant(),
    batch_size=32,
)

# Train probe on each layer
results = {}
for layer in [8, 12, 16, 20, 24]:
    prepared = acts.select(layer=layer).pool("sequence", "mean")
    probe = pl.probes.Logistic().fit(prepared, labels)

    test_prepared = test_acts.select(layer=layer).pool("sequence", "mean")
    scores = probe.predict(test_prepared)
    results[layer] = pl.metrics.auroc(test_labels, scores.scores[:, 1])

print(results)  # {8: 0.72, 12: 0.81, 16: 0.89, 20: 0.85, 24: 0.78}
```

### Dataset Registry

Discover and load datasets easily:

```python
import probelab as pl

# List all available datasets
pl.datasets.list_datasets()
# ['ai_audit', 'ai_liar', 'alpaca', 'benign_instructions', ...]

# List datasets by category
pl.datasets.list_datasets(category="deception")
# ['ai_audit', 'ai_liar', 'dolus_chat', 'insider_trading', ...]

# List all categories
pl.datasets.list_categories()
# ['agents', 'creative', 'cybersecurity', 'deception', 'harmfulness', ...]

# Get dataset info
pl.datasets.info("circuit_breakers")
# {'name': 'circuit_breakers', 'category': 'harmfulness', 'description': ...}

# Load by name
dataset = pl.datasets.load("dolus_chat")

# Or use class directly
dataset = pl.datasets.DolusChatDataset()
```

### Composable Masks

Fine-grained control over which tokens to detect:

```python
import probelab as pl

# Only detect on assistant messages
mask = pl.masks.assistant()

# Detect on last assistant message only
mask = pl.masks.assistant() & pl.masks.nth_message(-1)

# Exclude special tokens
mask = pl.masks.assistant() & ~pl.masks.special_tokens()

# Detect on tokens containing specific text
mask = pl.masks.contains("yes") | pl.masks.contains("no")

# Complex combinations
mask = (
    pl.masks.assistant()
    & pl.masks.nth_message(-1)
    & ~pl.masks.special_tokens()
)
```

### Metrics with Bootstrap Confidence Intervals

```python
import probelab as pl
from probelab.metrics import with_bootstrap

# Standard metrics
auroc = pl.metrics.auroc(y_true, y_pred)
recall = pl.metrics.recall_at_fpr(y_true, y_pred, fpr=0.05)

# Get metric by name (useful for configs)
metric_fn = pl.metrics.get_metric_by_name("recall@5")  # 5% FPR
score = metric_fn(y_true, y_pred)

# Add bootstrap confidence intervals to any metric
auroc_with_ci = with_bootstrap(n_bootstrap=1000)(pl.metrics.auroc)
score, ci_lower, ci_upper = auroc_with_ci(y_true, y_pred)
print(f"AUROC: {score:.3f} (95% CI: {ci_lower:.3f}-{ci_upper:.3f})")
```

## Development

### Running Tests

```bash
# Run all tests
uv run pytest tests/

# Run specific test categories
uv run pytest tests/probes/
uv run pytest tests/processing/

# Exclude slow tests
uv run pytest -m "not slow"

# With coverage
uv run pytest tests/ --cov=probelab --cov-report=html
```

## Citation

If you use `probelab` in your research, please cite:

```bibtex
@software{probelab2025,
  title = {probelab: A library for training probes on LLM activations},
  author = {Alex Serrano},
  url = {https://github.com/serteal/probelab},
  version = {0.1.0},
  year = {2025},
}
```
