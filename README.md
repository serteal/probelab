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
from probelab.transforms import pre

model = AutoModelForCausalLM.from_pretrained(
    model_name := "meta-llama/Llama-3.1-8B-Instruct",
    torch_dtype=torch.bfloat16,
    device_map="auto",
)
tokenizer = AutoTokenizer.from_pretrained(model_name)

train_ds, test_ds = pl.datasets.load("repe").split(0.8)

train_acts, test_acts = (
    pl.collect_activations(
        model=model,
        tokenizer=tokenizer,
        data=data,
        layers=[16],  # collect activations from layer 16
        mask=pl.masks.assistant(),  # collect only assistant messages
        batch_size=32,
    )
    for data in (train_ds, test_ds)
)

# Create a pipeline with preprocessing steps + probe
pipeline = pl.Pipeline([
    ("select", pre.SelectLayer(16)),
    ("pool", pre.Pool(dim="sequence", method="mean")),
    ("probe", pl.probes.Logistic(device="cuda")),
])

pipeline.fit(train_acts, train_ds.labels)
probs = pipeline.predict(test_acts)  # Returns [batch, 2] probabilities

y_pred = probs[:, 1].cpu().numpy()
y_true = [label.value for label in test_ds.labels]
print(f"AUROC: {pl.metrics.auroc(y_true, y_pred):.3f}")
```

## What does `probelab` allow?

### Multi-Pipeline Training and Evaluation

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

import probelab as pl
from probelab.transforms import pre

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

# Create pipelines with preprocessing steps + probe
pipelines = {
    "logistic": pl.Pipeline([
        ("select", pre.SelectLayer(16)),
        ("pool", pre.Pool(dim="sequence", method="mean")),
        ("probe", pl.probes.Logistic(device="cuda")),
    ]),
    "mlp": pl.Pipeline([
        ("select", pre.SelectLayer(16)),
        ("pool", pre.Pool(dim="sequence", method="mean")),
        ("probe", pl.probes.MLP(device="cuda")),
    ]),
}

# Step 1: Collect activations
train_acts = pl.collect_activations(
    model=model,
    tokenizer=tokenizer,
    data=train_dataset,
    layers=[16],
    mask=mask,
    batch_size=32,
)

# Step 2: Train pipelines
for name, pipeline in pipelines.items():
    pipeline.fit(train_acts, train_dataset.labels)

# Step 3: Evaluate
test_acts = pl.collect_activations(
    model=model,
    tokenizer=tokenizer,
    data=test_dataset,
    layers=[16],
    mask=mask,
    batch_size=32,
)

for name, pipeline in pipelines.items():
    probs = pipeline.predict(test_acts)
    y_pred = probs[:, 1].cpu().numpy()
    y_true = [label.value for label in test_dataset.labels]
    print(f"{name}: AUROC={pl.metrics.auroc(y_true, y_pred):.3f}")
```

### Token-Level Training with Post-Probe Aggregation

Train on individual tokens, then aggregate predictions at inference time using methods from the GDM paper:

```python
import probelab as pl
from probelab.transforms import pre, post

# Token-level probe with EMA aggregation (GDM paper style)
pipeline = pl.Pipeline([
    ("select", pre.SelectLayer(16)),
    ("probe", pl.probes.Logistic(device="cuda")),  # Token-level predictions
    ("ema", post.EMAPool(alpha=0.5)),  # Aggregate with exponential moving average
])

# Or use rolling window aggregation
pipeline = pl.Pipeline([
    ("select", pre.SelectLayer(16)),
    ("probe", pl.probes.Logistic(device="cuda")),
    ("rolling", post.RollingPool(window_size=10)),
])
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
