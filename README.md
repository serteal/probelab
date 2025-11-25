# `probelib`

A library for training probes on LLM activations.

## Features

- Efficient extraction of activations from any layer of transformer models
- Train various probe architectures (logistic regression, MLPs, attention-based)
- Work with dialogue/message-based data using fine-grained detection masks

## Installation

### From Source (Development)

```bash
uv add git+https://github.com/serteal/probelib.git
```

or:

```bash
git clone https://github.com/serteal/probelib.git
cd probelib
uv sync
```

## Quick Start

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

import probelib as pl

model = AutoModelForCausalLM.from_pretrained(
    model_name := "meta-llama/Llama-3.1-8B-Instruct",
    torch_dtype=torch.bfloat16,
    device_map="auto",
)
tokenizer = AutoTokenizer.from_pretrained(model_name)

train_ds, test_ds = pl.datasets.REPEDataset().split(0.8)

train_activations, test_activations = (
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
    ("select", pl.preprocessing.SelectLayer(16)),
    ("agg", pl.preprocessing.AggregateSequences("mean")),
    ("probe", pl.probes.Logistic(device="cuda")),
])

pipeline.fit(train_activations, train_ds.labels)
predictions = pipeline.predict_proba(test_activations)

print(pl.metrics.auroc(test_ds.labels, predictions))
```

## What does `probelib` allow?

### High-level, multi-pipeline training and evaluation

```python
import functools

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

import probelib as pl

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
# This mask selects only assistant messages and drops special tokens (eg. bos, eos, pad)
mask = pl.masks.assistant() & ~pl.masks.special_tokens()

# Create pipelines with preprocessing steps + probe
# Each pipeline selects layer 16, aggregates token scores, then classifies
pipelines = {
    "logistic": pl.Pipeline([
        ("select", pl.preprocessing.SelectLayer(16)),
        ("agg", pl.preprocessing.AggregateTokenScores("mean")),
        ("probe", pl.probes.Logistic(device="cuda")),
    ]),
    "mlp": pl.Pipeline([
        ("select", pl.preprocessing.SelectLayer(16)),
        ("agg", pl.preprocessing.AggregateTokenScores("mean")),
        ("probe", pl.probes.MLP(device="cuda")),
    ]),
}

# Convenience function: one-step training from model
pl.scripts.train_from_model(
    pipelines=pipelines,
    data=train_dataset,
    model=model,
    tokenizer=tokenizer,
    layers=[16],  # required for convenience functions
    mask=mask,
    batch_size=32,
    streaming=True,  # use streaming to train on large datasets efficiently
)

predictions, metrics = pl.scripts.evaluate_from_model(
    pipelines=pipelines,
    data=test_dataset,
    model=model,
    tokenizer=tokenizer,
    layers=[16],  # required for convenience functions
    mask=mask,
    batch_size=32,
    metrics=[
        pl.metrics.auroc,
        functools.partial(pl.metrics.recall_at_fpr, fpr=0.01),
    ],
)

pl.visualization.print_metrics(metrics)
```

### Low-Level Building Blocks (manual control)

```python
import probelib as pl
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained(
    model_name := "meta-llama/Llama-3.1-8B-Instruct",
    torch_dtype=torch.bfloat16,
    device_map="auto",
)
tokenizer = AutoTokenizer.from_pretrained(model_name)

train_dataset, test_dataset = pl.datasets.DolusChatDataset()[:2048].split(0.8)

# Mix role, position, and lexical masks
mask = (pl.masks.assistant() & pl.masks.last_token()) | pl.masks.contains("lie")
# Optional: sanity-check what the mask selects
pl.visualize_mask(train_dataset.dialogues[0], mask, tokenizer, force_terminal=True)

# Collect activations in streaming mode
train_stream = pl.collect_activations(
    model=model,
    tokenizer=tokenizer,
    data=train_dataset,
    layers=[12],
    mask=mask,
    batch_size=16,
    streaming=True, # stream activations so we can partial_fit on large corpora
)

# Create pipeline with preprocessing + probe
pipeline = pl.Pipeline([
    ("select", pl.preprocessing.SelectLayer(12)),
    ("agg", pl.preprocessing.AggregateTokenScores("mean")),
    ("probe", pl.probes.MLP(hidden_dim=256, device="cuda")),
])

# Streaming training with pipeline.partial_fit()
labels = torch.tensor([label.value for label in train_dataset.labels])
for batch in train_stream:
    pipeline.partial_fit(batch, labels[batch.batch_indices])

# Collect test activations in-memory and run bespoke metrics
test_acts = pl.collect_activations(
    model=model,
    tokenizer=tokenizer,
    data=test_dataset,
    layers=[12],
    mask=mask,
    batch_size=32,
)

scores = pipeline.predict_proba(test_acts)[:, 1].cpu().numpy()
y_true = torch.tensor([label.value for label in test_dataset.labels])
auroc, lo, hi = pl.metrics.auroc(y_true, scores)
print(f"AUROC: {auroc:.3f} (95% CI {lo:.3f}-{hi:.3f})")
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
uv run pytest tests/ --cov=probelib --cov-report=html
```

## Citation

If you use `probelib` in your research, please cite:

```bibtex
@software{probelib2025,
  title = {probelib: A library for training probes on LLM activations},
  author = {Alex Serrano},
  url = {https://github.com/serteal/probelib},
  version = {0.1.0},
  year = {2025},
}
```
