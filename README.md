# `probelab`

A library for training probes on LLM activations.

## Installation

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

# Load model
model = AutoModelForCausalLM.from_pretrained(
    model_name := "meta-llama/Llama-3.1-8B-Instruct",
    torch_dtype=torch.bfloat16,
    device_map="auto",
)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Combine datasets to build a balanced binary task
train_ds, test_ds = (
    pl.datasets.CircuitBreakersDataset() + pl.datasets.BenignInstructionsDataset()
).split(0.8)

# Tokenize with mask selecting assistant tokens only
train_tokens = pl.processing.tokenize_dataset(train_ds, tokenizer, mask=pl.masks.assistant())
test_tokens = pl.processing.tokenize_dataset(test_ds, tokenizer, mask=pl.masks.assistant())

# Collect activations
train_acts = pl.processing.collect_activations(model, train_tokens, layers=[16])
test_acts = pl.processing.collect_activations(model, test_tokens, layers=[16])

# Prepare: select layer, pool over sequence
train_prepared = train_acts.select(layer=16).pool("sequence", "mean")
test_prepared = test_acts.select(layer=16).pool("sequence", "mean")

# Train probes
probes = {
    "logistic": pl.probes.Logistic(device="cuda").fit(train_prepared, train_ds.labels),
    "mlp": pl.probes.MLP(device="cuda").fit(train_prepared, train_ds.labels),
}

# Evaluate
for name, probe in probes.items():
    scores = probe.predict(test_prepared)
    y_pred = scores.scores[:, 1].cpu().numpy()
    y_true = [label.value for label in test_ds.labels]
    print(f"{name}: AUROC={pl.metrics.auroc(y_true, y_pred):.3f}")
```

## Development

```bash
uv run pytest tests/
uv run pytest tests/ --cov=probelab --cov-report=html
```

## Citation

```bibtex
@software{probelab2025,
  title = {probelab: A library for training probes on LLM activations},
  author = {Alex Serrano},
  url = {https://github.com/serteal/probelab},
  version = {0.1.0},
  year = {2025},
}
```
