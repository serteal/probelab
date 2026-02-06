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

# Load and combine datasets
train_ds, test_ds = (
    pl.datasets.load("circuit_breakers") + pl.datasets.load("benign_instructions")
).split(0.8)

# Tokenize with mask selecting assistant tokens only
train_tokens = pl.tokenize_dataset(train_ds, tokenizer, mask=pl.masks.assistant())
test_tokens = pl.tokenize_dataset(test_ds, tokenizer, mask=pl.masks.assistant())

# Collect activations (single layer returns no LAYER axis)
train_acts = pl.collect_activations(model, train_tokens, layers=[16])
test_acts = pl.collect_activations(model, test_tokens, layers=[16])

# Pool over sequence dimension
train_prepared = train_acts.mean_pool()
test_prepared = test_acts.mean_pool()

# Train and evaluate (probes auto-detect device from input)
for name, probe in [
    ("logistic", pl.probes.Logistic()),
    ("mlp", pl.probes.MLP()),
]:
    probe.fit(train_prepared, train_ds.labels)
    probs = probe.predict(test_prepared)
    print(f"{name}: AUROC={pl.metrics.auroc(test_ds.labels, probs):.3f}")
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
