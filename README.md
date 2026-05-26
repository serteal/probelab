# `probelab`

A library for training probes on LLM activations.

## Installation

Install the base library for datasets, masks, activation containers, pooling,
metrics, and probes:

```bash
uv add git+https://github.com/serteal/probelab.git
```

Install tokenization or activation-collection dependencies only when you need
them:

```bash
uv add "probelab[tokenization] @ git+https://github.com/serteal/probelab.git"
uv add "probelab[collection] @ git+https://github.com/serteal/probelab.git"
```

`probelab` does not choose a CUDA, ROCm, XPU, or CPU PyTorch build for you.
Use your application environment or lockfile to select the torch backend. For
example:

```bash
# Let uv select the best local torch backend.
uv pip install torch --torch-backend=auto

# Or pin an explicit CUDA build in an application/project environment.
uv pip install torch --index-url https://download.pytorch.org/whl/cu128
```

For source development:

```bash
git clone https://github.com/serteal/probelab.git
cd probelab
uv sync --extra collection
```

## Quick Start: Bring Your Own Activations

```python
import torch
import probelab as pl

train_ds, test_ds = (
    pl.datasets.load("circuit_breakers") + pl.datasets.load("benign_instructions")
).split(0.8)

# These tensors can come from any collector: raw PyTorch hooks,
# TransformerLens, nnsight, mirin, saved files, etc.
train_hidden = torch.randn(len(train_ds), 128, 4096)
test_hidden = torch.randn(len(test_ds), 128, 4096)
train_attention = torch.ones(len(train_ds), 128, dtype=torch.bool)
test_attention = torch.ones(len(test_ds), 128, dtype=torch.bool)

train_acts = pl.Activations.from_padded(
    train_hidden,
    attention_mask=train_attention,
    dims="bsh",
    metadata={
        "model": "my-model",
        "site": "resid_post",
        "collector": "custom-hooks",
        "split": "train",
    },
)
test_acts = pl.Activations.from_padded(
    test_hidden,
    attention_mask=test_attention,
    dims="bsh",
    metadata={
        "model": "my-model",
        "site": "resid_post",
        "collector": "custom-hooks",
        "split": "test",
    },
)

train_prepared = train_acts.mean("s")
test_prepared = test_acts.mean("s")

for name, probe in [("logistic", pl.probes.Logistic()), ("mlp", pl.probes.MLP())]:
    probe.fit(train_prepared, train_ds.labels)
    probs = probe.predict(test_prepared)
    print(f"{name}: AUROC={pl.metrics.auroc(test_ds.labels, probs):.3f}")
```

`Activations.metadata` is an arbitrary runtime dictionary for provenance.
Storage helpers persist it when it is JSON-serializable.

## Optional mirin Collection

```python
import mirin as mi
import probelab as pl
from probelab.collection.mirin import collect_activations
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "meta-llama/Llama-3.1-8B-Instruct"
hf_model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    device_map="auto",
)
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = mi.Model(hf_model, rename=mi.renames.llm, tokenizer=tokenizer)

dataset = pl.datasets.load("circuit_breakers")
tokens = pl.tokenize_dataset(dataset, tokenizer, mask=pl.masks.assistant())
acts = collect_activations(model, tokens, layers=[16])
probe = pl.probes.Logistic().fit(acts.mean("s"), dataset.labels)
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
