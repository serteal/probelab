# probelab

`probelab` trains probes and activation monitors on language model activations.
It is collector-agnostic: use activations from Transformers, TransformerLens,
NNsight, nnterp, vLLM-Lens, mirin, PyTorch hooks, or saved tensors.

## Installation

```bash
pip install probelab
# or
uv add probelab
```
`probelab` does not choose a CUDA, ROCm, XPU, or CPU PyTorch build for you. Use
your environment or lockfile to select the torch backend.

## Quick Start

This example trains a probe on synthetic activations. Replace `dataset` with
activations from any collector and the probing code stays the same.

```python
import torch
import probelab as pl

n_train, n_test = 96, 32
seq_len, hidden_size = 24, 128
n = n_train + n_test

dataset = torch.randn(n, seq_len, hidden_size)
# shape: [(B)atch_size, (S)eq_len, (H)idden_size]
labels = torch.tensor([0, 1] * (n // 2))

train_acts = pl.Activations(dataset[:n_train], dims="bsh")
test_acts = pl.Activations(dataset[n_train:], dims="bsh")

# Simple probes train on one feature vector per sample.
train_features = train_acts.mean("s")  # [B, H]
test_features = test_acts.mean("s")  # [B, H]

probe = pl.probes.Logistic().fit(train_features, labels[:n_train])

scores = probe.predict(test_features)
print("AUROC:", pl.metrics.auroc(labels[n_train:], scores))
print("Recall@1%FPR:", pl.metrics.recall_at_fpr(labels[n_train:], scores, fpr=0.01))
```

## Collecting activations

In practice you collect activations from a real model. The pipeline is:
*dataset → tokenize (with a detection mask) → collect → probe*. The collection
adapter is reachable as `probelab.collection` and imports its backend lazily:

```python
import probelab as pl
from probelab import collection

# 1. Load (or build) a dataset of dialogues + labels.
dataset = pl.datasets.load("circuit_breakers")
train, test = dataset.split(0.8, stratified=True)

# 2. Tokenize, choosing which tokens to score with a mask.
tokens = pl.tokenize_dataset(train, tokenizer, mask=pl.masks.assistant())

# 3. Collect pooled activations for one or more layers.
#    (requires `probelab[collection]` and the mirin backend)
acts = collection.collect_activations(model, tokens, layers=[12], pool="mean")

# 4. Train and evaluate a probe — same code as the synthetic example above.
probe = pl.probes.Logistic().fit(acts, train.labels)
```

Pass `pool=None` to `collect_activations` to keep token-level activations, then
reduce them yourself with `acts.mean("s")`, `acts.last()`, or train a
sequence probe (`pl.probes.Attention`, `pl.probes.MHA`, ...) directly.

## Documentation

In-depth guides live in [`docs/`](docs/): the activation data model, the mask
cookbook, the probe gallery, collecting activations, and storage. Runnable
scripts are in [`examples/`](examples/).

## Development

See [CONTRIBUTING.md](CONTRIBUTING.md) for the full development workflow, test
markers, and the release process.

```bash
uv sync --all-extras --dev
make check   # lint + test + build
```

## Citation

```bibtex
@software{probelab2026,
  title = {probelab: A library for training probes on LLM activations},
  author = {Alex Serrano},
  url = {https://github.com/serteal/probelab},
  version = {0.1.0},
  year = {2026},
}
```
