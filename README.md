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

## Development

```bash
make test
make test-cov
make test-integration
make test-gpu
make test-e2e
```

### Release

PyPI publishing runs from GitHub Releases using Trusted Publishing.

1. Update `version` in `pyproject.toml`.
2. Add the release notes to `CHANGELOG.md`.
3. Run `make check`.
4. Commit, push, and wait for CI to pass on `main`.
5. Create the GitHub release:

```bash
gh release create v0.1.1 \
  --target main \
  --title "v0.1.1" \
  --notes-file CHANGELOG.md
```

PyPI versions are immutable. If a published release has a bug, publish a new
patch version instead of trying to replace the existing one.

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
