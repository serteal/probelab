# Contributing to probelab

Thanks for your interest in improving `probelab`! This document covers the
development workflow, testing, and the release process.

## Development setup

`probelab` uses [uv](https://docs.astral.sh/uv/) for environment and dependency
management.

```bash
uv sync --all-extras --dev
uv run pre-commit install   # or: make hooks
```

`probelab` does not pin a PyTorch build (CUDA/ROCm/XPU/CPU). Select the torch
backend through your own environment or lockfile.

## Running the checks

```bash
make lint              # ruff
make test              # unit tests (excludes e2e by default)
make test-cov          # tests with HTML coverage
make test-integration  # tests that exercise optional deps / external services
make test-gpu          # tests that require accelerator hardware
make test-e2e          # full external workflows (cluster/network)
make check             # lint + test + build
```

### Test markers and offline runs

Tests are grouped with pytest markers (see `pyproject.toml`):

- `integration` — exercises optional dependencies or the network (e.g. tokenizer
  downloads from the HuggingFace Hub).
- `gpu` — requires accelerator hardware.
- `e2e` — full external workflows, excluded from the default run.
- `slow` — too slow for normal local feedback.

Tests that need a HuggingFace tokenizer use `tests/helpers.load_tokenizer_or_skip`,
which **skips** (rather than errors) when the model cannot be fetched, so the
suite stays runnable offline. The default `make test` run only requires network
access for the `integration` tests.

## Coding conventions

- Match the style of the surrounding code; `ruff` (lint + format) is the source
  of truth and runs in `pre-commit` and CI.
- Public API lives behind `probelab/__init__.py` and the subpackage `__all__`
  lists. Keep optional/heavy imports (mirin, transformers, h5py) lazy so that
  `import probelab` stays light.
- Add or update tests for behavior changes. Prefer offline-capable tests.

## Release process

PyPI publishing runs from GitHub Releases using Trusted Publishing.

1. Bump the version in `probelab/_version.py` (the single source of truth — the
   build, `probelab.__version__`, and saved-artifact version tags all read it).
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
