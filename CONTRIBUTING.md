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

Releases are **tag-driven**. The version is derived from the git tag at build
time by [`hatch-vcs`](https://github.com/ofek/hatch-vcs) — there is no version
file to edit, and the tag is the single source of truth. PyPI publishing uses
Trusted Publishing (no API tokens).

To cut a release:

1. Add a section to `CHANGELOG.md` with a `## <version>` heading
   (e.g. `## 0.1.1 - 2026-06-01`) and merge it to `main`.
2. Tag the release commit and push the tag:

   ```bash
   git tag v0.1.1
   git push origin v0.1.1
   ```

Pushing a `v*` tag triggers `publish.yml`, which:

- builds the dists (the version is read from the tag),
- verifies the tag matches the built version,
- creates a GitHub Release (notes from the matching `CHANGELOG.md` section, or
  auto-generated if none is found),
- uploads to PyPI.

`probelab.__version__` is read at runtime from the installed package metadata
via `importlib.metadata`.

Run `make check` locally before tagging. PyPI versions are immutable: if a
published release has a bug, tag a new patch version rather than trying to
replace the existing one.

