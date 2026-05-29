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

Releases are automated from a version bump. PyPI publishing uses Trusted
Publishing (no API tokens).

To cut a release, open a normal PR that:

1. Bumps `__version__` in `probelab/_version.py` (the single source of truth —
   the build, `probelab.__version__`, and saved-artifact version tags all read
   it). The new version must be a valid PEP 440 version, strictly greater than
   the current one.
2. Adds a matching section to `CHANGELOG.md` with a `## <version>` heading
   (e.g. `## 0.1.1 - 2026-06-01`).

The PR-time **Release check** workflow (`.github/workflows/release-check.yml`)
enforces this: if the PR changes `probelab/_version.py`, the bump must be valid
(PEP 440, strictly increasing, tag not already used) and `CHANGELOG.md` must
carry the matching section, or the check fails and blocks merge. PRs that do not
touch the version pass untouched.

When the PR merges to `main` and CI passes, the **Release** workflow
(`.github/workflows/release.yml`) automatically:

- re-validates the version bump,
- creates the `v<version>` tag and a GitHub Release with the changelog notes.

Publishing the GitHub Release triggers `publish.yml`, which builds and uploads
to PyPI.

Merging a PR that does **not** bump `probelab/_version.py` releases nothing — it
is just a normal change. Run `make check` locally before opening any PR.

PyPI versions are immutable. If a published release has a bug, bump to a new
patch version rather than trying to replace the existing one.

