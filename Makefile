.PHONY: hooks lint test test-cov test-integration test-gpu test-e2e build check

hooks:
	uvx pre-commit install --overwrite --install-hooks --hook-type pre-commit --hook-type post-checkout --hook-type pre-push

lint:
	uv run ruff check probelab tests

test:
	uv run pytest tests

test-cov:
	uv run pytest tests --cov=probelab --cov-report=html

test-integration:
	uv run pytest -m "integration and not e2e" tests

test-gpu:
	uv run pytest -m "gpu and not e2e" tests

test-e2e:
	uv run pytest -m e2e tests/test_seed_e2e.py -v -s

build:
	uv build
	uvx twine check dist/*

check: lint test build
