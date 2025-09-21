"""
Pytest configuration and shared fixtures for probelib tests.
"""

import pytest
from transformers import AutoTokenizer


def pytest_configure(config):
    """Configure pytest with custom markers."""
    # These are also declared in pyproject.toml; keep here to avoid warnings when running standalone
    config.addinivalue_line("markers", "integration: integration tests (GPU heavy)")
    config.addinivalue_line(
        "markers", "llama3: tests that run against LLaMA 3 models/tokenizers"
    )
    config.addinivalue_line(
        "markers", "gemma2: tests that run against Gemma 2 models/tokenizers"
    )


@pytest.fixture(scope="session")
def device():
    """Return the device to use for tests."""
    import torch

    return "cuda" if torch.cuda.is_available() else "cpu"


@pytest.fixture(scope="session")
def gemma_tokenizer():
    """Gemma-2 tokenizer."""
    return AutoTokenizer.from_pretrained("google/gemma-2-2b-it")


@pytest.fixture(scope="session")
def llama_tokenizer():
    """LLaMA-3 tokenizer."""
    return AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")
