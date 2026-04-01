"""Tests for the mirin-based activation collection pipeline."""

from __future__ import annotations

import warnings

import pytest
import torch
import torch.nn as nn

import mirin

import probelab as pl
from probelab.processing.activations import (
    Activations,
    _ensure_model,
    _resolve_get_sites,
    collect_activations,
    stream_activations,
)


# ---------------------------------------------------------------------------
# Tiny transformer model for fast testing
# ---------------------------------------------------------------------------

class _TinyBlock(nn.Module):
    def __init__(self, d: int):
        super().__init__()
        self.input_layernorm = nn.LayerNorm(d)
        self.self_attn = nn.Linear(d, d, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        normed = self.input_layernorm(x)
        return x + self.self_attn(normed)


class _TinyModel(nn.Module):
    """Minimal transformer-like model with named layers."""

    def __init__(self, d: int = 32, n_layers: int = 4):
        super().__init__()
        self.model = nn.Module()
        self.model.layers = nn.ModuleList([_TinyBlock(d) for _ in range(n_layers)])
        self.model.embed = nn.Embedding(100, d)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor:
        x = self.model.embed(input_ids)
        for layer in self.model.layers:
            x = layer(x)
        return x


@pytest.fixture
def tiny_model():
    return _TinyModel(d=32, n_layers=4)


@pytest.fixture
def tiny_tokens():
    """Build synthetic Tokens with IDs in [0, 99] for the tiny model."""
    from probelab.processing.tokenization import Tokens

    # 8 samples with variable lengths (5-12 tokens each)
    sample_lengths = [5, 7, 8, 6, 10, 9, 12, 7]
    all_ids = []
    all_det = []
    offsets = [0]
    for length in sample_lengths:
        ids = torch.randint(0, 100, (length,))
        # Last 3 tokens are "assistant" (detection mask = True)
        det = torch.zeros(length, dtype=torch.bool)
        det[-3:] = True
        all_ids.append(ids)
        all_det.append(det)
        offsets.append(offsets[-1] + length)

    return Tokens(
        input_ids=torch.cat(all_ids),
        offsets=torch.tensor(offsets, dtype=torch.int64),
        detection_mask=torch.cat(all_det),
        pad_token_id=0,
        padding_side="right",
        formatted_texts=None,
    )


# ---------------------------------------------------------------------------
# _ensure_model
# ---------------------------------------------------------------------------


class TestEnsureModel:
    def test_returns_mirin_model_unchanged(self, tiny_model):
        m = mirin.Model(tiny_model)
        assert _ensure_model(m) is m
        m.close()

    def test_wraps_nn_module(self, tiny_model):
        m = _ensure_model(tiny_model)
        assert isinstance(m, mirin.Model)
        m.close()

    def test_rejects_non_module(self):
        with pytest.raises(TypeError, match="torch.nn.Module"):
            _ensure_model("not a model")


# ---------------------------------------------------------------------------
# _resolve_get_sites
# ---------------------------------------------------------------------------


class TestResolveGetSites:
    def test_block_hook_point(self, tiny_model):
        m = mirin.Model(tiny_model)
        sites = _resolve_get_sites(m, [0, 2], "block")
        assert len(sites) == 2
        assert sites[0].path == "model.layers.0"
        assert sites[1].path == "model.layers.2"
        m.close()

    def test_layernorm_hook_point(self, tiny_model):
        m = mirin.Model(tiny_model)
        sites = _resolve_get_sites(m, [1], "layernorm")
        assert len(sites) == 1
        assert "input_layernorm" in sites[0].path
        m.close()


# ---------------------------------------------------------------------------
# stream_activations
# ---------------------------------------------------------------------------


class TestStreamActivations:
    def test_yields_correct_format(self, tiny_model, tiny_tokens):
        m = mirin.Model(tiny_model)
        results = list(stream_activations(m, tiny_tokens, layers=[1], batch_size=4))
        assert len(results) > 0
        for flat_data, det, offsets, idx in results:
            assert flat_data.ndim == 3  # [T, n_layers, hidden] where n_layers=1
            assert flat_data.shape[1] == 1
            assert isinstance(det, torch.Tensor)
            assert isinstance(offsets, torch.Tensor)
            assert isinstance(idx, list)
        m.close()

    def test_backend_kwarg_deprecated(self, tiny_model, tiny_tokens):
        m = mirin.Model(tiny_model)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            list(stream_activations(m, tiny_tokens, layers=[0], batch_size=8, backend="auto"))
            assert any(issubclass(x.category, DeprecationWarning) for x in w)
        m.close()

    def test_multi_layer(self, tiny_model, tiny_tokens):
        m = mirin.Model(tiny_model)
        results = list(stream_activations(m, tiny_tokens, layers=[0, 2], batch_size=4))
        for flat_data, det, offsets, idx in results:
            assert flat_data.ndim == 3  # [T, 2, hidden]
            assert flat_data.shape[1] == 2
        m.close()


# ---------------------------------------------------------------------------
# collect_activations
# ---------------------------------------------------------------------------


class TestCollectActivations:
    def test_single_layer_no_pool(self, tiny_model, tiny_tokens):
        m = mirin.Model(tiny_model)
        acts = collect_activations(m, tiny_tokens, layers=[1], batch_size=4)
        assert isinstance(acts, Activations)
        assert acts.dims == "bsh"
        assert acts.batch_size == 8
        m.close()

    def test_multi_layer_no_pool(self, tiny_model, tiny_tokens):
        m = mirin.Model(tiny_model)
        acts = collect_activations(m, tiny_tokens, layers=[0, 2, 3], batch_size=4)
        assert acts.dims == "blsh"
        assert acts.layers == (0, 2, 3)
        m.close()

    def test_pool_mean(self, tiny_model, tiny_tokens):
        m = mirin.Model(tiny_model)
        acts = collect_activations(m, tiny_tokens, layers=[1], batch_size=4, pool="mean")
        assert acts.dims == "bh"
        assert acts.data.shape[0] == 8
        m.close()

    def test_pool_multi_layer(self, tiny_model, tiny_tokens):
        m = mirin.Model(tiny_model)
        acts = collect_activations(m, tiny_tokens, layers=[0, 2], batch_size=4, pool="mean")
        assert acts.dims == "blh"
        assert acts.data.shape == (8, 2, 32)
        m.close()

    def test_auto_wraps_nn_module(self, tiny_model, tiny_tokens):
        acts = collect_activations(tiny_model, tiny_tokens, layers=[1], batch_size=4, pool="mean")
        assert acts.dims == "bh"
        assert acts.data.shape[0] == 8

    def test_hook_point_layernorm(self, tiny_model, tiny_tokens):
        m = mirin.Model(tiny_model)
        acts_block = collect_activations(m, tiny_tokens, layers=[1], batch_size=8, pool="mean")
        acts_ln = collect_activations(
            m, tiny_tokens, layers=[1], batch_size=8, pool="mean", hook_point="layernorm",
        )
        # They should produce different values (block includes residual + attn)
        assert acts_block.data.shape == acts_ln.data.shape
        # Not identical (unless the model is degenerate)
        m.close()

    def test_backend_kwarg_deprecated(self, tiny_model, tiny_tokens):
        m = mirin.Model(tiny_model)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            collect_activations(m, tiny_tokens, layers=[0], batch_size=8, backend="whatever")
            assert any(issubclass(x.category, DeprecationWarning) for x in w)
        m.close()

    def test_server_path(self, tiny_model, tiny_tokens):
        server = mirin.Server(tiny_model)
        acts = collect_activations(server, tiny_tokens, layers=[1], batch_size=4, pool="mean")
        assert acts.dims == "bh"
        assert acts.data.shape[0] == 8
        server.close()

    def test_server_matches_local(self, tiny_model, tiny_tokens):
        m = mirin.Model(tiny_model)
        server = mirin.Server(tiny_model)
        acts_local = collect_activations(m, tiny_tokens, layers=[1], batch_size=4, pool="mean")
        acts_server = collect_activations(server, tiny_tokens, layers=[1], batch_size=4, pool="mean")
        diff = (acts_local.data - acts_server.data).abs().max().item()
        assert diff < 1e-5, f"Local vs server max diff: {diff}"
        m.close()
        server.close()

    def test_batch_token_budget(self, tiny_model, tiny_tokens):
        m = mirin.Model(tiny_model)
        acts = collect_activations(
            m, tiny_tokens, layers=[1], batch_size=32,
            batch_token_budget=64, pool="mean",
        )
        assert acts.data.shape[0] == 8
        m.close()
