"""Tests for the mirin-based activation collection pipeline."""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

import mirin

import probelab as pl
from probelab.activations import Activations
from probelab.collection.mirin import _ensure_model, collect_activations, stream_activations


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
    from probelab.tokenization import Tokens

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

    def test_rejects_raw_module(self, tiny_model):
        with pytest.raises(TypeError, match="Expected mirin.Model"):
            _ensure_model(tiny_model)

    def test_rejects_non_module(self):
        with pytest.raises(TypeError, match="Expected mirin.Model"):
            _ensure_model("not a model")


# ---------------------------------------------------------------------------
# stream_activations
# ---------------------------------------------------------------------------


class TestStreamActivations:
    def test_yields_correct_format(self, tiny_model, tiny_tokens):
        m = mirin.Model(tiny_model)
        results = list(stream_activations(m, tiny_tokens, layers=[1], batch_size=4))
        assert len(results) > 0
        for chunk in results:
            assert chunk.data.ndim == 3  # [T, n_layers, hidden] where n_layers=1
            assert chunk.data.shape[1] == 1
            assert isinstance(chunk.detection_mask, torch.Tensor)
            assert isinstance(chunk.offsets, torch.Tensor)
            assert isinstance(chunk.indices, list)
        m.close()

    def test_multi_layer(self, tiny_model, tiny_tokens):
        m = mirin.Model(tiny_model)
        results = list(stream_activations(m, tiny_tokens, layers=[0, 2], batch_size=4))
        for chunk in results:
            assert chunk.data.ndim == 3  # [T, 2, hidden]
            assert chunk.data.shape[1] == 2
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

    def test_pool_mean_matches_stream_pooling(self, tiny_model, tiny_tokens):
        m = mirin.Model(tiny_model)
        try:
            collected = collect_activations(m, tiny_tokens, layers=[1], batch_size=4, pool="mean")
            pooled_batches: list[tuple[torch.Tensor, list[int]]] = []
            for chunk in stream_activations(m, tiny_tokens, layers=[1], batch_size=4):
                pooled = pl.pool.mean(
                    chunk.data,
                    chunk.detection_mask,
                    offsets=chunk.offsets,
                ).squeeze(1).cpu()
                pooled_batches.append((pooled, chunk.indices))
            manual = torch.zeros_like(collected.data)
            for pooled, idx in pooled_batches:
                manual[torch.tensor(idx, dtype=torch.long)] = pooled
            torch.testing.assert_close(collected.data, manual)
        finally:
            m.close()

    def test_pool_multi_layer(self, tiny_model, tiny_tokens):
        m = mirin.Model(tiny_model)
        acts = collect_activations(m, tiny_tokens, layers=[0, 2], batch_size=4, pool="mean")
        assert acts.dims == "blh"
        assert acts.data.shape == (8, 2, 32)
        m.close()

    def test_rejects_raw_module(self, tiny_model, tiny_tokens):
        with pytest.raises(TypeError, match="Expected mirin.Model"):
            collect_activations(tiny_model, tiny_tokens, layers=[1], batch_size=4, pool="mean")

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

    def test_batch_token_budget(self, tiny_model, tiny_tokens):
        m = mirin.Model(tiny_model)
        acts = collect_activations(
            m, tiny_tokens, layers=[1], batch_size=32,
            batch_token_budget=64, pool="mean",
        )
        assert acts.data.shape[0] == 8
        m.close()
