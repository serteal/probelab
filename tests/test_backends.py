"""Tests for backend auto-detection and dispatch."""

import torch
import pytest

import probelab as pl
from probelab.backends.transformers import _iter_batches
from probelab.processing.tokenization import Tokens


class _FakeActivationResult:
    def __init__(self, activations: dict[int, list[torch.Tensor]], num_tokens: list[int]):
        self.activations = activations
        self.num_tokens = num_tokens


class _FakeFlatResult:
    def __init__(self, activations: dict[int, list[torch.Tensor]], num_tokens: list[int]):
        self._activations = activations
        self._num_tokens = num_tokens

    def to_activation_result(self) -> _FakeActivationResult:
        return _FakeActivationResult(self._activations, self._num_tokens)


class _FakeActivationEngine:
    """Mimics the vLLM ActivationEngine interface used by the backend."""

    def __init__(self, hidden_size: int = 4):
        self.hidden_size = hidden_size
        self.tokenizer = object()
        self.last_collect_flat_kwargs: dict = {}

    def collect(self, *args, **kwargs):
        return None

    def collect_flat(self, token_ids: list[list[int]], **kwargs):
        self.last_collect_flat_kwargs = kwargs
        activations: dict[int, list[torch.Tensor]] = {}
        for layer in (0, 2):
            layer_out: list[torch.Tensor] = []
            for sample_idx, ids in enumerate(token_ids):
                # Deterministic values so we can assert placement.
                value = float(layer + sample_idx + 1)
                layer_out.append(
                    torch.full((len(ids), self.hidden_size), value, dtype=torch.float32)
                )
            activations[layer] = layer_out
        num_tokens = [len(ids) for ids in token_ids]
        return _FakeFlatResult(activations, num_tokens)


def _make_tokens() -> Tokens:
    return Tokens(
        input_ids=torch.tensor([11, 12, 13, 21, 22], dtype=torch.long),
        offsets=torch.tensor([0, 3, 5], dtype=torch.int64),
        detection_mask=torch.tensor([True, True, True, True, True]),
        pad_token_id=0,
        padding_side="right",
    )


def test_collect_activations_auto_detects_vllm_engine():
    engine = _FakeActivationEngine(hidden_size=3)
    tokens = _make_tokens()

    acts = pl.collect_activations(
        engine,
        tokens,
        layers=[0, 2],
        batch_size=2,
        backend="auto",
    )

    assert acts.dims == "blsh"
    # Flat representation: data is [total_tokens, n_layers, hidden]
    assert acts.data.ndim == 3
    assert acts.n_layers == 2
    assert acts.batch_size == 2
    assert acts.hidden_size == 3
    # total_tokens = 3 + 2 = 5
    assert acts.total_tokens == 5

    # Verify data values via to_padded()
    padded, padded_det = acts.to_padded()
    # padded: [2, 2, 3, 3] (batch, layers, max_seq=3, hidden=3)
    assert padded.shape == (2, 2, 3, 3)
    # Layer 0, sample 0 has value 1.0 for real tokens.
    assert torch.allclose(padded[0, 0, :3], torch.ones((3, 3)))
    # Layer 2, sample 1 has value 4.0 for real tokens.
    assert torch.allclose(padded[1, 1, :2], torch.full((2, 3), 4.0))


def test_backend_override_type_error_for_mismatched_object():
    engine = _FakeActivationEngine()
    tokens = _make_tokens()

    with pytest.raises(TypeError):
        pl.collect_activations(
            engine,
            tokens,
            layers=[0],
            batch_size=2,
            backend="transformers",
        )


def test_backend_context_defaults_propagate_to_vllm_backend():
    engine = _FakeActivationEngine()
    tokens = _make_tokens()

    with pl.backends.Context(vllm_batch_token_budget=1234):
        _ = pl.collect_activations(
            engine,
            tokens,
            layers=[0],
            batch_size=2,
            backend="vllm",
        )

    assert engine.last_collect_flat_kwargs["batch_token_budget"] == 1234
    assert engine.last_collect_flat_kwargs["preserve_input_order"] is True


def test_backend_can_be_selected_from_context_defaults():
    engine = _FakeActivationEngine()
    tokens = _make_tokens()

    with pl.backends.Context(backend="vllm"):
        acts = pl.collect_activations(
            engine,
            tokens,
            layers=[0],
            batch_size=2,
        )

    assert acts.dims == "bsh"


def test_stream_activations_yields_flat_format():
    """Test that stream_activations yields (flat_data, det, offsets, indices)."""
    engine = _FakeActivationEngine(hidden_size=3)
    tokens = _make_tokens()

    for flat_data, det, offsets, idx in pl.processing.stream_activations(
        engine, tokens, layers=[0, 2], batch_size=2, backend="vllm"
    ):
        assert flat_data.ndim == 3  # [T, n_layers, hidden]
        assert det.ndim == 1  # [T]
        assert offsets.ndim == 1  # [batch+1]
        assert offsets.dtype == torch.int64
        assert det.dtype == torch.bool
        assert flat_data.shape[1] == 2  # n_layers
        assert flat_data.shape[2] == 3  # hidden


def _make_varlen_tokens(lengths: list[int]) -> Tokens:
    total = sum(lengths)
    input_ids = torch.arange(total, dtype=torch.long)
    offsets = torch.zeros(len(lengths) + 1, dtype=torch.int64)
    running = 0
    for i, length in enumerate(lengths):
        running += length
        offsets[i + 1] = running
    detection_mask = torch.ones(total, dtype=torch.bool)
    return Tokens(
        input_ids=input_ids,
        offsets=offsets,
        detection_mask=detection_mask,
        pad_token_id=0,
        padding_side="right",
    )


def test_transformers_iter_batches_respects_token_budget():
    tokens = _make_varlen_tokens([5, 4, 3, 2])

    batches = list(
        _iter_batches(tokens, batch_size=4, batch_token_budget=10, sort_by_length=True)
    )

    # Sorted by length and token-budget packed: [5,4] then [3,2]
    idx_lists = [idx for _, idx in batches]
    assert idx_lists == [[0, 1], [2, 3]]

    # Each batch should satisfy padded-token budget
    for _, idx in batches:
        lengths = tokens.lengths[idx]
        padded_tokens = int(lengths.max().item()) * len(idx)
        assert padded_tokens <= 10


def test_transformers_iter_batches_preserves_order_when_unsorted():
    tokens = _make_varlen_tokens([2, 5, 4, 3])

    batches = list(
        _iter_batches(tokens, batch_size=4, batch_token_budget=10, sort_by_length=False)
    )
    idx_lists = [idx for _, idx in batches]
    assert idx_lists == [[0, 1], [2, 3]]


def test_transformers_iter_batches_allows_singleton_over_budget():
    tokens = _make_varlen_tokens([12, 2, 2])

    batches = list(
        _iter_batches(tokens, batch_size=4, batch_token_budget=10, sort_by_length=True)
    )
    idx_lists = [idx for _, idx in batches]
    assert idx_lists[0] == [0]
    assert idx_lists[1] == [1, 2]


def test_transformers_iter_batches_rejects_nonpositive_token_budget():
    tokens = _make_varlen_tokens([3, 2, 1])
    with pytest.raises(ValueError):
        _ = list(_iter_batches(tokens, batch_size=4, batch_token_budget=0))
