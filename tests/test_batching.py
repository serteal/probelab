"""Tests for tensor batching helpers."""

import pytest
import torch

from probelab.batching import (
    iter_feature_batches,
    pad_sequence_batch,
    iter_sequence_batch_indices,
    iter_sequence_batches,
)


def test_iter_feature_batches_preserves_order_without_shuffle():
    features = torch.arange(12).reshape(6, 2)
    labels = torch.arange(6)

    batches = list(iter_feature_batches(features, labels, batch_size=2, shuffle=False))

    assert [idx.tolist() for _, _, idx in batches] == [[0, 1], [2, 3], [4, 5]]
    assert torch.equal(batches[1][0], features[2:4])
    assert torch.equal(batches[1][1], labels[2:4])


def test_iter_feature_batches_rejects_invalid_batch_size():
    features = torch.arange(6).reshape(3, 2)

    with pytest.raises(ValueError, match="batch_size must be positive"):
        list(iter_feature_batches(features, batch_size=0))


def test_iter_feature_batches_respects_subset_indices():
    features = torch.arange(12).reshape(6, 2)
    labels = torch.arange(6)

    batches = list(
        iter_feature_batches(
            features,
            labels,
            indices=torch.tensor([4, 1, 3]),
            batch_size=2,
            shuffle=False,
        )
    )

    assert [idx.tolist() for _, _, idx in batches] == [[4, 1], [3]]
    assert torch.equal(batches[0][0], features[[4, 1]])
    assert torch.equal(batches[0][1], labels[[4, 1]])


def test_iter_feature_batches_shuffle_is_deterministic_with_generator():
    features = torch.arange(16).reshape(8, 2)
    g1 = torch.Generator().manual_seed(123)
    g2 = torch.Generator().manual_seed(123)

    order1 = [
        idx.tolist()
        for _, _, idx in iter_feature_batches(features, batch_size=3, generator=g1)
    ]
    order2 = [
        idx.tolist()
        for _, _, idx in iter_feature_batches(features, batch_size=3, generator=g2)
    ]

    assert order1 == order2


def test_iter_sequence_batch_indices_rejects_invalid_options():
    offsets = torch.tensor([0, 2, 5])

    with pytest.raises(ValueError, match="batch_size must be positive"):
        list(iter_sequence_batch_indices(offsets, batch_size=0))

    with pytest.raises(ValueError, match="max_padded_tokens must be positive"):
        list(iter_sequence_batch_indices(offsets, max_padded_tokens=0))


def test_iter_sequence_batch_indices_respects_max_padded_tokens():
    offsets = torch.tensor([0, 5, 8, 10])

    batches = list(
        iter_sequence_batch_indices(
            offsets,
            batch_size=3,
            max_padded_tokens=6,
            sort_by_length=True,
            shuffle=False,
        )
    )

    assert batches == [[0], [1, 2]]


def test_iter_sequence_batch_indices_yields_single_long_sequence_over_budget():
    offsets = torch.tensor([0, 10, 12, 14])

    batches = list(
        iter_sequence_batch_indices(
            offsets,
            batch_size=3,
            max_padded_tokens=6,
            sort_by_length=True,
            shuffle=False,
        )
    )

    assert batches[0] == [0]
    assert batches[1] == [1, 2]


def test_iter_sequence_batches_pads_to_local_max():
    data = torch.arange(10, dtype=torch.float32).reshape(5, 2)
    offsets = torch.tensor([0, 3, 5])
    detection_mask = torch.tensor([True, False, True, True, True])
    labels = torch.tensor([1.0, 0.0])

    batch_seq, batch_mask, batch_labels, batch_idx = next(
        iter_sequence_batches(
            data,
            offsets,
            detection_mask,
            labels,
            batch_size=2,
            sort_by_length=False,
            shuffle=False,
        )
    )

    assert batch_seq.shape == (2, 3, 2)
    assert batch_mask.tolist() == [[True, False, True], [True, True, False]]
    assert torch.equal(batch_labels, labels)
    assert batch_idx.tolist() == [0, 1]


def test_pad_sequence_batch_preserves_layer_axis_for_3d_flat_data():
    data = torch.arange(10, dtype=torch.float32).reshape(5, 2, 1)
    offsets = torch.tensor([0, 2, 5])
    detection_mask = torch.tensor([True, False, True, True, False])

    batch_seq, batch_mask = pad_sequence_batch(data, offsets, detection_mask, [1, 0])

    assert batch_seq.shape == (2, 2, 3, 1)
    assert batch_mask.tolist() == [[True, True, False], [True, False, False]]
    torch.testing.assert_close(batch_seq[0, :, :3], data[2:5].transpose(0, 1))
    torch.testing.assert_close(batch_seq[1, :, :2], data[0:2].transpose(0, 1))
