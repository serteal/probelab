"""Tests for base dataset classes."""

import numpy as np

from probelab.datasets.base import DialogueDataset
from probelab.types import Dialogue, Label, Message, DialogueDataType


class SimpleDialogueDataset(DialogueDataset):
    """Simple concrete implementation of DialogueDataset for testing."""

    def _get_dialogues(self, **kwargs) -> DialogueDataType:
        """Just return empty data - we'll use direct initialization in tests."""
        return [], [], None


class TestDialogueDataset:
    """Test cases for DialogueDataset base class."""

    def test_dialogue_dataset_initialization(self):
        """Test basic DialogueDataset initialization."""
        dialogues = [
            Dialogue(
                [
                    Message(role="user", content="Hello"),
                    Message(role="assistant", content="Hi there"),
                ]
            )
        ]
        labels = [Label.POSITIVE]

        dataset = SimpleDialogueDataset(dialogues=dialogues, labels=labels)

        assert len(dataset) == 1
        assert dataset.dialogues == dialogues
        assert dataset.labels == labels

    def test_dialogue_dataset_getitem(self):
        """Test __getitem__ functionality."""
        dialogues = [
            Dialogue(
                [
                    Message(role="user", content="Test 1"),
                    Message(role="assistant", content="Response 1"),
                ]
            ),
            Dialogue(
                [
                    Message(role="user", content="Test 2"),
                    Message(role="assistant", content="Response 2"),
                ]
            ),
        ]
        labels = [Label.POSITIVE, Label.NEGATIVE]

        dataset = SimpleDialogueDataset(
            dialogues=dialogues, labels=labels, shuffle_upon_init=False
        )

        # Test indexing - returns a new dataset with single item
        subset = dataset[0]
        assert len(subset) == 1
        assert subset.dialogues[0] == dialogues[0]
        assert subset.labels[0] == labels[0]

        # Test slicing
        subset = dataset[:1]
        assert len(subset) == 1
        assert subset.dialogues == dialogues[:1]
        assert subset.labels == labels[:1]

    def test_dialogue_dataset_filter_by_label(self):
        """Test filtering dataset by label."""
        dialogues = [
            Dialogue([Message(role="user", content=f"Test {i}")]) for i in range(5)
        ]
        labels = [
            Label.POSITIVE,
            Label.NEGATIVE,
            Label.POSITIVE,
            Label.NEGATIVE,
            Label.NEGATIVE,
        ]

        dataset = SimpleDialogueDataset(dialogues=dialogues, labels=labels)

        # Filter positive only
        positive_dataset = dataset.get_with_label(Label.POSITIVE)
        assert len(positive_dataset) == 2
        assert all(label == Label.POSITIVE for label in positive_dataset.labels)

        # Filter negative only
        negative_dataset = dataset.get_with_label(Label.NEGATIVE)
        assert len(negative_dataset) == 3
        assert all(label == Label.NEGATIVE for label in negative_dataset.labels)

    def test_dialogue_dataset_add(self):
        """Test concatenating datasets."""
        dialogues1 = [Dialogue([Message(role="user", content="Test 1")])]
        labels1 = [Label.POSITIVE]
        dataset1 = SimpleDialogueDataset(
            dialogues=dialogues1, labels=labels1, shuffle_upon_init=False
        )

        dialogues2 = [Dialogue([Message(role="user", content="Test 2")])]
        labels2 = [Label.NEGATIVE]
        dataset2 = SimpleDialogueDataset(
            dialogues=dialogues2, labels=labels2, shuffle_upon_init=False
        )

        # Test concatenation
        combined = dataset1 + dataset2
        assert len(combined) == 2
        # Due to shuffling, we can't guarantee order, but check contents
        assert set(d[0].content for d in combined.dialogues) == {"Test 1", "Test 2"}
        assert sorted(combined.labels) == sorted(labels1 + labels2)

    def test_dataset_shuffling_does_not_touch_global_rng(self):
        """Shuffling should not reset NumPy's global RNG state."""
        dialogues = [
            Dialogue([Message(role="user", content=f"Test {i}")]) for i in range(5)
        ]
        labels = [Label.POSITIVE] * 5

        np.random.seed(123)
        _ = np.random.random()  # Advance RNG once

        SimpleDialogueDataset(dialogues=dialogues, labels=labels)

        second_draw = np.random.random()

        np.random.seed(123)
        _ = np.random.random()  # Skip first draw
        expected_second = np.random.random()

        assert second_draw == expected_second

    def test_dataset_shuffle_seed_reproducible(self):
        """Providing a shuffle seed should produce reproducible order."""
        dialogues = [
            Dialogue([Message(role="user", content=f"Test {i}")]) for i in range(5)
        ]
        labels = [Label.POSITIVE] * 5

        dataset_a = SimpleDialogueDataset(
            dialogues=dialogues,
            labels=labels,
            shuffle_seed=7,
        )
        dataset_b = SimpleDialogueDataset(
            dialogues=dialogues,
            labels=labels,
            shuffle_seed=7,
        )

        assert dataset_a.labels == dataset_b.labels
        assert dataset_a.dialogues == dataset_b.dialogues

    def test_dialogue_dataset_metadata(self):
        """Test dataset with metadata."""
        dialogues = [Dialogue([Message(role="user", content="Test")])]
        labels = [Label.POSITIVE]
        metadata = {"source": "test", "version": "1.0"}

        dataset = SimpleDialogueDataset(
            dialogues=dialogues, labels=labels, metadata=metadata
        )

        assert dataset.metadata == metadata

    def test_dialogue_dataset_validation(self):
        """Test validation of dialogues and labels."""
        dialogues = [Dialogue([Message(role="user", content="Test")])]
        labels = [Label.POSITIVE, Label.NEGATIVE]  # Mismatched length

        # The base class doesn't validate length mismatch, but it will cause issues
        # when using the dataset. Let's create it and check the lengths are mismatched
        dataset = SimpleDialogueDataset(
            dialogues=dialogues, labels=labels, shuffle_upon_init=False
        )
        assert len(dataset.dialogues) != len(dataset.labels)
