"""Tests for base Dataset class."""

import numpy as np

from probelab.datasets.base import Dataset
from probelab.types import Label, Message


class TestDataset:
    """Test cases for Dataset dataclass."""

    def test_dataset_initialization(self):
        """Test basic Dataset initialization."""
        dialogues = [[Message("user", "Hello"), Message("assistant", "Hi there")]]
        labels = [Label.POSITIVE]

        dataset = Dataset(dialogues, labels, "test")

        assert len(dataset) == 1
        assert dataset.dialogues == dialogues
        assert dataset.labels == labels
        assert dataset.name == "test"

    def test_dataset_getitem_int(self):
        """Test __getitem__ with integer index."""
        dialogues = [
            [Message("user", "Test 1"), Message("assistant", "Response 1")],
            [Message("user", "Test 2"), Message("assistant", "Response 2")],
        ]
        labels = [Label.POSITIVE, Label.NEGATIVE]

        dataset = Dataset(dialogues, labels, "test")

        subset = dataset[0]
        assert len(subset) == 1
        assert subset.dialogues[0] == dialogues[0]
        assert subset.labels[0] == labels[0]

    def test_dataset_getitem_slice(self):
        """Test __getitem__ with slice."""
        dialogues = [
            [Message("user", "Test 1")],
            [Message("user", "Test 2")],
            [Message("user", "Test 3")],
        ]
        labels = [Label.POSITIVE, Label.NEGATIVE, Label.POSITIVE]

        dataset = Dataset(dialogues, labels, "test")

        subset = dataset[:2]
        assert len(subset) == 2
        assert subset.dialogues == dialogues[:2]
        assert subset.labels == labels[:2]

    def test_dataset_getitem_list(self):
        """Test __getitem__ with list of indices."""
        dialogues = [
            [Message("user", f"Test {i}")] for i in range(5)
        ]
        labels = [Label.POSITIVE] * 5

        dataset = Dataset(dialogues, labels, "test")

        subset = dataset[[0, 2, 4]]
        assert len(subset) == 3
        assert subset.dialogues[0] == dialogues[0]
        assert subset.dialogues[1] == dialogues[2]
        assert subset.dialogues[2] == dialogues[4]

    def test_dataset_positive_negative_properties(self):
        """Test filtering dataset by label using properties."""
        dialogues = [[Message("user", f"Test {i}")] for i in range(5)]
        labels = [Label.POSITIVE, Label.NEGATIVE, Label.POSITIVE, Label.NEGATIVE, Label.NEGATIVE]

        dataset = Dataset(dialogues, labels, "test")

        positive = dataset.positive
        assert len(positive) == 2
        assert all(l == Label.POSITIVE for l in positive.labels)

        negative = dataset.negative
        assert len(negative) == 3
        assert all(l == Label.NEGATIVE for l in negative.labels)

    def test_dataset_where(self):
        """Test where() method for conditional filtering."""
        dialogues = [[Message("user", f"Test {i}")] for i in range(5)]
        labels = [Label.POSITIVE, Label.NEGATIVE, Label.POSITIVE, Label.NEGATIVE, Label.NEGATIVE]

        dataset = Dataset(dialogues, labels, "test")

        positive = dataset.where([l == Label.POSITIVE for l in labels])
        assert len(positive) == 2

    def test_dataset_add(self):
        """Test concatenating datasets."""
        dialogues1 = [[Message("user", "Test 1")]]
        labels1 = [Label.POSITIVE]
        dataset1 = Dataset(dialogues1, labels1, "test1")

        dialogues2 = [[Message("user", "Test 2")]]
        labels2 = [Label.NEGATIVE]
        dataset2 = Dataset(dialogues2, labels2, "test2")

        combined = dataset1 + dataset2
        assert len(combined) == 2
        assert combined.name == "test1+test2"
        assert set(d[0].content for d in combined.dialogues) == {"Test 1", "Test 2"}

    def test_dataset_shuffle(self):
        """Test shuffle returns new dataset with same seed."""
        dialogues = [[Message("user", f"Test {i}")] for i in range(10)]
        labels = [Label.POSITIVE] * 10

        dataset = Dataset(dialogues, labels, "test")

        shuffled1 = dataset.shuffle(seed=42)
        shuffled2 = dataset.shuffle(seed=42)

        assert shuffled1.dialogues == shuffled2.dialogues
        assert shuffled1.labels == shuffled2.labels

    def test_dataset_split(self):
        """Test split() method."""
        dialogues = [[Message("user", f"Test {i}")] for i in range(10)]
        labels = [Label.POSITIVE] * 10

        dataset = Dataset(dialogues, labels, "test")

        train, test = dataset.split(frac=0.8)
        assert len(train) == 8
        assert len(test) == 2

    def test_dataset_metadata(self):
        """Test dataset with metadata."""
        dialogues = [[Message("user", "Test")]]
        labels = [Label.POSITIVE]
        metadata = {"source": ["test_source"]}

        dataset = Dataset(dialogues, labels, "test", metadata)

        assert dataset.metadata == metadata
        subset = dataset[0]
        assert subset.metadata == {"source": ["test_source"]}

    def test_dataset_repr(self):
        """Test dataset string representation."""
        dialogues = [[Message("user", "Test")]]
        labels = [Label.POSITIVE]

        dataset = Dataset(dialogues, labels, "test")

        repr_str = repr(dataset)
        assert "test" in repr_str
        assert "n=1" in repr_str
        assert "pos=1" in repr_str
        assert "neg=0" in repr_str

    def test_dataset_sample_basic(self):
        """Test sample() method with basic random sampling."""
        dialogues = [[Message("user", f"Test {i}")] for i in range(100)]
        labels = [Label.POSITIVE] * 50 + [Label.NEGATIVE] * 50

        dataset = Dataset(dialogues, labels, "test")

        sampled = dataset.sample(10)
        assert len(sampled) == 10
        assert sampled.name == "test"

    def test_dataset_sample_reproducible(self):
        """Test sample() method is reproducible with same seed."""
        dialogues = [[Message("user", f"Test {i}")] for i in range(100)]
        labels = [Label.POSITIVE] * 50 + [Label.NEGATIVE] * 50

        dataset = Dataset(dialogues, labels, "test")

        sample1 = dataset.sample(10, seed=42)
        sample2 = dataset.sample(10, seed=42)

        assert sample1.dialogues == sample2.dialogues
        assert sample1.labels == sample2.labels

    def test_dataset_sample_larger_than_dataset(self):
        """Test sample() returns full dataset when n >= len(dataset)."""
        dialogues = [[Message("user", f"Test {i}")] for i in range(10)]
        labels = [Label.POSITIVE] * 10

        dataset = Dataset(dialogues, labels, "test")

        sampled = dataset.sample(100)
        assert len(sampled) == 10
        assert sampled is dataset  # Should return same object

    def test_dataset_sample_stratified(self):
        """Test stratified sampling preserves label proportions."""
        dialogues = [[Message("user", f"Test {i}")] for i in range(100)]
        labels = [Label.POSITIVE] * 80 + [Label.NEGATIVE] * 20

        dataset = Dataset(dialogues, labels, "test")

        sampled = dataset.sample(50, stratified=True, seed=42)
        assert len(sampled) == 50

        pos_count = sum(1 for l in sampled.labels if l == Label.POSITIVE)
        neg_count = sum(1 for l in sampled.labels if l == Label.NEGATIVE)

        # Should preserve 80/20 ratio (40 pos, 10 neg)
        assert pos_count == 40
        assert neg_count == 10

    def test_dataset_sample_stratified_imbalanced(self):
        """Test stratified sampling with highly imbalanced data."""
        dialogues = [[Message("user", f"Test {i}")] for i in range(100)]
        labels = [Label.POSITIVE] * 95 + [Label.NEGATIVE] * 5

        dataset = Dataset(dialogues, labels, "test")

        sampled = dataset.sample(20, stratified=True, seed=42)
        assert len(sampled) == 20

        pos_count = sum(1 for l in sampled.labels if l == Label.POSITIVE)
        neg_count = sum(1 for l in sampled.labels if l == Label.NEGATIVE)

        # Should roughly preserve proportions
        assert pos_count >= 18  # ~95% of 20
        assert neg_count <= 2   # ~5% of 20

    def test_dataset_split_stratified(self):
        """Test stratified split preserves label proportions."""
        dialogues = [[Message("user", f"Test {i}")] for i in range(100)]
        labels = [Label.POSITIVE] * 60 + [Label.NEGATIVE] * 40

        dataset = Dataset(dialogues, labels, "test")

        train, test = dataset.split(frac=0.8, stratified=True, seed=42)

        assert len(train) == 80
        assert len(test) == 20

        # Check proportions in train set (should be 60% pos)
        train_pos = sum(1 for l in train.labels if l == Label.POSITIVE)
        train_neg = sum(1 for l in train.labels if l == Label.NEGATIVE)
        assert train_pos == 48  # 60% of 80
        assert train_neg == 32  # 40% of 80

        # Check proportions in test set
        test_pos = sum(1 for l in test.labels if l == Label.POSITIVE)
        test_neg = sum(1 for l in test.labels if l == Label.NEGATIVE)
        assert test_pos == 12  # 60% of 20
        assert test_neg == 8   # 40% of 20

    def test_dataset_sample_with_metadata(self):
        """Test sample() preserves metadata correctly."""
        dialogues = [[Message("user", f"Test {i}")] for i in range(100)]
        labels = [Label.POSITIVE] * 50 + [Label.NEGATIVE] * 50
        metadata = {"source": [f"src_{i}" for i in range(100)]}

        dataset = Dataset(dialogues, labels, "test", metadata)

        sampled = dataset.sample(10, seed=42)
        assert len(sampled) == 10
        assert sampled.metadata is not None
        assert len(sampled.metadata["source"]) == 10

    def test_add_creates_source_metadata(self):
        """Test __add__ creates source metadata from dataset names."""
        ds1 = Dataset([[Message("user", "1")]], [Label.POSITIVE], "truthful_qa")
        ds2 = Dataset([[Message("user", "2")]], [Label.NEGATIVE], "circuit_breakers")

        combined = ds1 + ds2
        assert combined.metadata["source"] == ["truthful_qa", "circuit_breakers"]

    def test_add_preserves_existing_source(self):
        """Test __add__ preserves existing source metadata."""
        ds1 = Dataset([[Message("user", "1")], [Message("user", "2")]], [Label.POSITIVE] * 2, "mix",
                      metadata={"source": ["dataset_a", "dataset_b"]})
        ds2 = Dataset([[Message("user", "3")]], [Label.NEGATIVE], "circuit_breakers")

        combined = ds1 + ds2
        assert combined.metadata["source"] == ["dataset_a", "dataset_b", "circuit_breakers"]

    def test_add_merges_different_keys(self):
        """Test __add__ merges metadata with different keys."""
        ds1 = Dataset([[Message("user", "1")]], [Label.POSITIVE], "ds1", metadata={"difficulty": ["easy"]})
        ds2 = Dataset([[Message("user", "2")]], [Label.NEGATIVE], "ds2", metadata={"category": ["safety"]})

        combined = ds1 + ds2
        assert "source" in combined.metadata
        assert "difficulty" in combined.metadata
        assert "category" in combined.metadata
        assert combined.metadata["difficulty"] == ["easy", None]
        assert combined.metadata["category"] == [None, "safety"]

    def test_add_fills_missing_with_none(self):
        """Test __add__ fills missing metadata keys with None."""
        ds1 = Dataset([[Message("user", "1")]], [Label.POSITIVE], "ds1", metadata={"key1": ["val1"]})
        ds2 = Dataset([[Message("user", "2")]], [Label.NEGATIVE], "ds2")

        combined = ds1 + ds2
        assert combined.metadata["key1"] == ["val1", None]
        assert combined.metadata["source"] == ["ds1", "ds2"]

    def test_add_chained(self):
        """Test chaining multiple __add__ operations."""
        ds1 = Dataset([[Message("user", "1")]], [Label.POSITIVE], "ds1")
        ds2 = Dataset([[Message("user", "2")]], [Label.NEGATIVE], "ds2")
        ds3 = Dataset([[Message("user", "3")]], [Label.POSITIVE], "ds3")

        combined = ds1 + ds2 + ds3
        assert len(combined) == 3
        assert combined.metadata["source"] == ["ds1", "ds2", "ds3"]
