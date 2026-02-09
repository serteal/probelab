"""Tests for hallucination detection datasets."""

from unittest.mock import patch

from probelab.datasets.hallucination import (
    _convert_longform,
    _convert_triviaqa,
    _has_hallucination,
)
from probelab.datasets.registry import Topic, list_datasets
from probelab.types import Label


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_longform_item(
    prompt: str = "What is Python?",
    completion: str = "Python is a programming language. It was created by Guido.",
    annotations: list | None = None,
):
    """Create a mock HF dataset item in longform annotation format."""
    if annotations is None:
        annotations = [
            {"span": "Python is a programming language.", "label": "Supported", "index": 0},
            {"span": "It was created by Guido.", "label": "Not Supported", "index": 35},
        ]
    return {
        "conversation": [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": completion},
        ],
        "annotations": annotations,
        "model": "test-model",
    }


def _make_triviaqa_item(
    question: str = "Who invented Python?",
    completion: str = "Python was invented by Guido van Rossum.",
    exact_answer: str = "Guido van Rossum",
    label: str = "S",
):
    """Create a mock HF dataset item in TriviaQA format."""
    return {
        "question": question,
        "gt_completion": completion,
        "exact_answer": exact_answer,
        "llm_judge_label": label,
    }


class _MockHFDataset:
    """Minimal mock HuggingFace dataset."""

    def __init__(self, items: list[dict]):
        self._items = items
        self.features = {k: None for k in items[0]} if items else {}

    def __iter__(self):
        return iter(self._items)

    def __len__(self):
        return len(self._items)

    def __getitem__(self, idx):
        return self._items[idx]


# ---------------------------------------------------------------------------
# _has_hallucination
# ---------------------------------------------------------------------------

class TestHasHallucination:
    def test_with_hallucinated_span(self):
        spans = [{"text": "foo", "label": 1.0, "index": 0}]
        assert _has_hallucination(spans) is True

    def test_without_hallucinated_span(self):
        spans = [{"text": "foo", "label": 0.0, "index": 0}]
        assert _has_hallucination(spans) is False

    def test_empty_spans(self):
        assert _has_hallucination([]) is False

    def test_mixed_spans(self):
        spans = [
            {"text": "a", "label": 0.0, "index": 0},
            {"text": "b", "label": 1.0, "index": 5},
        ]
        assert _has_hallucination(spans) is True


# ---------------------------------------------------------------------------
# Longform conversion
# ---------------------------------------------------------------------------

class TestConvertLongform:
    def test_basic_conversion(self):
        items = [_make_longform_item()]
        ds = _convert_longform(_MockHFDataset(items), "test_longform")

        assert len(ds) == 1
        assert ds.labels[0] == Label.POSITIVE  # Has a "Not Supported" span
        assert ds.name == "test_longform"

    def test_dialogue_structure(self):
        items = [_make_longform_item()]
        ds = _convert_longform(_MockHFDataset(items), "test")

        dlg = ds.dialogues[0]
        assert len(dlg) == 2
        assert dlg[0].role == "user"
        assert dlg[1].role == "assistant"
        assert dlg[0].content == "What is Python?"

    def test_all_supported_is_negative(self):
        items = [_make_longform_item(
            annotations=[{"span": "Python is great.", "label": "Supported", "index": 0}],
            completion="Python is great.",
        )]
        ds = _convert_longform(_MockHFDataset(items), "test")
        assert ds.labels[0] == Label.NEGATIVE

    def test_metadata_contains_spans(self):
        items = [_make_longform_item()]
        ds = _convert_longform(_MockHFDataset(items), "test")

        assert ds.metadata is not None
        assert "spans" in ds.metadata
        assert len(ds.metadata["spans"][0]) == 2
        assert ds.metadata["spans"][0][0]["label"] == 0.0
        assert ds.metadata["spans"][0][1]["label"] == 1.0

    def test_metadata_contains_subset(self):
        items = [_make_longform_item()]
        ds = _convert_longform(_MockHFDataset(items), "test")

        assert "subset" in ds.metadata
        assert ds.metadata["subset"][0] == "test-model"

    def test_invalid_annotations_skipped(self):
        items = [_make_longform_item(annotations=[
            None,
            {"span": "ok", "label": "Supported", "index": None},
            {"span": "", "label": "Supported", "index": 0},
            {"span": "not in completion", "label": "Not Supported", "index": 0},
            {"span": "Python is a programming language.", "label": "Supported", "index": 0},
        ])]
        ds = _convert_longform(_MockHFDataset(items), "test")

        # Only the last valid annotation should survive
        assert len(ds.metadata["spans"][0]) == 1

    def test_multiple_items(self):
        items = [
            _make_longform_item(
                completion="All good.",
                annotations=[{"span": "All good.", "label": "Supported", "index": 0}],
            ),
            _make_longform_item(
                completion="Bad info.",
                annotations=[{"span": "Bad info.", "label": "Not Supported", "index": 0}],
            ),
        ]
        ds = _convert_longform(_MockHFDataset(items), "test")

        assert len(ds) == 2
        assert ds.labels[0] == Label.NEGATIVE
        assert ds.labels[1] == Label.POSITIVE


# ---------------------------------------------------------------------------
# TriviaQA conversion
# ---------------------------------------------------------------------------

class TestConvertTriviaQA:
    def test_supported_is_negative(self):
        items = [_make_triviaqa_item(label="S")]
        ds = _convert_triviaqa(_MockHFDataset(items), "test_trivia")

        assert len(ds) == 1
        assert ds.labels[0] == Label.NEGATIVE

    def test_not_supported_is_positive(self):
        items = [_make_triviaqa_item(label="NS")]
        ds = _convert_triviaqa(_MockHFDataset(items), "test_trivia")

        assert ds.labels[0] == Label.POSITIVE

    def test_na_is_negative(self):
        # N/A maps to -100.0 which is not 1.0, so _has_hallucination returns False
        items = [_make_triviaqa_item(label="N/A")]
        ds = _convert_triviaqa(_MockHFDataset(items), "test_trivia")

        assert ds.labels[0] == Label.NEGATIVE

    def test_invalid_label_skipped(self):
        items = [_make_triviaqa_item(label="INVALID")]
        ds = _convert_triviaqa(_MockHFDataset(items), "test_trivia")

        assert len(ds) == 0

    def test_exact_answer_not_found_skipped(self):
        items = [_make_triviaqa_item(exact_answer="nonexistent")]
        ds = _convert_triviaqa(_MockHFDataset(items), "test_trivia")

        assert len(ds) == 0

    def test_span_index_is_correct(self):
        items = [_make_triviaqa_item(
            completion="Python was invented by Guido van Rossum.",
            exact_answer="Guido van Rossum",
        )]
        ds = _convert_triviaqa(_MockHFDataset(items), "test")

        span = ds.metadata["spans"][0][0]
        assert span["text"] == "Guido van Rossum"
        assert span["index"] == 23  # .find("Guido van Rossum")

    def test_dialogue_structure(self):
        items = [_make_triviaqa_item()]
        ds = _convert_triviaqa(_MockHFDataset(items), "test")

        dlg = ds.dialogues[0]
        assert len(dlg) == 2
        assert dlg[0].role == "user"
        assert dlg[0].content == "Who invented Python?"
        assert dlg[1].role == "assistant"


# ---------------------------------------------------------------------------
# Registry integration
# ---------------------------------------------------------------------------

class TestHallucinationRegistry:
    def test_hallucination_topic_exists(self):
        assert hasattr(Topic, "HALLUCINATION")
        assert Topic.HALLUCINATION.value == "hallucination"

    def test_datasets_registered(self):
        all_datasets = list_datasets(category="hallucination")
        assert "longfact" in all_datasets
        assert "longfact_augmented" in all_datasets
        assert "triviaqa_hallucination" in all_datasets
        assert "healthbench" in all_datasets

    @patch("probelab.datasets.hallucination.load_dataset")
    def test_longfact_variant_parameter(self, mock_load):
        mock_load.return_value = _MockHFDataset([_make_longform_item()])

        from probelab.datasets.hallucination import longfact

        longfact(variant="Meta-Llama-3.1-8B-Instruct")
        mock_load.assert_called_once_with(
            "obalcells/longfact-annotations",
            "Meta-Llama-3.1-8B-Instruct",
            split="train",
        )

    @patch("probelab.datasets.hallucination.load_dataset")
    def test_longfact_no_variant(self, mock_load):
        mock_load.return_value = _MockHFDataset([_make_longform_item()])

        from probelab.datasets.hallucination import longfact

        longfact()
        mock_load.assert_called_once_with(
            "obalcells/longfact-annotations", split="train"
        )

    @patch("probelab.datasets.hallucination.load_dataset")
    def test_triviaqa_variant_parameter(self, mock_load):
        mock_load.return_value = _MockHFDataset([_make_triviaqa_item()])

        from probelab.datasets.hallucination import triviaqa_hallucination

        triviaqa_hallucination(variant="Meta-Llama-3.1-8B-Instruct")
        mock_load.assert_called_once_with(
            "obalcells/triviaqa-balanced",
            "Meta-Llama-3.1-8B-Instruct",
            split="train",
        )
