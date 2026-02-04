"""
Legal and finance domain datasets.

These datasets provide legal documents, financial text, and domain-specific
conversations, useful for training probes to detect legal/financial content.
"""

from typing import Any, ClassVar

from datasets import load_dataset

from ..types import Dialogue, DialogueDataType, Label, Message
from .base import DialogueDataset
from .builders import sample_hf_dataset
from .hf_dataset import DatasetSpec, HFDataset
from .registry import register


@register("legal_finance", "Case law documents")
class CaselawDataset(HFDataset):
    """
    HFforLegal Case Law dataset.

    Comprehensive collection of legal decisions from various countries
    in a standardized format.

    Source: https://huggingface.co/datasets/HFforLegal/case-law

    Metadata fields:
        - country: Country code (ISO 3166-1 alpha-2)
        - court: Court or tribunal name
    """

    base_name = "caselaw"
    spec = DatasetSpec(
        hf_path="HFforLegal/case-law",
        hf_config="US",  # Default to US cases
        shape="text",
        text_field="text",
        text_as_assistant=True,
        default_max_samples=10000,
        metadata_fields={
            "court": ("court", "tribunal"),
        },
    )


@register("legal_finance", "Finance tasks")
class FinanceTasksDataset(HFDataset):
    """
    AdaptLLM Finance Tasks dataset.

    Finance-specific instruction tasks for domain adaptation.

    Source: https://huggingface.co/datasets/AdaptLLM/finance-tasks
    """

    base_name = "finance_tasks"
    spec = DatasetSpec(
        hf_path="AdaptLLM/finance-tasks",
        shape="fields",
        user_fields=("instruction", "input"),
        assistant_fields=("output", "response"),
    )


@register("legal_finance", "Financial phrasebank")
class FinancialPhrasebankDataset(DialogueDataset):
    """
    Financial Phrasebank: Financial news sentiment dataset.

    Contains 4,840 sentences from financial news categorized by sentiment
    (positive, negative, neutral).

    Source: https://huggingface.co/datasets/takala/financial_phrasebank

    Metadata fields:
        - sentiment: The sentiment label (positive/negative/neutral)
    """

    base_name = "financial_phrasebank"

    # Sentiment mapping from numeric labels
    SENTIMENT_MAP: ClassVar[dict[int, str]] = {
        0: "negative",
        1: "neutral",
        2: "positive",
    }

    def _get_dialogues(self, **kwargs) -> DialogueDataType:
        max_samples = kwargs.get("max_samples")
        config = kwargs.get("config", "sentences_allagree")

        dataset = load_dataset("takala/financial_phrasebank", config)
        split = dataset["train"]

        split = sample_hf_dataset(split, max_samples)

        dialogues: list[Dialogue] = []
        labels: list[Label] = []
        metadata: dict[str, list[Any]] = {
            "sentiment": [],
        }

        for item in split:
            sentence = item.get("sentence", "")
            label = item.get("label", 1)

            if sentence:
                dialogue: Dialogue = [
                    Message(role="assistant", content=sentence),
                ]
                dialogues.append(dialogue)
                labels.append(Label.NEGATIVE)  # All financial text
                metadata["sentiment"].append(self.SENTIMENT_MAP.get(label, "neutral"))

        return dialogues, labels, metadata


@register("legal_finance", "Legal advice from Reddit")
class LegalAdviceRedditDataset(HFDataset):
    """
    Legal Advice from Reddit (subset of Pile of Law).

    Contains posts and discussions from r/legaladvice subreddit.

    Source: https://huggingface.co/datasets/pile-of-law/pile-of-law (r_legaladvice subset)
    """

    base_name = "legal_advice_reddit"
    spec = DatasetSpec(
        hf_path="pile-of-law/pile-of-law",
        hf_config="r_legaladvice",
        shape="text",
        text_field="text",
        text_as_assistant=False,  # Reddit posts as user content
        default_max_samples=10000,
    )
