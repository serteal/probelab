"""
Multilingual conversation datasets.

These datasets provide conversations in multiple languages, useful for
training probes to detect language or analyze cross-lingual patterns.
"""

from .hf_dataset import DatasetSpec, HFDataset
from .registry import register


@register("multilingual", "WildChat multilingual conversations")
class WildChatDataset(HFDataset):
    """
    WildChat-1M: Real conversations with ChatGPT from 210K unique users.

    Contains 1M+ conversations in 68 languages with rich metadata including
    language, country, model used, and toxicity flags.

    Source: https://huggingface.co/datasets/allenai/WildChat-1M

    Metadata fields:
        - language: Detected language of the conversation
        - model: GPT model used (e.g., "gpt-4", "gpt-3.5-turbo")
        - country: User's country
        - toxic: Whether conversation was flagged as toxic
        - turn: Number of conversation turns
    """

    base_name = "wildchat"
    spec = DatasetSpec(
        hf_path="allenai/WildChat-1M",
        shape="messages",
        messages_field="conversation",
        metadata_fields={
            "language": ("language",),
            "model": ("model",),
            "country": ("country",),
            "toxic": ("toxic",),
            "turn": ("turn",),
        },
    )


@register("multilingual", "Multilingual thinking traces")
class MultilingualThinkingDataset(HFDataset):
    """
    Multilingual reasoning dataset with chain-of-thought in multiple languages.

    Contains reasoning traces translated from English into Spanish, French,
    Italian, and German.

    Source: https://huggingface.co/datasets/HuggingFaceH4/Multilingual-Thinking

    Metadata fields:
        - language: Target language of the reasoning trace
    """

    base_name = "multilingual_thinking"
    spec = DatasetSpec(
        hf_path="HuggingFaceH4/Multilingual-Thinking",
        shape="messages",
        messages_field="messages",
        metadata_fields={
            "language": ("language",),
        },
    )


@register("multilingual", "Palo multilingual dataset")
class PaloMultilingualDataset(HFDataset):
    """
    PALO multilingual vision-language conversation dataset.

    Contains conversations in English, Chinese, French, Spanish, Russian,
    Japanese, Arabic, Hindi, Bengali, and Urdu.

    Source: https://huggingface.co/datasets/MBZUAI/palo_multilingual_dataset

    Metadata fields:
        - language: Language of the conversation
    """

    base_name = "palo_multilingual"
    spec = DatasetSpec(
        hf_path="MBZUAI/palo_multilingual_dataset",
        shape="messages",
        messages_field="conversations",
        role_field="from",
        content_field="value",
        metadata_fields={
            "language": ("language",),
        },
    )
