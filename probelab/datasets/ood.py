"""
General conversation datasets (out-of-distribution / baseline).

These datasets provide general-purpose conversations useful as baselines
or for training probes to detect specific domains vs general conversation.
"""

from typing import Any

from datasets import load_dataset

from ..types import Dialogue, DialogueDataType, Label, Message
from .base import DialogueDataset
from .builders import sample_hf_dataset
from .registry import register


@register("ood", "UltraChat conversations")
class UltraChatDataset(DialogueDataset):
    """
    UltraChat dataset from causal-lm.

    Single-turn instruction-response pairs.

    Source: https://huggingface.co/datasets/causal-lm/ultrachat
    """

    base_name = "ultrachat"

    def _get_dialogues(self, **kwargs) -> DialogueDataType:
        dataset = load_dataset("causal-lm/ultrachat")
        split = sample_hf_dataset(dataset["train"], kwargs.get("max_samples"))

        dialogues = []
        labels = []
        for element in split:
            dialogues.append(
                [
                    Message(
                        content=element["instruction"],
                        role="user",
                    ),
                    Message(
                        content=element["output"],
                        role="assistant",
                    ),
                ]
            )
            labels.append(Label.NEGATIVE)

        return dialogues, labels, None


@register("ood", "UltraChat 200k subset")
class UltraChat200kDataset(DialogueDataset):
    """
    UltraChat 200k: High-quality multi-turn conversations from HuggingFaceH4.

    Contains 200K+ cleaned and filtered conversations used for training Zephyr.

    Source: https://huggingface.co/datasets/HuggingFaceH4/ultrachat_200k

    Splits available: train_sft, test_sft, train_gen, test_gen
    """

    base_name = "ultrachat_200k"

    def _get_dialogues(self, **kwargs) -> DialogueDataType:
        split_name = kwargs.get("split", "train_sft")
        dataset = load_dataset("HuggingFaceH4/ultrachat_200k", split=split_name)
        dataset = sample_hf_dataset(dataset, kwargs.get("max_samples"))

        dialogues: list[Dialogue] = []
        labels: list[Label] = []

        for item in dataset:
            messages = item.get("messages", [])
            dialogue: Dialogue = []

            for msg in messages:
                role = msg.get("role", "user")
                content = msg.get("content", "")
                if role in ("user", "assistant", "system"):
                    dialogue.append(Message(role=role, content=content))

            if len(dialogue) > 0:
                dialogues.append(dialogue)
                labels.append(Label.NEGATIVE)

        return dialogues, labels, None


@register("ood", "LMSYS chat conversations")
class LmsysChatDataset(DialogueDataset):
    """
    LMSYS-Chat-1M: Real conversations with 25 different LLMs.

    Contains 1M conversations from Vicuna demo and Chatbot Arena,
    with metadata on model used and language.

    Source: https://huggingface.co/datasets/lmsys/lmsys-chat-1m

    Note: Requires accepting license agreement on HuggingFace.

    Metadata fields:
        - model: The LLM model used in the conversation
        - language: Detected language
        - redacted: Whether PII was redacted
    """

    base_name = "lmsys_chat"

    def _get_dialogues(self, **kwargs) -> DialogueDataType:
        dataset = load_dataset("lmsys/lmsys-chat-1m")
        split = sample_hf_dataset(dataset["train"], kwargs.get("max_samples"))

        dialogues: list[Dialogue] = []
        labels: list[Label] = []
        metadata: dict[str, list[Any]] = {
            "model": [],
            "language": [],
            "redacted": [],
            "conversation_id": [],
        }

        for item in split:
            # LMSYS uses OpenAI API format: list of {"role": ..., "content": ...}
            conversation = item.get("conversation", [])
            dialogue: Dialogue = []

            for msg in conversation:
                role = msg.get("role", "user")
                content = msg.get("content", "")
                if role in ("user", "assistant", "system"):
                    dialogue.append(Message(role=role, content=content))

            if len(dialogue) > 0:
                dialogues.append(dialogue)
                labels.append(Label.NEGATIVE)

                metadata["model"].append(item.get("model"))
                metadata["language"].append(item.get("language"))
                metadata["redacted"].append(item.get("redacted", False))
                metadata["conversation_id"].append(item.get("conversation_id"))

        return dialogues, labels, metadata


@register("ood", "ShareGPT conversations")
class ShareGPTDataset(DialogueDataset):
    """
    ShareGPT52K: Early ChatGPT conversations scraped from ShareGPT.

    Contains ~52K conversations from the ShareGPT API before it was shut down.

    Source: https://huggingface.co/datasets/RyokoAI/ShareGPT52K
    """

    base_name = "sharegpt"

    def _get_dialogues(self, **kwargs) -> DialogueDataType:
        dataset = load_dataset("RyokoAI/ShareGPT52K")
        split = sample_hf_dataset(dataset["train"], kwargs.get("max_samples"))

        dialogues: list[Dialogue] = []
        labels: list[Label] = []
        metadata: dict[str, list[Any]] = {
            "conversation_id": [],
        }

        for item in split:
            # ShareGPT uses "from" field with values "human" or "gpt"
            conversations = item.get("conversations", [])
            dialogue: Dialogue = []

            role_map = {"human": "user", "gpt": "assistant"}

            for msg in conversations:
                from_field = msg.get("from", "human")
                role = role_map.get(from_field, "user")
                content = msg.get("value", "") or msg.get("text", "") or ""

                if content:
                    dialogue.append(Message(role=role, content=content))

            if len(dialogue) > 0:
                dialogues.append(dialogue)
                labels.append(Label.NEGATIVE)
                metadata["conversation_id"].append(item.get("id"))

        return dialogues, labels, metadata


@register("ood", "Chatbot Arena conversations")
class ChatbotArenaDataset(DialogueDataset):
    """
    Chatbot Arena Conversations: Pairwise human preferences on LLM responses.

    Contains 33K conversations with human preference judgments.

    Source: https://huggingface.co/datasets/lmsys/chatbot_arena_conversations

    Metadata fields:
        - model_a: First model in comparison
        - model_b: Second model in comparison
        - winner: Which model won the comparison
        - language: Detected language
    """

    base_name = "chatbot_arena"

    def _get_dialogues(self, **kwargs) -> DialogueDataType:
        dataset = load_dataset("lmsys/chatbot_arena_conversations")
        split = sample_hf_dataset(dataset["train"], kwargs.get("max_samples"))

        dialogues: list[Dialogue] = []
        labels: list[Label] = []
        metadata: dict[str, list[Any]] = {
            "model_a": [],
            "model_b": [],
            "winner": [],
            "language": [],
        }

        for item in split:
            # Use conversation_a by default (could also use conversation_b)
            conversation = item.get("conversation_a", [])
            dialogue: Dialogue = []

            for msg in conversation:
                role = msg.get("role", "user")
                content = msg.get("content", "")
                if role in ("user", "assistant", "system"):
                    dialogue.append(Message(role=role, content=content))

            if len(dialogue) > 0:
                dialogues.append(dialogue)
                labels.append(Label.NEGATIVE)

                metadata["model_a"].append(item.get("model_a"))
                metadata["model_b"].append(item.get("model_b"))
                metadata["winner"].append(item.get("winner"))
                metadata["language"].append(item.get("language"))

        return dialogues, labels, metadata


@register("ood", "Alpaca instruction dataset")
class AlpacaDataset(DialogueDataset):
    base_name = "alpaca"

    def _get_dialogues(self, **kwargs) -> DialogueDataType:
        dataset = load_dataset("yahma/alpaca-cleaned")
        split = sample_hf_dataset(dataset["train"], kwargs.get("max_samples"))

        dialogues = []
        labels = []
        for element in split:
            dialogues.append(
                [
                    Message(
                        content=element["instruction"],
                        role="user",
                    ),
                    Message(
                        content=element["output"],
                        role="assistant",
                    ),
                ]
            )
            labels.append(Label.NEGATIVE)

        return dialogues, labels, None


@register("ood", "Generic Spanish conversations")
class GenericSpanishDataset(DialogueDataset):
    base_name = "generic_spanish"

    def _get_dialogues(self, **kwargs) -> DialogueDataType:
        dataset = load_dataset("FreedomIntelligence/evol-instruct-spanish")
        split = sample_hf_dataset(dataset["train"], kwargs.get("max_samples"))

        dialogues = []
        labels = []
        for element in split:
            dialogues.append(
                [
                    Message(
                        content=element["conversations"][0]["value"],
                        role="user",
                    ),
                    Message(
                        content=element["conversations"][1]["value"],
                        role="assistant",
                    ),
                ]
            )
            labels.append(Label.NEGATIVE)

        return dialogues, labels, None


@register("ood", "Math instruction dataset")
class MathInstructDataset(DialogueDataset):
    base_name = "math_instruct"

    def _get_dialogues(self, **kwargs) -> DialogueDataType:
        dataset = load_dataset("TIGER-Lab/MathInstruct")
        split = sample_hf_dataset(dataset["train"], kwargs.get("max_samples"))

        dialogues = []
        labels = []
        for element in split:
            dialogues.append(
                [
                    Message(
                        content=element["instruction"],
                        role="user",
                    ),
                    Message(
                        content=element["output"],
                        role="assistant",
                    ),
                ]
            )
            labels.append(Label.NEGATIVE)

        return dialogues, labels, None
