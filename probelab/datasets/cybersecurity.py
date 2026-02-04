"""
Cybersecurity and defensive security conversation datasets.

These datasets provide security-focused instruction tuning data,
useful for training probes to detect cybersecurity domain content.
"""

from typing import Any

from datasets import load_dataset

from ..types import Dialogue, DialogueDataType, Label, Message
from .base import DialogueDataset
from .builders import sample_hf_dataset
from .hf_dataset import DatasetSpec, HFDataset
from .registry import register


@register("cybersecurity", "Trendyol cybersecurity")
class TrendyolCybersecurityDataset(HFDataset):
    """
    Trendyol Cybersecurity Instruction Tuning Dataset.

    Contains 53K meticulously curated instruction examples covering 200+
    specialized cybersecurity domains including cloud security, threat
    intelligence, incident response, and defensive operations.

    Source: https://huggingface.co/datasets/Trendyol/Trendyol-Cybersecurity-Instruction-Tuning-Dataset

    Fields:
        - system: System prompt for security assistant role
        - user: Security-related user query
        - assistant: Detailed defensive security response
    """

    base_name = "trendyol_cybersecurity"
    spec = DatasetSpec(
        hf_path="Trendyol/Trendyol-Cybersecurity-Instruction-Tuning-Dataset",
        shape="fields",
        system_fields=("system",),
        user_fields=("user",),
        assistant_fields=("assistant",),
    )


@register("cybersecurity", "Cybersecurity DPO")
class CybersecurityDPODataset(DialogueDataset):
    """
    Code Vulnerability Security DPO Dataset.

    Synthetic DPO pairs focusing on secure vs insecure code across
    multiple programming languages.

    Source: https://huggingface.co/datasets/CyberNative/Code_Vulnerability_Security_DPO

    Labels:
        - NEGATIVE: Secure code examples (chosen)
        - POSITIVE: Insecure/vulnerable code examples (rejected)

    Note: This dataset creates two dialogues per item (one for chosen,
    one for rejected) unless include_rejected=False is specified.
    """

    base_name = "cybersecurity_dpo"

    def _get_dialogues(self, **kwargs) -> DialogueDataType:
        max_samples = kwargs.get("max_samples")
        include_rejected = kwargs.get("include_rejected", True)

        dataset = load_dataset("CyberNative/Code_Vulnerability_Security_DPO")
        split = dataset["train"]

        split = sample_hf_dataset(split, max_samples)

        dialogues: list[Dialogue] = []
        labels: list[Label] = []
        metadata: dict[str, list[Any]] = {
            "is_secure": [],
        }

        for item in split:
            prompt = item.get("prompt", item.get("instruction", ""))
            chosen = item.get("chosen", "")
            rejected = item.get("rejected", "")

            # Add secure (chosen) example
            if prompt and chosen:
                dialogue: Dialogue = [
                    Message(role="user", content=prompt),
                    Message(role="assistant", content=chosen),
                ]
                dialogues.append(dialogue)
                labels.append(Label.NEGATIVE)  # Secure = NEGATIVE
                metadata["is_secure"].append(True)

            # Optionally add insecure (rejected) example
            if include_rejected and prompt and rejected:
                dialogue = [
                    Message(role="user", content=prompt),
                    Message(role="assistant", content=rejected),
                ]
                dialogues.append(dialogue)
                labels.append(Label.POSITIVE)  # Insecure = POSITIVE
                metadata["is_secure"].append(False)

        return dialogues, labels, metadata


@register("cybersecurity", "Defensive cybersecurity")
class DefensiveCybersecurityDataset(HFDataset):
    """
    Defensive Cybersecurity Dataset V1.

    Contains 2,500 high-quality instruction-response pairs focused on
    defensive cybersecurity education.

    Source: https://huggingface.co/datasets/AlicanKiraz0/Cybersecurity-Dataset-v1
    """

    base_name = "defensive_cybersecurity"
    spec = DatasetSpec(
        hf_path="AlicanKiraz0/Cybersecurity-Dataset-v1",
        shape="fields",
        user_fields=("instruction", "input", "question"),
        assistant_fields=("response", "output", "answer"),
    )
