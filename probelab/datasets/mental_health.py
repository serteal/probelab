"""
Mental health and counseling conversation datasets.

These datasets provide therapy, counseling, and emotional support conversations,
useful for training probes to detect mental health domain content.
"""

from ..types import Label
from .hf_dataset import DatasetSpec, HFDataset
from .registry import register


def _prosocial_label(item: dict) -> Label:
    """Label function for ProsocialDialog - casual is NEGATIVE, others POSITIVE."""
    safety_label = item.get("safety_label", "__casual__")
    return Label.NEGATIVE if safety_label == "__casual__" else Label.POSITIVE


@register("mental_health", "Mental health chat")
class MentalChatDataset(HFDataset):
    """
    MentalChat16K: Synthetic counselor-client conversations.

    Contains 16K+ conversations covering 33 mental health topics including
    anxiety, depression, relationships, and family conflict.

    Source: https://huggingface.co/datasets/ShenLab/MentalChat16K

    Fields:
        - instruction: System prompt for counselor role
        - input: Patient's mental health question/concern
        - output: Counselor's response
    """

    base_name = "mentalchat"
    spec = DatasetSpec(
        hf_path="ShenLab/MentalChat16K",
        shape="fields",
        system_fields=("instruction",),
        user_fields=("input",),
        assistant_fields=("output",),
    )


@register("mental_health", "Mental health counseling")
class MentalHealthCounselingDataset(HFDataset):
    """
    Mental Health Counseling Conversations dataset.

    Source: https://huggingface.co/datasets/Amod/mental_health_counseling_conversations
    """

    base_name = "mental_health_counseling"
    spec = DatasetSpec(
        hf_path="Amod/mental_health_counseling_conversations",
        shape="fields",
        user_fields=("Context", "context", "input"),
        assistant_fields=("Response", "response", "output"),
    )


@register("mental_health", "Prosocial dialogue")
class ProsocialDialogDataset(HFDataset):
    """
    ProsocialDialog: Teaching agents to respond prosocially to problematic content.

    Contains 58K dialogues with safety annotations. Includes both problematic
    user utterances and prosocial assistant responses grounded in social norms.

    Source: https://huggingface.co/datasets/allenai/prosocial-dialog

    Labels:
        - NEGATIVE: Casual/acceptable content (safety_label == "__casual__")
        - POSITIVE: Content that needs caution or intervention

    Metadata fields:
        - safety_label: Original safety classification
        - rots: Rules-of-thumb (social norms) guiding the response
        - source: Origin of the seed text
    """

    base_name = "prosocial_dialog"
    spec = DatasetSpec(
        hf_path="allenai/prosocial-dialog",
        shape="fields",
        user_fields=("context",),
        assistant_fields=("response",),
        metadata_fields={
            "safety_label": ("safety_label",),
            "rots": ("rots",),
            "source": ("source",),
        },
        label_fn=_prosocial_label,
    )


@register("mental_health", "Emotional support conversations")
class EmotionalSupportDataset(HFDataset):
    """
    Mental Health Chatbot Dataset for emotional support conversations.

    Curated from healthcare blogs like WebMD, Mayo Clinic, and HealthLine.

    Source: https://huggingface.co/datasets/heliosbrahma/mental_health_chatbot_dataset
    """

    base_name = "emotional_support"
    spec = DatasetSpec(
        hf_path="heliosbrahma/mental_health_chatbot_dataset",
        shape="fields",
        user_fields=("question", "input", "text", "context"),
        assistant_fields=("answer", "output", "response", "reply"),
    )
