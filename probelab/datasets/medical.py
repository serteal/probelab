"""
Medical and healthcare conversation datasets.

These datasets provide doctor-patient conversations and medical dialogues,
useful for training probes to detect medical domain content.
"""

from ..types import Dialogue, Message
from .hf_dataset import DatasetSpec, HFDataset


def _build_meddialog(item: dict) -> Dialogue:
    """Builder for MedDialog format - text contains full conversation."""
    text = item.get("text", "")
    if text:
        return [
            Message(role="user", content="Medical consultation:"),
            Message(role="assistant", content=text),
        ]
    return []


def _build_clinical_notes(item: dict) -> Dialogue:
    """Builder for clinical notes - handles both string and list formats."""
    conversation = item.get("conversation", item.get("dialogue", ""))

    if not conversation:
        return []

    if isinstance(conversation, str):
        return [Message(role="assistant", content=conversation)]

    if isinstance(conversation, list):
        dialogue: Dialogue = []
        for msg in conversation:
            if isinstance(msg, dict):
                role = msg.get("role", "assistant")
                content = msg.get("content", "")
                if content:
                    dialogue.append(Message(role=role, content=content))
            elif isinstance(msg, str):
                dialogue.append(Message(role="assistant", content=msg))
        return dialogue

    return []


class MedDialogDataset(HFDataset):
    """
    MedDialog: Doctor-patient conversations in English.

    Contains 260K+ dialogues between doctors and patients covering
    various medical topics and conditions.

    Source: https://huggingface.co/datasets/bigbio/meddialog

    Note: Dataset structure may vary - check loading script for details.
    """

    base_name = "meddialog"
    spec = DatasetSpec(
        hf_path="bigbio/meddialog",
        hf_config="meddialog_en_bigbio_text",
        shape="custom",
        builder_fn=_build_meddialog,
    )


class MedicalDialogueSOAPDataset(HFDataset):
    """
    Medical Dialogue to SOAP Summary dataset.

    Contains 10K synthetic doctor-patient dialogues with SOAP summaries.

    Source: https://huggingface.co/datasets/omi-health/medical-dialogue-to-soap-summary

    Metadata fields:
        - soap_summary: The SOAP summary of the dialogue
    """

    base_name = "medical_soap"
    spec = DatasetSpec(
        hf_path="omi-health/medical-dialogue-to-soap-summary",
        shape="fields",
        user_fields=("dialogue",),
        assistant_fields=(),
        metadata_fields={
            "soap_summary": ("soap",),
        },
    )


class ClinicalNotesDataset(HFDataset):
    """
    Augmented Clinical Notes with synthetic doctor-patient conversations.

    Based on PMC-Patients with 167K patient summaries and NoteChat
    synthetic conversations.

    Source: https://huggingface.co/datasets/AGBonnet/augmented-clinical-notes

    Metadata fields:
        - patient_summary: Summary of the patient case
    """

    base_name = "clinical_notes"
    spec = DatasetSpec(
        hf_path="AGBonnet/augmented-clinical-notes",
        shape="custom",
        builder_fn=_build_clinical_notes,
        metadata_fields={
            "patient_summary": ("patient_summary", "summary"),
        },
    )


class KnowMedicalDialogueDataset(HFDataset):
    """
    Know Medical Dialogue V2: Medical Q&A conversations.

    Collection of conversational exchanges between patients and doctors
    on various medical topics.

    Source: https://huggingface.co/datasets/knowrohit07/know_medical_dialogue_v2
    """

    base_name = "know_medical"
    spec = DatasetSpec(
        hf_path="knowrohit07/know_medical_dialogue_v2",
        shape="fields",
        user_fields=("Patient", "patient", "input"),
        assistant_fields=("Doctor", "doctor", "output"),
    )
