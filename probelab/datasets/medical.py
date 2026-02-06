"""Medical and healthcare conversation datasets."""

from typing import Any

from datasets import load_dataset

from ..types import Label, Message
from .base import Dataset
from .registry import Topic, _register_dataset


@_register_dataset("meddialog", Topic.MEDICAL, "MedDialog")
def meddialog() -> Dataset:
    """MedDialog 260K+ doctor-patient conversations."""
    data = load_dataset("bigbio/meddialog", "meddialog_en_bigbio_text")["train"]

    dialogues, labels = [], []

    for item in data:
        if text := item.get("text", ""):
            dialogues.append([Message("user", "Medical consultation:"), Message("assistant", text)])
            labels.append(Label.NEGATIVE)

    return Dataset(dialogues, labels, "meddialog").shuffle()


@_register_dataset("medical_soap", Topic.MEDICAL, "Medical SOAP")
def medical_soap() -> Dataset:
    """Medical Dialogue to SOAP Summary 10K."""
    data = load_dataset("omi-health/medical-dialogue-to-soap-summary")["train"]

    dialogues, labels = [], []
    metadata: dict[str, list[Any]] = {"soap_summary": []}

    for item in data:
        dialogue_text = item.get("dialogue", "")
        if not dialogue_text:
            continue

        dialogues.append([Message("user", dialogue_text)])
        labels.append(Label.NEGATIVE)
        metadata["soap_summary"].append(item.get("soap"))

    return Dataset(dialogues, labels, "medical_soap", metadata).shuffle()


@_register_dataset("clinical_notes", Topic.MEDICAL, "Clinical notes")
def clinical_notes() -> Dataset:
    """Augmented Clinical Notes with synthetic conversations."""
    data = load_dataset("AGBonnet/augmented-clinical-notes")["train"]

    dialogues, labels = [], []
    metadata: dict[str, list[Any]] = {"patient_summary": []}

    for item in data:
        conversation = item.get("conversation", item.get("dialogue", ""))
        if not conversation:
            continue

        if isinstance(conversation, str):
            dialogue = [Message("assistant", conversation)]
        elif isinstance(conversation, list):
            dialogue = []
            for msg in conversation:
                if isinstance(msg, dict) and (content := msg.get("content", "")):
                    dialogue.append(Message(msg.get("role", "assistant"), content))
                elif isinstance(msg, str):
                    dialogue.append(Message("assistant", msg))
        else:
            continue

        if dialogue:
            dialogues.append(dialogue)
            labels.append(Label.NEGATIVE)
            metadata["patient_summary"].append(item.get("patient_summary") or item.get("summary"))

    return Dataset(dialogues, labels, "clinical_notes", metadata).shuffle()


@_register_dataset("know_medical", Topic.MEDICAL, "Know medical dialogue")
def know_medical() -> Dataset:
    """Know Medical Dialogue V2 - medical Q&A."""
    data = load_dataset("knowrohit07/know_medical_dialogue_v2")["train"]

    dialogues, labels = [], []

    for item in data:
        user = item.get("Patient") or item.get("patient") or item.get("input") or ""
        assistant = item.get("Doctor") or item.get("doctor") or item.get("output") or ""
        if not user:
            continue

        dialogue = [Message("user", user)]
        if assistant:
            dialogue.append(Message("assistant", assistant))

        dialogues.append(dialogue)
        labels.append(Label.NEGATIVE)

    return Dataset(dialogues, labels, "know_medical").shuffle()
