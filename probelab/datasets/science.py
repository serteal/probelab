"""Science and STEM domain datasets."""

import logging
from typing import Any

from datasets import load_dataset

from ..types import Label, Message
from .base import Dataset
from .registry import Topic, _register_dataset

logger = logging.getLogger(__name__)

CHOICE_LETTERS = ["A", "B", "C", "D", "E"]
MMMU_SUBJECTS = ["Biology", "Chemistry", "Physics", "Math", "Computer_Science", "Economics", "Psychology", "History", "Art", "Music"]


@_register_dataset("scienceqa", Topic.SCIENCE, "ScienceQA")
def scienceqa(split: str = "train") -> Dataset:
    """ScienceQA 21K+ multimodal science questions."""
    data = load_dataset("derek-thomas/ScienceQA")[split]

    dialogues, labels = [], []
    metadata: dict[str, list[Any]] = {"subject": [], "topic": [], "grade": [], "correct_answer": []}

    for item in data:
        if not (question := item.get("question", "")):
            continue

        user_content = f"Context: {hint}\n\nQuestion: {question}" if (hint := item.get("hint", "")) else question
        if choices := item.get("choices", []):
            user_content += "\n\nChoices:\n" + "\n".join(f"{CHOICE_LETTERS[i]}. {c}" for i, c in enumerate(choices) if i < len(CHOICE_LETTERS))

        dialogue = [Message("user", user_content)]
        answer_idx = item.get("answer", 0)
        if choices and answer_idx < len(choices):
            answer_text = f"The answer is {CHOICE_LETTERS[answer_idx]}: {choices[answer_idx]}"
            if solution := item.get("solution", ""):
                answer_text += f"\n\nExplanation: {solution}"
            dialogue.append(Message("assistant", answer_text))

        dialogues.append(dialogue)
        labels.append(Label.NEGATIVE)
        metadata["subject"].append(item.get("subject", ""))
        metadata["topic"].append(item.get("topic", ""))
        metadata["grade"].append(item.get("grade", ""))
        metadata["correct_answer"].append(answer_idx)

    return Dataset(dialogues, labels, "scienceqa", metadata).shuffle()


@_register_dataset("mmmu", Topic.SCIENCE, "MMMU")
def mmmu(subjects: list[str] | None = None, split: str = "validation") -> Dataset:
    """MMMU 1.5K+ college-level multimodal questions."""
    subjects = subjects or ["Biology", "Chemistry", "Physics"]
    dialogues, labels = [], []
    metadata: dict[str, list[Any]] = {"subject": [], "subfield": []}

    for subject in subjects:
        if subject not in MMMU_SUBJECTS:
            continue
        try:
            data = load_dataset("MMMU/MMMU", subject)[split]
            for item in data:
                if not (question := item.get("question", "")):
                    continue

                user_content = question
                if options := item.get("options", []):
                    user_content += "\n\nOptions:\n" + "\n".join(f"{chr(65+i)}. {opt}" for i, opt in enumerate(options))

                dialogue = [Message("user", user_content)]
                if answer := item.get("answer", ""):
                    dialogue.append(Message("assistant", f"Answer: {answer}"))

                dialogues.append(dialogue)
                labels.append(Label.NEGATIVE)
                metadata["subject"].append(subject)
                metadata["subfield"].append(item.get("subfield", ""))
        except Exception:
            continue

    return Dataset(dialogues, labels, "mmmu", metadata).shuffle()


@_register_dataset("biology_tot", Topic.SCIENCE, "Biology ToT")
def biology_tot() -> Dataset:
    """Biology Tree-of-Thought 5.7K Q&A."""
    data = load_dataset("LLMTeamAkiyama/cleand_moremilk_ToT-Biology")["train"]

    dialogues, labels = [], []
    metadata: dict[str, list[Any]] = {"id": [], "total_token": [], "base_datasets_id": [], "thought": []}

    for item in data:
        question = item.get("question", "")
        answer = item.get("answer", "")
        if not question:
            continue

        dialogue = [Message("user", question)]
        if answer:
            dialogue.append(Message("assistant", answer))

        dialogues.append(dialogue)
        labels.append(Label.NEGATIVE)
        metadata["id"].append(item.get("id"))
        metadata["total_token"].append(item.get("total_token"))
        metadata["base_datasets_id"].append(item.get("base_datasets_id"))
        metadata["thought"].append(item.get("thought"))

    return Dataset(dialogues, labels, "biology_tot", metadata).shuffle()


@_register_dataset("biochem_reasoning", Topic.SCIENCE, "Biochem reasoning")
def biochem_reasoning() -> Dataset:
    """Biochemistry reasoning 10K+ from PrimeKG."""
    data = load_dataset("extrasensory/reasoning-biochem")["train"]

    dialogues, labels = [], []
    metadata: dict[str, list[Any]] = {"reasoning_chains": []}

    for item in data:
        question = item.get("question", "")
        response = item.get("response", "")
        if not question:
            continue

        dialogue = [Message("user", question)]
        if response:
            dialogue.append(Message("assistant", response))

        dialogues.append(dialogue)
        labels.append(Label.NEGATIVE)
        metadata["reasoning_chains"].append(item.get("reasoning_chains"))

    return Dataset(dialogues, labels, "biochem_reasoning", metadata).shuffle()


@_register_dataset("stem_qa", Topic.SCIENCE, "STEM Q&A")
def stem_qa() -> Dataset:
    """Combined STEM Q&A from CAMEL-AI physics."""
    data = load_dataset("camel-ai/physics")["train"]

    dialogues, labels = [], []
    metadata: dict[str, list[Any]] = {"topic": []}

    for item in data:
        user = item.get("message_1") or item.get("instruction") or ""
        assistant = item.get("message_2") or item.get("response") or ""
        if not user:
            continue

        dialogue = [Message("user", user)]
        if assistant:
            dialogue.append(Message("assistant", assistant))

        dialogues.append(dialogue)
        labels.append(Label.NEGATIVE)
        metadata["topic"].append(item.get("topic") or item.get("subject"))

    return Dataset(dialogues, labels, "stem_qa", metadata).shuffle()


@_register_dataset("wmdp_bio", Topic.SCIENCE, "WMDP biosecurity MC questions")
def wmdp_bio() -> Dataset:
    """WMDP-bio: 1273 multiple-choice biosecurity questions from CAIS."""
    data = load_dataset("cais/wmdp", "wmdp-bio")["test"]

    dialogues, labels = [], []

    for item in data:
        question = item["question"]
        choices = item["choices"]
        answer_idx = item["answer"]
        formatted = question + "\n\n" + "\n".join(
            f"{chr(65 + i)}. {c}" for i, c in enumerate(choices)
        )
        answer_text = f"The answer is {chr(65 + answer_idx)}: {choices[answer_idx]}"
        dialogues.append([Message("user", formatted), Message("assistant", answer_text)])
        labels.append(Label.POSITIVE)

    return Dataset(dialogues, labels, "wmdp_bio").shuffle()


@_register_dataset("virology_qa", Topic.SCIENCE, "Virology Q/A pairs")
def virology_qa() -> Dataset:
    """Virology 8.6K+ dual-use virology Q/A pairs (augmented, multilingual)."""
    try:
        hf_ds = load_dataset("serteal/virology-qa-pairs", split="train")
    except Exception as e:
        logger.debug("HF load failed for serteal/virology-qa-pairs: %s", e)
        raise FileNotFoundError(
            "virology-qa-pairs not found on HuggingFace (serteal/virology-qa-pairs)"
        ) from e

    dialogues, labels = [], []
    metadata: dict[str, list[Any]] = {
        "topic": [], "subtopic": [], "question_type": [],
        "augmentation": [], "language": [], "base_id": [],
    }

    for item in hf_ds:
        question = item.get("question", "")
        answer = item.get("answer", "")
        if not question or not answer:
            continue
        dialogues.append([Message("user", question), Message("assistant", answer)])
        labels.append(Label.POSITIVE)
        metadata["topic"].append(item.get("topic", ""))
        metadata["subtopic"].append(item.get("subtopic", ""))
        metadata["question_type"].append(item.get("question_type", ""))
        metadata["augmentation"].append(item.get("augmentation", ""))
        metadata["language"].append(item.get("language", ""))
        metadata["base_id"].append(item.get("base_id", -1))

    return Dataset(dialogues, labels, "virology_qa", metadata).shuffle()
