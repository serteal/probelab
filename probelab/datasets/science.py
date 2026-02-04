"""
Science and STEM domain datasets.

These datasets provide scientific questions, academic content, and
educational material, useful for training probes to detect STEM content.
"""

from typing import Any, ClassVar

from datasets import load_dataset

from ..types import Dialogue, DialogueDataType, Label, Message
from .base import DialogueDataset
from .builders import sample_hf_dataset
from .hf_dataset import DatasetSpec, HFDataset
from .registry import register


@register("science", "Science QA dataset")
class ScienceQADataset(DialogueDataset):
    """
    ScienceQA: Multimodal science question answering dataset.

    Contains 21K+ science questions spanning natural science, language science,
    and social science with multiple choice answers and explanations.

    Source: https://huggingface.co/datasets/derek-thomas/ScienceQA

    Metadata fields:
        - subject: Content area (natural_science, language_science, social_science)
        - topic: Specific topic within subject
        - grade: Educational level
        - correct_answer: Index of the correct answer
    """

    base_name = "scienceqa"

    def _get_dialogues(self, **kwargs) -> DialogueDataType:
        max_samples = kwargs.get("max_samples")
        split_name = kwargs.get("split", "train")

        dataset = load_dataset("derek-thomas/ScienceQA")
        split = dataset[split_name]

        split = sample_hf_dataset(split, max_samples)

        dialogues: list[Dialogue] = []
        labels: list[Label] = []
        metadata: dict[str, list[Any]] = {
            "subject": [],
            "topic": [],
            "grade": [],
            "correct_answer": [],
        }

        choice_letters = ["A", "B", "C", "D", "E"]

        for item in split:
            question = item.get("question", "")
            choices = item.get("choices", [])
            answer_idx = item.get("answer", 0)
            hint = item.get("hint", "")
            solution = item.get("solution", "")

            if question:
                # Format question with context and choices
                user_content = question
                if hint:
                    user_content = f"Context: {hint}\n\nQuestion: {question}"

                if choices:
                    choices_str = "\n".join(
                        [
                            f"{choice_letters[i]}. {c}"
                            for i, c in enumerate(choices)
                            if i < len(choice_letters)
                        ]
                    )
                    user_content += f"\n\nChoices:\n{choices_str}"

                dialogue: Dialogue = [
                    Message(role="user", content=user_content),
                ]

                # Add answer and solution
                if choices and answer_idx < len(choices):
                    answer_text = (
                        f"The answer is {choice_letters[answer_idx]}: {choices[answer_idx]}"
                    )
                    if solution:
                        answer_text += f"\n\nExplanation: {solution}"
                    dialogue.append(Message(role="assistant", content=answer_text))

                dialogues.append(dialogue)
                labels.append(Label.NEGATIVE)

                metadata["subject"].append(item.get("subject", ""))
                metadata["topic"].append(item.get("topic", ""))
                metadata["grade"].append(item.get("grade", ""))
                metadata["correct_answer"].append(answer_idx)

        return dialogues, labels, metadata


@register("science", "MMMU benchmark")
class MMMUDataset(DialogueDataset):
    """
    MMMU: Massive Multi-discipline Multimodal Understanding benchmark.

    Contains ~1.5K questions across Biology, Chemistry, Physics, and other
    subjects requiring college-level knowledge.

    Source: https://huggingface.co/datasets/MMMU/MMMU

    Metadata fields:
        - subject: The academic subject (Biology, Chemistry, Physics, etc.)
        - subfield: More specific topic area
    """

    base_name = "mmmu"

    # Available subjects
    SUBJECTS: ClassVar[list[str]] = [
        "Biology",
        "Chemistry",
        "Physics",
        "Math",
        "Computer_Science",
        "Economics",
        "Psychology",
        "History",
        "Art",
        "Music",
    ]

    def _get_dialogues(self, **kwargs) -> DialogueDataType:
        max_samples = kwargs.get("max_samples")
        subjects = kwargs.get("subjects", ["Biology", "Chemistry", "Physics"])
        split_name = kwargs.get("split", "validation")  # MMMU uses validation

        all_dialogues: list[Dialogue] = []
        all_labels: list[Label] = []
        all_metadata: dict[str, list[Any]] = {
            "subject": [],
            "subfield": [],
        }

        for subject in subjects:
            if subject not in self.SUBJECTS:
                continue

            try:
                dataset = load_dataset("MMMU/MMMU", subject)
                split = dataset[split_name]

                for item in split:
                    question = item.get("question", "")
                    options = item.get("options", [])
                    answer = item.get("answer", "")

                    if question:
                        user_content = question
                        if options:
                            options_str = "\n".join(
                                [f"{chr(65+i)}. {opt}" for i, opt in enumerate(options)]
                            )
                            user_content += f"\n\nOptions:\n{options_str}"

                        dialogue: Dialogue = [
                            Message(role="user", content=user_content),
                        ]
                        if answer:
                            dialogue.append(
                                Message(role="assistant", content=f"Answer: {answer}")
                            )

                        all_dialogues.append(dialogue)
                        all_labels.append(Label.NEGATIVE)
                        all_metadata["subject"].append(subject)
                        all_metadata["subfield"].append(item.get("subfield", ""))
            except Exception:
                # Skip subjects that fail to load
                continue

        # Apply sample limit after collecting all subjects
        if max_samples and len(all_dialogues) > max_samples:
            import random

            random.seed(42)
            indices = random.sample(range(len(all_dialogues)), max_samples)
            all_dialogues = [all_dialogues[i] for i in indices]
            all_labels = [all_labels[i] for i in indices]
            all_metadata = {k: [v[i] for i in indices] for k, v in all_metadata.items()}

        return all_dialogues, all_labels, all_metadata


@register("science", "Biology tree-of-thought")
class BiologyToTDataset(HFDataset):
    """
    Biology Tree-of-Thought reasoning dataset.

    Contains 5,752 biology Q&A examples with detailed chain-of-thought reasoning,
    focusing on cellular biology topics like membrane structure, signaling,
    and protein interactions.

    Source: https://huggingface.co/datasets/LLMTeamAkiyama/cleand_moremilk_ToT-Biology

    Metadata fields:
        - id: Unique identifier
        - total_token: Token count per record
        - base_datasets_id: Reference to source data
        - thought: Chain-of-thought reasoning process
    """

    base_name = "biology_tot"
    spec = DatasetSpec(
        hf_path="LLMTeamAkiyama/cleand_moremilk_ToT-Biology",
        shape="fields",
        user_fields=("question",),
        assistant_fields=("answer",),
        metadata_fields={
            "id": ("id",),
            "total_token": ("total_token",),
            "base_datasets_id": ("base_datasets_id",),
            "thought": ("thought",),
        },
    )


@register("science", "Biochemistry reasoning")
class BiochemReasoningDataset(HFDataset):
    """
    Biochemistry reasoning dataset from PrimeKG knowledge graph.

    Contains 10,327 biomedical reasoning examples covering chemistry, biology,
    drug interactions, disease treatments, molecular pathways, and protein functions.
    Includes detailed step-by-step reasoning chains.

    Source: https://huggingface.co/datasets/extrasensory/reasoning-biochem

    Metadata fields:
        - reasoning_chains: Step-by-step logical reasoning traces
    """

    base_name = "biochem_reasoning"
    spec = DatasetSpec(
        hf_path="extrasensory/reasoning-biochem",
        shape="fields",
        user_fields=("question",),
        assistant_fields=("response",),
        metadata_fields={
            "reasoning_chains": ("reasoning_chains",),
        },
    )


@register("science", "STEM Q&A")
class StemQADataset(HFDataset):
    """
    Combined STEM Q&A from multiple sources.

    Aggregates science questions from various educational datasets.

    Source: https://huggingface.co/datasets/camel-ai/physics

    Metadata fields:
        - topic: Topic/subject of the question
    """

    base_name = "stem_qa"
    spec = DatasetSpec(
        hf_path="camel-ai/physics",
        shape="fields",
        user_fields=("message_1", "instruction"),
        assistant_fields=("message_2", "response"),
        metadata_fields={
            "topic": ("topic", "subject"),
        },
    )
