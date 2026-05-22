"""Science and STEM domain datasets."""

import json
import logging
from pathlib import Path
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


def _parse_bioprobe_item(item: dict, task_type: str) -> tuple[str, str]:
    """Extract (question, answer) from a BioProBench item."""
    match task_type:
        case "PQA":
            q = item.get("question", "")
            choices = item.get("choices", [])
            if choices:
                q += "\n\n" + "\n".join(
                    f"{chr(65 + i)}. {c}" for i, c in enumerate(choices)
                )
            answer = str(item.get("answer", ""))
        case "GEN":
            q = item.get("input", "")
            output = item.get("output", [])
            answer = "\n".join(
                f"{i + 1}. {s}" for i, s in enumerate(output)
            ) if isinstance(output, list) else str(output)
        case "ORD":
            q = item.get("question", "")
            steps = item.get("correct_steps", [])
            answer = "\n".join(
                f"{i + 1}. {s}" for i, s in enumerate(steps)
            ) if isinstance(steps, list) else str(steps)
        case "ERR":
            q = f"Check this protocol step for errors:\n{item.get('corrupted_text', '')}"
            answer = item.get("corrected_text", "")
        case _:
            q, answer = "", ""
    return q, answer


@_register_dataset("bioprobe_bench", Topic.SCIENCE, "BioProBench biomedical protocols")
def bioprobe_bench(path: str | None = None) -> Dataset:
    """BioProBench: 4K+ biomedical protocol understanding tasks.

    Loads from HuggingFace (serteal/bioprobe-bench), falls back to local files.

    Args:
        path: Optional directory containing {ERR,GEN,ORD,PQA}_test.json files.
    """
    dialogues, labels = [], []
    metadata: dict[str, list[Any]] = {"task_type": [], "id": []}

    # Try HuggingFace first
    try:
        hf_ds = load_dataset("serteal/bioprobe-bench", split="train")
        for row in hf_ds:
            item = json.loads(row["data"])
            task_type = row["task_type"]
            q, answer = _parse_bioprobe_item(item, task_type)
            if not q:
                continue
            dialogues.append([Message("user", q), Message("assistant", str(answer))])
            labels.append(Label.NEGATIVE)
            metadata["task_type"].append(task_type)
            metadata["id"].append(row.get("id", ""))
        return Dataset(dialogues, labels, "bioprobe_bench", metadata).shuffle()
    except Exception as e:
        logger.debug("HF load failed for serteal/bioprobe-bench: %s", e)

    # Fall back to local files
    if path is None:
        raise FileNotFoundError(
            "BioProBench not found on HuggingFace (serteal/bioprobe-bench) "
            "and no local path provided"
        )
    base = Path(path)
    for task_type in ["PQA", "GEN", "ORD", "ERR"]:
        json_path = base / f"{task_type}_test.json"
        if not json_path.exists():
            continue
        data = json.loads(json_path.read_text())
        for item in data:
            q, answer = _parse_bioprobe_item(item, task_type)
            if not q:
                continue
            dialogues.append([Message("user", q), Message("assistant", str(answer))])
            labels.append(Label.NEGATIVE)
            metadata["task_type"].append(task_type)
            metadata["id"].append(item.get("id", ""))

    return Dataset(dialogues, labels, "bioprobe_bench", metadata).shuffle()


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


@_register_dataset("wmdp_chem", Topic.SCIENCE, "WMDP chemical security MC questions")
def wmdp_chem() -> Dataset:
    """WMDP-chem: 408 multiple-choice chemical security questions from CAIS."""
    data = load_dataset("cais/wmdp", "wmdp-chem")["test"]

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

    return Dataset(dialogues, labels, "wmdp_chem").shuffle()


_CBRN_SYNTH_TAXONOMY = (
    "protocol_troubleshooting",
    "experimental_design",
    "tacit_lab_failure_diagnosis",
    "literature_to_procedure_translation",
    "pathogenicity_host_range_reasoning",
    "assay_selection_strategy",
    "operational_constraint_handling",
)

_CBRN_SYNTH_TRAIN_TAXONOMY = (
    "protocol_troubleshooting",
    "experimental_design",
    "tacit_lab_failure_diagnosis",
    "literature_to_procedure_translation",
    "pathogenicity_host_range_reasoning",
)
_CBRN_SYNTH_HELDOUT_TAXONOMY = (
    "assay_selection_strategy",
    "operational_constraint_handling",
)
_CBRN_SYNTH_PRIMARY_GENERATOR = "huihui-ai/gemma-3-27b-it-abliterated"
_CBRN_SYNTH_HELDOUT_GENERATOR = "huihui-ai/Qwen2.5-72B-Instruct-abliterated"


def _load_cbrn_synth_rows(path: str | None = None) -> list[dict]:
    """Load cbrn_synth_v1 raw rows from HF (preferred) or local JSON."""
    try:
        hf_ds = load_dataset("serteal/cbrn-synth-v1", split="train")
        return list(hf_ds)
    except Exception as e:
        logger.debug("HF load failed for serteal/cbrn-synth-v1: %s", e)
    if path is None:
        candidates = sorted(
            Path(__file__).resolve().parents[3]
            .glob("dataset-generation/cbrn_synth/outputs/cbrn_synth_*/qa_pairs_judged.json")
        )
        if not candidates:
            raise FileNotFoundError(
                "cbrn-synth-v1 not found on HuggingFace (serteal/cbrn-synth-v1) "
                "and no local outputs at "
                "dataset-generation/cbrn_synth/outputs/*/qa_pairs_judged.json. "
                "Run dataset-generation/cbrn_synth/generate_dataset.py first."
            )
        path = str(candidates[-1])
    with open(path) as f:
        return json.load(f)


def _build_cbrn_synth_dataset(
    rows: list[dict], name: str, judge_threshold: float = 0.6,
) -> "Dataset":
    """Filter judged rows -> Dataset with metadata for downstream split filters."""
    dialogues, labels = [], []
    metadata: dict[str, list[Any]] = {
        "id": [], "base_id": [], "generator_model": [], "taxonomy_category": [],
        "topic": [], "subtopic": [], "language": [], "augmentation": [],
        "label_reason": [], "judge_score": [],
    }
    for row in rows:
        question = row.get("question", "")
        answer = row.get("answer", "")
        if not question or not answer:
            continue
        score = float(row.get("judge_score", 0.0))
        if score < judge_threshold:
            continue
        dialogues.append([Message("user", question), Message("assistant", answer)])
        labels.append(Label.POSITIVE)
        metadata["id"].append(row.get("id"))
        metadata["base_id"].append(row.get("base_id"))
        metadata["generator_model"].append(row.get("generator_model", ""))
        metadata["taxonomy_category"].append(row.get("taxonomy_category", ""))
        metadata["topic"].append(row.get("topic", ""))
        metadata["subtopic"].append(row.get("subtopic", ""))
        metadata["language"].append(row.get("language", "English"))
        metadata["augmentation"].append(row.get("augmentation", ""))
        metadata["label_reason"].append(row.get("label_reason", ""))
        metadata["judge_score"].append(score)
    return Dataset(dialogues, labels, name, metadata).shuffle()


@_register_dataset(
    "cbrn_synth_v1", Topic.SCIENCE,
    "CBRN synthetic positives, taxonomy-tagged, judge-labeled",
)
def cbrn_synth_v1(path: str | None = None, judge_threshold: float = 0.6) -> Dataset:
    """Synthetic CBRN misuse-assistance Q/A from uncensored generators.

    Returns *all* rows that passed the judge. Use the `_train` / `_iid` /
    `_ood_held_generator` variants below for the actual splits used in E3.

    Args:
        path: Optional path to the judged JSON file. Defaults to the most recent
            output in ``dataset-generation/cbrn_synth/outputs/``.
        judge_threshold: Minimum ``judge_score`` to keep as a positive. Default
            0.6 (matches the rubric's "materially_increases_misuse_capability"
            cutoff).
    """
    rows = _load_cbrn_synth_rows(path)
    return _build_cbrn_synth_dataset(rows, "cbrn_synth_v1", judge_threshold)


@_register_dataset(
    "cbrn_synth_v1_train", Topic.SCIENCE,
    "cbrn_synth_v1 train split (primary generator, training taxonomy)",
)
def cbrn_synth_v1_train(path: str | None = None, judge_threshold: float = 0.6) -> Dataset:
    """Primary-generator rows scoped to the training taxonomy categories."""
    rows = [
        r for r in _load_cbrn_synth_rows(path)
        if r.get("generator_model", "") == _CBRN_SYNTH_PRIMARY_GENERATOR
        and r.get("taxonomy_category", "") in _CBRN_SYNTH_TRAIN_TAXONOMY
    ]
    return _build_cbrn_synth_dataset(rows, "cbrn_synth_v1_train", judge_threshold)


@_register_dataset(
    "cbrn_synth_v1_iid", Topic.SCIENCE,
    "cbrn_synth_v1 IID split (primary generator, held-out taxonomy)",
)
def cbrn_synth_v1_iid(path: str | None = None, judge_threshold: float = 0.6) -> Dataset:
    """Primary generator on held-out taxonomy categories — IID-style eval."""
    rows = [
        r for r in _load_cbrn_synth_rows(path)
        if r.get("generator_model", "") == _CBRN_SYNTH_PRIMARY_GENERATOR
        and r.get("taxonomy_category", "") in _CBRN_SYNTH_HELDOUT_TAXONOMY
    ]
    return _build_cbrn_synth_dataset(rows, "cbrn_synth_v1_iid", judge_threshold)


@_register_dataset(
    "cbrn_synth_v1_ood_held_generator", Topic.SCIENCE,
    "cbrn_synth_v1 OOD split (held-out generator, all taxonomy)",
)
def cbrn_synth_v1_ood_held_generator(
    path: str | None = None, judge_threshold: float = 0.6,
) -> Dataset:
    """Held-out generator across all taxonomy — final OOD positive."""
    rows = [
        r for r in _load_cbrn_synth_rows(path)
        if r.get("generator_model", "") == _CBRN_SYNTH_HELDOUT_GENERATOR
    ]
    return _build_cbrn_synth_dataset(
        rows, "cbrn_synth_v1_ood_held_generator", judge_threshold,
    )


@_register_dataset(
    "lab_bench", Topic.SCIENCE,
    "LAB-Bench: practical biology lab reasoning (FutureHouse)",
)
def lab_bench(subtask: str | None = None) -> Dataset:
    """LAB-Bench: practical wet-lab biology questions used as benign hard
    negatives for the CBRN probe (the questions test biology *competence* but
    are not misuse-assistance).

    Args:
        subtask: Optional LAB-Bench subtask ('LitQA2', 'CloningScenarios',
            'ProtocolQA', 'SeqQA', 'FigQA', 'TableQA', 'DbQA', 'SuppQA').
            Default ``None`` loads all available subtasks.
    """
    subtasks = (
        [subtask] if subtask is not None
        else [
            "LitQA2", "CloningScenarios", "ProtocolQA", "SeqQA",
            "FigQA", "TableQA", "DbQA", "SuppQA",
        ]
    )
    dialogues, labels = [], []
    metadata: dict[str, list[Any]] = {"subtask": [], "id": []}

    for st in subtasks:
        try:
            data = load_dataset("futurehouse/lab-bench", st, split="train")
        except Exception as e:
            logger.debug("LAB-Bench subtask %s unavailable: %s", st, e)
            continue
        for item in data:
            question = item.get("question") or item.get("input") or ""
            if not question:
                continue
            choices = item.get("choices") or item.get("options") or []
            if isinstance(choices, list) and choices:
                question = question + "\n\n" + "\n".join(
                    f"{chr(65 + i)}. {c}" for i, c in enumerate(choices)
                )
            answer = (
                item.get("ideal") or item.get("answer")
                or item.get("correct_answer") or ""
            )
            dialogue = [Message("user", question)]
            if answer:
                dialogue.append(Message("assistant", str(answer)))
            dialogues.append(dialogue)
            labels.append(Label.NEGATIVE)
            metadata["subtask"].append(st)
            metadata["id"].append(item.get("id", ""))

    if not dialogues:
        raise FileNotFoundError(
            "LAB-Bench not loadable from HuggingFace (futurehouse/lab-bench). "
            "Check network or HF auth."
        )
    return Dataset(dialogues, labels, "lab_bench", metadata).shuffle()


@_register_dataset(
    "pmc_virology_abstracts", Topic.SCIENCE,
    "PubMed/PMC virology abstracts as benign biology hard negatives",
)
def pmc_virology_abstracts(max_samples: int = 10_000) -> Dataset:
    """Biomedical abstracts presented as user-question / assistant-summary
    dialogues. Used as benign hard negatives for the CBRN probe (real
    biological literature, no misuse-assistance content).

    Source: ``qiaojin/PubMedQA`` (pqa_artificial split). Each row's
    ``question`` becomes the user turn; the ``long_answer`` (which is the
    abstract conclusion) becomes the assistant turn.

    Args:
        max_samples: Cap on the number of returned rows. Default 10k keeps
            the activation collection under ~160 GB at 5 layers × bf16.
    """
    try:
        data = load_dataset("qiaojin/PubMedQA", "pqa_artificial", split="train")
    except Exception as e:
        raise FileNotFoundError(
            "PubMedQA not loadable from HuggingFace (qiaojin/PubMedQA). "
            f"Original error: {e}"
        ) from e

    dialogues, labels = [], []
    metadata: dict[str, list[Any]] = {"pubid": [], "context_size": []}
    n = 0
    for item in data:
        if n >= max_samples:
            break
        question = item.get("question", "")
        answer = item.get("long_answer", "")
        if not question or not answer:
            continue
        dialogues.append([Message("user", question), Message("assistant", answer)])
        labels.append(Label.NEGATIVE)
        metadata["pubid"].append(item.get("pubid"))
        ctx = item.get("context", {}) or {}
        metadata["context_size"].append(
            len(ctx.get("contexts", [])) if isinstance(ctx, dict) else 0
        )
        n += 1
    return Dataset(dialogues, labels, "pmc_virology_abstracts", metadata).shuffle()


@_register_dataset("virology_qa", Topic.SCIENCE, "Virology Q/A pairs")
def virology_qa(split: str | None = None, path: str | None = None) -> Dataset:
    """Virology 8.6K+ dual-use virology Q/A pairs (augmented, multilingual).

    Args:
        split: ``"train"`` or ``"test"`` to load a pre-split file
            (``qa_pairs_train.json`` / ``qa_pairs_test.json``).
            ``None`` loads the full HuggingFace dataset.
        path: Directory containing the split JSON files.  Required when
            *split* is not ``None``.
    """
    if split is not None:
        try:
            hf_ds = load_dataset("serteal/virology-qa-pairs", split=split)
            items = list(hf_ds)
        except Exception as e:
            logger.debug("HF load failed for virology_qa/%s: %s", split, e)
            if path is None:
                raise FileNotFoundError(
                    f"virology-qa-pairs split '{split}' not found on HuggingFace "
                    "and no local path provided"
                ) from e
            split_file = Path(path) / f"qa_pairs_{split}.json"
            with open(split_file) as f:
                items = json.load(f)
    else:
        try:
            hf_ds = load_dataset("serteal/virology-qa-pairs", split="train")
        except Exception as e:
            logger.debug("HF load failed for serteal/virology-qa-pairs: %s", e)
            raise FileNotFoundError(
                "virology-qa-pairs not found on HuggingFace (serteal/virology-qa-pairs)"
            ) from e
        items = list(hf_ds)

    dialogues, labels = [], []
    metadata: dict[str, list[Any]] = {
        "topic": [], "subtopic": [], "question_type": [],
        "augmentation": [], "language": [], "base_id": [],
    }

    for item in items:
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
