"""Harmfulness detection datasets."""

import logging
from pathlib import Path

import pandas as pd
from datasets import load_dataset

from ..types import Label, Message
from .base import Dataset
from .registry import Topic, _register_dataset

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
PROJECT_ROOT = REPO_ROOT.parent  # scaling-probes/


def _sanitize_prompt(prompt: str) -> str:
    return prompt.replace("<bos><start_of_turn>user\n", "").replace("<end_of_turn>\n<start_of_turn>model\n", "")


@_register_dataset("wild_jailbreak", Topic.HARMFULNESS, "Wild jailbreak dataset")
def wild_jailbreak() -> Dataset:
    """WildJailbreak training split from HuggingFace. Uses `adversarial`
    prompts as the user turn and the paired completion as assistant.
    Keeps adversarial_harmful + adversarial_benign so the label is
    well-defined."""
    ds = load_dataset(
        "allenai/wildjailbreak", "train",
        delimiter="\t", keep_default_na=False,
    )["train"]
    dialogues, labels = [], []
    for row in ds:
        dtype = str(row["data_type"])
        if dtype not in ("adversarial_harmful", "adversarial_benign"):
            continue
        label_val = (
            Label.POSITIVE if dtype == "adversarial_harmful"
            else Label.NEGATIVE
        )
        prompt = str(row["adversarial"] or "")
        completion = str(row["completion"] or "")
        dialogues.append([
            Message("user", prompt),
            Message("assistant", completion),
        ])
        labels.append(label_val)
    return Dataset(dialogues, labels, "wild_jailbreak").shuffle()


@_register_dataset(
    "wild_jailbreak_aug", Topic.HARMFULNESS,
    "WildJailbreak balanced subset for augmenting harmful training data",
)
def wild_jailbreak_aug() -> Dataset:
    """Balanced 80k-sample subset of WildJailbreak, 20k per data_type.
    Deterministic (seed=0). Used as additional harmful training data
    on top of wildguard_mix_train for §6B harmful-scaling."""
    import random
    from collections import defaultdict

    ds = load_dataset(
        "allenai/wildjailbreak", "train",
        delimiter="\t", keep_default_na=False,
    )["train"]

    per_type: dict[str, list[int]] = defaultdict(list)
    for i, row in enumerate(ds):
        per_type[str(row["data_type"])].append(i)

    # Balanced 50k subset (12_500 per data_type) — sized to fit a
    # ~960 GB 5-layer activation cache in the remaining disk budget.
    # 50k + the 48k existing wildguard_mix_train gets harmful training
    # to ~98k total, which is enough to anchor a pre-registered
    # extrapolation point at n=100k.
    PER_TYPE = 12_500
    rng = random.Random(0)
    keep: set[int] = set()
    for dtype in ("adversarial_harmful", "adversarial_benign",
                  "vanilla_harmful", "vanilla_benign"):
        pool = list(per_type.get(dtype, []))
        rng.shuffle(pool)
        keep.update(pool[:PER_TYPE])

    dialogues, labels = [], []
    for i, row in enumerate(ds):
        if i not in keep:
            continue
        dtype = str(row["data_type"])
        label_val = (
            Label.POSITIVE if "harmful" in dtype else Label.NEGATIVE
        )
        # prefer the adversarial text when available
        prompt = (
            str(row["adversarial"] or "").strip()
            or str(row["vanilla"] or "").strip()
        )
        if not prompt:
            continue
        completion = str(row["completion"] or "")
        dialogues.append([
            Message("user", prompt),
            Message("assistant", completion),
        ])
        labels.append(label_val)

    return Dataset(dialogues, labels, "wild_jailbreak_aug").shuffle()


@_register_dataset("circuit_breakers", Topic.HARMFULNESS, "Circuit breakers dataset")
def circuit_breakers(split: str = "train") -> Dataset:
    ds = load_dataset("Mechanistic-Anomaly-Detection/gemma2-jailbreaks")
    data = ds["circuit_breakers_test"] if split == "test" else ds["circuit_breakers_train"]

    dialogues = [[Message("user", _sanitize_prompt(d["prompt"])), Message("assistant", d["completion"])] for d in data]
    labels = [Label.POSITIVE] * len(dialogues)

    return Dataset(dialogues, labels, "circuit_breakers").shuffle()


@_register_dataset("benign_instructions", Topic.HARMFULNESS, "Benign instructions")
def benign_instructions(split: str = "train") -> Dataset:
    ds = load_dataset("Mechanistic-Anomaly-Detection/gemma2-jailbreaks")
    data = ds["benign_instructions_test"] if split == "test" else ds["benign_instructions_train"]

    dialogues = [[Message("user", _sanitize_prompt(d["prompt"])), Message("assistant", d["completion"])] for d in data]
    labels = [Label.NEGATIVE] * len(dialogues)

    return Dataset(dialogues, labels, "benign_instructions").shuffle()


@_register_dataset("wildguard_mix", Topic.HARMFULNESS, "WildGuard mix dataset")
def wildguard_mix(split: str = "train") -> Dataset:
    ds_split = "wildguardtest" if split == "test" else "wildguardtrain"
    data = load_dataset("allenai/wildguardmix", ds_split)[split]

    dialogues, labels = [], []
    metadata = {"prompt_harm_label": [], "response_refusal_label": [], "response_harm_label": [], "adversarial": [], "subcategory": [], "response": []}

    seen: set[tuple[str, int]] = set()
    for d in data:
        prompt = str(d["prompt"])
        label_val = Label.POSITIVE if str(d["prompt_harm_label"]).lower() == "harmful" else Label.NEGATIVE
        key = (prompt, label_val.value)
        if key in seen:
            continue
        seen.add(key)
        dialogues.append([Message("system", "You are a helpful, honest assistant."), Message("user", prompt)])
        labels.append(label_val)
        for k in metadata:
            metadata[k].append(d.get(k))

    return Dataset(dialogues, labels, "wildguard_mix", metadata).shuffle()


@_register_dataset("xstest_response", Topic.HARMFULNESS, "XSTest response dataset")
def xstest_response() -> Dataset:
    data = load_dataset("allenai/xstest-response")["response_harmfulness"]

    dialogues = [[Message("user", str(d["prompt"])), Message("assistant", str(d["response"]))] for d in data]
    labels = [Label.POSITIVE if d["label"] == "harmful" else Label.NEGATIVE for d in data]

    return Dataset(dialogues, labels, "xstest_response").shuffle()


def _load_clearharm(name: str, config: str, local_path: Path) -> Dataset:
    """Load a clearharm dataset from HF, falling back to local JSON."""
    try:
        hf_ds = load_dataset("serteal/clearharm", config, split="train")
        df = hf_ds.to_pandas()
    except Exception as e:
        logger.debug("HF load failed for serteal/clearharm/%s, trying local: %s", config, e)
        if not local_path.exists():
            raise FileNotFoundError(
                f"ClearHarm not found on HuggingFace (serteal/clearharm/{config}) "
                f"or locally ({local_path})"
            ) from e
        df = pd.read_json(local_path)

    dialogues = [[Message("user", str(row["prompt"])), Message("assistant", str(row["generated_text"]))] for _, row in df.iterrows()]
    labels = [Label.POSITIVE] * len(dialogues)

    return Dataset(dialogues, labels, name).shuffle()


@_register_dataset("clearharm_llama3", Topic.HARMFULNESS, "ClearHarm Llama3")
def clearharm_llama3() -> Dataset:
    return _load_clearharm(
        "clearharm_llama3", "llama3",
        PROJECT_ROOT / "data" / "harmfulness" / "rollouts" / "outputs_huihui-ai_Meta-Llama-3.json",
    )


@_register_dataset("clearharm_mistral", Topic.HARMFULNESS, "ClearHarm Mistral")
def clearharm_mistral() -> Dataset:
    return _load_clearharm(
        "clearharm_mistral", "mistral",
        PROJECT_ROOT / "data" / "harmfulness" / "rollouts" / "outputs_huihui-ai_Mistral-Small-24B-Instruct-2501-abliterated_20250702_233538.json",
    )


@_register_dataset("clearharm_qwen3", Topic.HARMFULNESS, "ClearHarm Qwen3")
def clearharm_qwen3() -> Dataset:
    return _load_clearharm(
        "clearharm_qwen3", "qwen3",
        PROJECT_ROOT / "data" / "harmfulness" / "rollouts" / "outputs_huihui-ai_Huihui-Qwen3-14B-abliterated-v2_20250702_212758_cleaned.json",
    )


@_register_dataset("fortress_public", Topic.HARMFULNESS, "ScaleAI FORTRESS adversarial safety prompts")
def fortress_public() -> Dataset:
    """ScaleAI FORTRESS: 500 adversarial/benign prompt pairs across 3 risk domains.

    Each sample yields two dialogues: adversarial (POSITIVE) and benign (NEGATIVE).
    Risk domains: CBRNE, Criminal & Financial, Political Violence & Terrorism.
    """
    data = load_dataset("ScaleAI/fortress_public")["train"]

    dialogues, labels = [], []
    for row in data:
        dialogues.append([Message("user", str(row["adversarial_prompt"]))])
        labels.append(Label.POSITIVE)
        dialogues.append([Message("user", str(row["benign_prompt"]))])
        labels.append(Label.NEGATIVE)

    return Dataset(dialogues, labels, "fortress_public").shuffle()


@_register_dataset("or_bench", Topic.HARMFULNESS, "OR-Bench over-refusal benchmark")
def or_bench(config: str = "or-bench-80k") -> Dataset:
    """Load OR-Bench dataset (over-refusal benchmark).

    Args:
        config: HF config name. One of:
            - 'or-bench-80k': 80K safe prompts (Label.NEGATIVE)
            - 'or-bench-hard-1k': 1K hard safe prompts (Label.NEGATIVE)
            - 'or-bench-toxic': 655 toxic prompts (Label.POSITIVE)
    """
    data = load_dataset("bench-llms/or-bench", config)["train"]

    label = Label.POSITIVE if config == "or-bench-toxic" else Label.NEGATIVE
    dialogues = [[Message("user", str(row["prompt"]))] for row in data]
    labels = [label] * len(dialogues)
    metadata = {"category": [row.get("category", "") for row in data]}

    return Dataset(dialogues, labels, f"or_bench:{config}", metadata).shuffle()
