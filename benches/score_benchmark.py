"""End-to-end score benchmark using mirin collection and Logistic."""

from __future__ import annotations

import argparse
import gc
import sys
import time
from typing import Any

import torch

import probelab as pl


def log(*args: object) -> None:
    sys.stdout.write(" ".join(str(x) for x in args) + "\n")
    sys.stdout.flush()


def _load_collection_deps() -> tuple[Any, Any, Any, Any]:
    try:
        import mirin as mi
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from probelab.collection.mirin import collect_activations
    except ImportError as exc:
        raise SystemExit(
            "This benchmark requires optional collection dependencies. "
            "Install with `uv sync --extra collection` or `pip install "
            "probelab[collection]`."
        ) from exc
    return mi, AutoModelForCausalLM, AutoTokenizer, collect_activations


def _mask_from_name(name: str) -> pl.masks.Mask:
    masks = {
        "all": pl.masks.all,
        "assistant": pl.masks.assistant,
        "user": pl.masks.user,
    }
    return masks[name]()


def _maybe_sample(dataset: pl.Dataset, n: int, *, seed: int) -> pl.Dataset:
    if n <= 0 or n >= len(dataset):
        return dataset
    return dataset.sample(n, stratified=len(set(dataset.labels)) > 1, seed=seed)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="google/gemma-3-27b-it")
    parser.add_argument("--dataset", default="wildguard_mix")
    parser.add_argument("--samples", type=int, default=0, help="0 uses full splits")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--layer", type=int, default=40)
    parser.add_argument(
        "--mask",
        choices=["all", "assistant", "user"],
        default="all",
        help="Detection mask used before mean pooling",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--probe-c", type=float, default=0.01)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    mi, AutoModelForCausalLM, AutoTokenizer, collect_activations = (
        _load_collection_deps()
    )
    mask = _mask_from_name(args.mask)

    log("Loading datasets...")
    t0 = time.perf_counter()
    full_train = _maybe_sample(
        pl.datasets.load(args.dataset),
        args.samples,
        seed=args.seed,
    )
    train_dataset, val_dataset = full_train.split(0.8, stratified=True, seed=args.seed)
    test_dataset = _maybe_sample(
        pl.datasets.load(args.dataset, split="test"),
        args.samples,
        seed=args.seed,
    )
    log(f"Datasets loaded in {time.perf_counter() - t0:.1f}s")
    log(
        f"Train: {len(train_dataset)}, Val: {len(val_dataset)}, "
        f"Test: {len(test_dataset)}"
    )

    log(f"\nLoading tokenizer: {args.model}")
    t0 = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    log(f"Tokenizer loaded in {time.perf_counter() - t0:.1f}s")

    log("\nTokenizing train...")
    t0 = time.perf_counter()
    train_tokens = pl.tokenize_dataset(train_dataset, tokenizer, mask=mask)
    log(
        f"Train tokenized in {time.perf_counter() - t0:.1f}s: "
        f"{len(train_tokens)} samples, {train_tokens.total_tokens:,} tokens, "
        f"max_seq={train_tokens.seq_len}"
    )

    log("Tokenizing test...")
    t0 = time.perf_counter()
    test_tokens = pl.tokenize_dataset(test_dataset, tokenizer, mask=mask)
    log(
        f"Test tokenized in {time.perf_counter() - t0:.1f}s: "
        f"{len(test_tokens)} samples, {test_tokens.total_tokens:,} tokens, "
        f"max_seq={test_tokens.seq_len}"
    )

    log("\nLoading model...")
    t0 = time.perf_counter()
    hf_model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    ).eval()
    for param in hf_model.parameters():
        param.requires_grad = False
    model = mi.Model(hf_model, rename=mi.renames.llm, tokenizer=tokenizer)
    n_layers = hf_model.config.num_hidden_layers
    layer = min(args.layer, n_layers - 1)
    log(f"Model loaded in {time.perf_counter() - t0:.1f}s; using layer {layer}")

    log("\nCollecting + pooling train activations...")
    t0 = time.perf_counter()
    train_prepared = collect_activations(
        model,
        train_tokens,
        layers=[layer],
        batch_size=args.batch_size,
        pool="mean",
        progress=True,
        progress_desc="train collect+pool",
    )
    elapsed = time.perf_counter() - t0
    log(
        f"Train done in {elapsed:.0f}s "
        f"({len(train_tokens) / elapsed:.1f} samples/s): "
        f"{tuple(train_prepared.data.shape)}"
    )

    log("Collecting + pooling test activations...")
    t0 = time.perf_counter()
    test_prepared = collect_activations(
        model,
        test_tokens,
        layers=[layer],
        batch_size=args.batch_size,
        pool="mean",
        progress=True,
        progress_desc="test collect+pool",
    )
    elapsed = time.perf_counter() - t0
    log(f"Test done in {elapsed:.0f}s: {tuple(test_prepared.data.shape)}")

    log("Freeing model...")
    close = getattr(model, "close", None)
    if callable(close):
        close()
    del model
    del hf_model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    probe_device = "cuda" if torch.cuda.is_available() else "cpu"
    log(f"\nTraining logistic probe (C={args.probe_c}, device={probe_device})...")
    t0 = time.perf_counter()
    logistic = pl.probes.Logistic(C=args.probe_c, device=probe_device).fit(
        train_prepared,
        train_dataset.labels,
    )
    log(f"Logistic trained in {time.perf_counter() - t0:.1f}s")

    log("\nResults:")
    train_probs = logistic.predict(train_prepared)
    test_probs = logistic.predict(test_prepared)
    train_auroc = pl.metrics.auroc(train_dataset.labels, train_probs)
    train_f1 = pl.metrics.f1(train_dataset.labels, train_probs)
    test_f1 = pl.metrics.f1(test_dataset.labels, test_probs)
    test_auroc = pl.metrics.auroc(test_dataset.labels, test_probs)
    log(
        "  logistic: "
        f"Train F1={train_f1:.3f}, Train AUROC={train_auroc:.3f}, "
        f"Test F1={test_f1:.3f}, Test AUROC={test_auroc:.3f}"
    )


if __name__ == "__main__":
    main()
