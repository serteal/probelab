"""Profile mirin activation collection, pooling, and probe scoring."""

from __future__ import annotations

import argparse
import gc
import statistics
import time
from typing import Any

import torch

import probelab as pl
from probelab import pool as P


def _load_collection_deps() -> tuple[Any, Any, Any, Any, Any]:
    try:
        import mirin as mi
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from probelab.collection.mirin import collect_activations, stream_activations
    except ImportError as exc:
        raise SystemExit(
            "This benchmark requires optional collection dependencies. "
            "Install with `uv sync --extra collection` or `pip install "
            "probelab[collection]`."
        ) from exc
    return mi, AutoModelForCausalLM, AutoTokenizer, collect_activations, stream_activations


def summarize_lengths(tokens: pl.Tokens, name: str) -> None:
    lengths = tokens.lengths.float().cpu()
    if lengths.numel() == 0:
        print(f"{name}: empty tokens")
        return
    quantiles = torch.quantile(lengths, torch.tensor([0.5, 0.9, 0.95, 0.99]))
    print(
        f"{name}: samples={len(tokens)}, total_tokens={tokens.total_tokens:,}, "
        f"max_seq={tokens.seq_len}, p50={int(quantiles[0])}, "
        f"p90={int(quantiles[1])}, p95={int(quantiles[2])}, "
        f"p99={int(quantiles[3])}"
    )


def _stats_str(values: list[float]) -> str:
    if not values:
        return "n/a"
    return (
        f"mean={statistics.mean(values):.4f}s, "
        f"p50={statistics.median(values):.4f}s, max={max(values):.4f}s"
    )


def _maybe_sample(dataset: pl.Dataset, n: int, *, seed: int) -> pl.Dataset:
    if n <= 0 or n >= len(dataset):
        return dataset
    return dataset.sample(n, stratified=len(set(dataset.labels)) > 1, seed=seed)


def _mask_from_name(name: str) -> pl.masks.Mask:
    masks = {
        "all": pl.masks.all,
        "user": pl.masks.user,
        "assistant": pl.masks.assistant,
    }
    return masks[name]()


def profile_collect_with_breakdown(
    model: object,
    tokens: pl.Tokens,
    layer: int,
    batch_size: int,
    *,
    stream_activations: Any,
    pool_name: str = "mean",
    hook_point: str = "block",
    sort_by_length: bool = True,
    batch_token_budget: int | None = None,
) -> tuple[pl.Activations, dict[str, float | int | str]]:
    pool_fn = getattr(P, pool_name, None)
    if pool_fn is None:
        raise ValueError(f"Unknown pool function: {pool_name}")

    n = len(tokens)
    out: torch.Tensor | None = None

    extract_times: list[float] = []
    pool_times: list[float] = []
    batch_token_counts: list[int] = []
    batch_sample_counts: list[int] = []

    iterator = iter(
        stream_activations(
            model,
            tokens,
            layers=[layer],
            batch_size=batch_size,
            hook_point=hook_point,
            sort_by_length=sort_by_length,
            batch_token_budget=batch_token_budget,
        )
    )
    while True:
        extract_start = time.perf_counter()
        try:
            chunk = next(iterator)
        except StopIteration:
            break
        extract_times.append(time.perf_counter() - extract_start)

        batch_chunk = int(chunk.offsets.shape[0] - 1)
        batch_tokens = int(chunk.offsets[-1].item()) if chunk.offsets.numel() else 0
        batch_sample_counts.append(batch_chunk)
        batch_token_counts.append(batch_tokens)

        if out is None:
            out = torch.zeros(
                n,
                chunk.data.shape[1],
                chunk.data.shape[-1],
                dtype=chunk.data.dtype,
                device=chunk.data.device,
            )

        pool_start = time.perf_counter()
        pooled = pool_fn(
            chunk.data,
            chunk.detection_mask,
            offsets=chunk.offsets,
        )
        out_idx = torch.tensor(chunk.indices, dtype=torch.long, device=pooled.device)
        out[out_idx] = pooled
        pool_times.append(time.perf_counter() - pool_start)

    if out is None:
        out = torch.zeros(n, 1, 0, dtype=torch.float32)

    prepared = pl.Activations.from_tensor(out.squeeze(1), dims="bh")
    total_extract = sum(extract_times)
    total_pool = sum(pool_times)
    total = total_extract + total_pool

    metrics: dict[str, float | int | str] = {
        "num_batches": len(extract_times),
        "total_extract_s": total_extract,
        "total_pool_s": total_pool,
        "total_s": total,
        "extract_share": (100.0 * total_extract / total) if total > 0 else 0.0,
        "pool_share": (100.0 * total_pool / total) if total > 0 else 0.0,
        "extract_stats": _stats_str(extract_times),
        "pool_stats": _stats_str(pool_times),
        "mean_batch_tokens": int(sum(batch_token_counts) / len(batch_token_counts))
        if batch_token_counts
        else 0,
        "max_batch_tokens": max(batch_token_counts) if batch_token_counts else 0,
        "mean_batch_samples": int(sum(batch_sample_counts) / len(batch_sample_counts))
        if batch_sample_counts
        else 0,
    }
    return prepared, metrics


def _print_profile_metrics(
    label: str,
    prepared: pl.Activations,
    tokens: pl.Tokens,
    elapsed: float,
    metrics: dict[str, float | int | str],
) -> None:
    print(
        f"{label} prepared shape={tuple(prepared.data.shape)} in {elapsed:.2f}s, "
        f"{len(tokens) / elapsed:.2f} samples/s, "
        f"{tokens.total_tokens / elapsed:,.0f} tok/s"
    )
    print(
        f"{label} breakdown: "
        f"extract={metrics['total_extract_s']:.2f}s "
        f"({metrics['extract_share']:.1f}%), "
        f"pool={metrics['total_pool_s']:.2f}s ({metrics['pool_share']:.1f}%), "
        f"batches={metrics['num_batches']}, "
        f"batch_tokens(mean/max)="
        f"{metrics['mean_batch_tokens']}/{metrics['max_batch_tokens']}"
    )
    print(f"  extract per-batch: {metrics['extract_stats']}")
    print(f"  pool per-batch:    {metrics['pool_stats']}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        default="meta-llama/Llama-3.2-1B-Instruct",
        help="HF model id",
    )
    parser.add_argument(
        "--dataset",
        default="wildguard_mix",
        help="Dataset name for pl.datasets.load",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=1000,
        help="Sample size for train and test; 0 uses full splits",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Activation extraction batch size",
    )
    parser.add_argument(
        "--batch-token-budget",
        type=int,
        default=None,
        help="Maximum padded tokens per collection batch",
    )
    parser.add_argument(
        "--layer",
        type=int,
        default=40,
        help="Requested layer index (clipped to model max)",
    )
    parser.add_argument(
        "--hook-point",
        default="block",
        help="mirin hook point passed to collect_activations/stream_activations",
    )
    parser.add_argument(
        "--mask",
        choices=["all", "user", "assistant"],
        default="all",
        help="Detection mask",
    )
    parser.add_argument(
        "--pool",
        choices=["mean", "max", "last_token"],
        default="mean",
        help="Pooling function used for stream breakdown and collect API compare",
    )
    parser.add_argument(
        "--no-sort-by-length",
        action="store_true",
        help="Disable length-sorted activation collection batches",
    )
    parser.add_argument(
        "--truncation",
        action="store_true",
        help="Enable tokenizer truncation",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=2048,
        help="Tokenizer max_length when --truncation is set",
    )
    parser.add_argument(
        "--tokenize-chunk-size",
        type=int,
        default=1024,
        help="tokenize_dataset chunk_size",
    )
    parser.add_argument(
        "--compare-collect-api",
        action="store_true",
        help="Also run collect_activations(pool=...) once for comparison",
    )
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    mi, AutoModelForCausalLM, AutoTokenizer, collect_activations, stream_activations = (
        _load_collection_deps()
    )
    mask = _mask_from_name(args.mask)
    sort_by_length = not args.no_sort_by_length

    print("Loading datasets...")
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
    print(f"Datasets loaded in {time.perf_counter() - t0:.2f}s")
    print(
        f"Train={len(train_dataset)} Val={len(val_dataset)} Test={len(test_dataset)}"
    )

    print("\nLoading tokenizer...")
    t0 = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    print(f"Tokenizer loaded in {time.perf_counter() - t0:.2f}s")

    tokenize_kwargs = {"chunk_size": args.tokenize_chunk_size}
    if args.truncation:
        tokenize_kwargs["truncation"] = True
        tokenize_kwargs["max_length"] = args.max_length

    print("\nTokenizing train...")
    t0 = time.perf_counter()
    train_tokens = pl.tokenize_dataset(
        train_dataset,
        tokenizer,
        mask=mask,
        **tokenize_kwargs,
    )
    print(f"Train tokenized in {time.perf_counter() - t0:.2f}s")
    summarize_lengths(train_tokens, "train")

    print("Tokenizing test...")
    t0 = time.perf_counter()
    test_tokens = pl.tokenize_dataset(
        test_dataset,
        tokenizer,
        mask=mask,
        **tokenize_kwargs,
    )
    print(f"Test tokenized in {time.perf_counter() - t0:.2f}s")
    summarize_lengths(test_tokens, "test")

    print("\nLoading model...")
    t0 = time.perf_counter()
    hf_model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    ).eval()
    for p in hf_model.parameters():
        p.requires_grad = False
    model = mi.Model(hf_model, rename=mi.renames.llm, tokenizer=tokenizer)
    print(f"Model loaded in {time.perf_counter() - t0:.2f}s")

    first_param = next(hf_model.parameters())
    n_layers = hf_model.config.num_hidden_layers
    layer = min(args.layer, n_layers - 1)
    print(
        f"Model placement: first_param={first_param.dtype} on {first_param.device}, "
        f"num_layers={n_layers}, using_layer={layer}"
    )
    if hasattr(hf_model, "hf_device_map"):
        print(f"hf_device_map entries: {len(hf_model.hf_device_map)}")

    print("\nProfiling train activation collection (stream + pooled)...")
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    train_prepared, train_metrics = profile_collect_with_breakdown(
        model=model,
        tokens=train_tokens,
        layer=layer,
        batch_size=args.batch_size,
        stream_activations=stream_activations,
        pool_name=args.pool,
        hook_point=args.hook_point,
        sort_by_length=sort_by_length,
        batch_token_budget=args.batch_token_budget,
    )
    elapsed = time.perf_counter() - t0
    _print_profile_metrics("Train", train_prepared, train_tokens, elapsed, train_metrics)
    if torch.cuda.is_available():
        print(
            "Peak GPU alloc (train collect): "
            f"{torch.cuda.max_memory_allocated() / 1e9:.2f} GB"
        )

    print("\nProfiling test activation collection (stream + pooled)...")
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    test_prepared, test_metrics = profile_collect_with_breakdown(
        model=model,
        tokens=test_tokens,
        layer=layer,
        batch_size=args.batch_size,
        stream_activations=stream_activations,
        pool_name=args.pool,
        hook_point=args.hook_point,
        sort_by_length=sort_by_length,
        batch_token_budget=args.batch_token_budget,
    )
    elapsed = time.perf_counter() - t0
    _print_profile_metrics("Test", test_prepared, test_tokens, elapsed, test_metrics)
    if torch.cuda.is_available():
        print(
            "Peak GPU alloc (test collect): "
            f"{torch.cuda.max_memory_allocated() / 1e9:.2f} GB"
        )

    if args.compare_collect_api:
        print(f"\nComparing against collect_activations(pool={args.pool!r}) on train...")
        t0 = time.perf_counter()
        _ = collect_activations(
            model,
            train_tokens,
            layers=[layer],
            batch_size=args.batch_size,
            pool=args.pool,
            hook_point=args.hook_point,
            sort_by_length=sort_by_length,
            batch_token_budget=args.batch_token_budget,
        )
        print(f"collect_activations() elapsed: {time.perf_counter() - t0:.2f}s")

    print("\nFreeing model...")
    close = getattr(model, "close", None)
    if callable(close):
        close()
    del model
    del hf_model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print("Training logistic probe...")
    t0 = time.perf_counter()
    probe = pl.probes.Logistic(
        C=0.01,
        device="cuda" if torch.cuda.is_available() else "cpu",
    )
    probe.fit(train_prepared, train_dataset.labels)
    print(f"Probe fit in {time.perf_counter() - t0:.3f}s")

    print("Evaluating...")
    t0 = time.perf_counter()
    train_probs = probe.predict(train_prepared)
    test_probs = probe.predict(test_prepared)
    print(
        "Metrics: "
        f"train_f1={pl.metrics.f1(train_dataset.labels, train_probs):.3f}, "
        f"train_auroc={pl.metrics.auroc(train_dataset.labels, train_probs):.3f}, "
        f"test_f1={pl.metrics.f1(test_dataset.labels, test_probs):.3f}, "
        f"test_auroc={pl.metrics.auroc(test_dataset.labels, test_probs):.3f}"
    )
    print(f"Eval in {time.perf_counter() - t0:.3f}s")


if __name__ == "__main__":
    main()
