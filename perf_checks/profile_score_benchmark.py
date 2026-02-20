import argparse
import gc
import statistics
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

import probelab as pl
from probelab import pool as P
from probelab.processing.activations import stream_activations


def summarize_lengths(tokens: pl.processing.Tokens, name: str) -> None:
    lengths = tokens.lengths.float().cpu()
    if lengths.numel() == 0:
        print(f"{name}: empty tokens")
        return
    quantiles = torch.quantile(lengths, torch.tensor([0.5, 0.9, 0.95, 0.99]))
    print(
        f"{name}: samples={len(tokens)}, total_tokens={tokens.total_tokens:,}, "
        f"max_seq={tokens.seq_len}, p50={int(quantiles[0])}, p90={int(quantiles[1])}, "
        f"p95={int(quantiles[2])}, p99={int(quantiles[3])}"
    )


def _stats_str(values: list[float]) -> str:
    if not values:
        return "n/a"
    return (
        f"mean={statistics.mean(values):.4f}s, p50={statistics.median(values):.4f}s, "
        f"max={max(values):.4f}s"
    )


def profile_collect_with_breakdown(
    model: object,
    tokens: pl.processing.Tokens,
    layer: int,
    batch_size: int,
    pool_name: str = "mean",
) -> tuple[pl.Activations, dict[str, float | int | str]]:
    pool_fn = getattr(P, pool_name)

    n = len(tokens)
    out: torch.Tensor | None = None

    extract_times: list[float] = []
    pool_times: list[float] = []
    batch_token_counts: list[int] = []
    batch_sample_counts: list[int] = []

    it = iter(
        stream_activations(model, tokens, layers=[layer], batch_size=batch_size)
    )
    while True:
        extract_start = time.perf_counter()
        try:
            flat_data, det, offsets, idx = next(it)
        except StopIteration:
            break
        extract_times.append(time.perf_counter() - extract_start)

        batch_chunk = int(offsets.shape[0] - 1)
        batch_tokens = int(offsets[-1].item()) if offsets.numel() else 0
        batch_sample_counts.append(batch_chunk)
        batch_token_counts.append(batch_tokens)

        if out is None:
            out = torch.zeros(
                n, 1, flat_data.shape[-1], dtype=flat_data.dtype, device=flat_data.device
            )

        pool_start = time.perf_counter()
        pooled = pool_fn(flat_data[:, 0, :], det, offsets=offsets)
        out_idx = torch.tensor(idx, dtype=torch.long, device=pooled.device)
        out[out_idx, 0] = pooled
        pool_times.append(time.perf_counter() - pool_start)

    if out is None:
        out = torch.zeros(n, 1, 0, dtype=torch.float32)

    prepared = pl.Activations(data=out.squeeze(1), dims="bh", layers=None)
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


def main() -> None:
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
        help="Sample size for train and test",
    )
    parser.add_argument(
        "--batch-size", type=int, default=8, help="Activation extraction batch size"
    )
    parser.add_argument(
        "--layer",
        type=int,
        default=40,
        help="Requested layer index (clipped to model max)",
    )
    parser.add_argument(
        "--mask",
        choices=["all", "user", "assistant"],
        default="all",
        help="Detection mask",
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
        "--compare-collect-api",
        action="store_true",
        help="Also run pl.collect_activations(pool='mean') once for comparison",
    )
    args = parser.parse_args()

    mask_map = {
        "all": pl.masks.all,
        "user": pl.masks.user,
        "assistant": pl.masks.assistant,
    }
    mask = mask_map[args.mask]()

    print("Loading datasets...")
    t0 = time.perf_counter()
    full_train = pl.datasets.load(args.dataset).sample(args.samples, stratified=True)
    train_dataset, val_dataset = full_train.split(0.8, stratified=True)
    test_dataset = pl.datasets.load(args.dataset, split="test").sample(
        args.samples, stratified=True
    )
    print(f"Datasets loaded in {time.perf_counter() - t0:.2f}s")
    print(
        f"Train={len(train_dataset)} Val={len(val_dataset)} Test={len(test_dataset)}"
    )

    print("\nLoading tokenizer...")
    t0 = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    print(f"Tokenizer loaded in {time.perf_counter() - t0:.2f}s")

    tokenize_kwargs = {}
    if args.truncation:
        tokenize_kwargs["truncation"] = True
        tokenize_kwargs["max_length"] = args.max_length

    print("\nTokenizing train...")
    t0 = time.perf_counter()
    train_tokens = pl.tokenize_dataset(train_dataset, tokenizer, mask=mask, **tokenize_kwargs)
    print(f"Train tokenized in {time.perf_counter() - t0:.2f}s")
    summarize_lengths(train_tokens, "train")

    print("Tokenizing test...")
    t0 = time.perf_counter()
    test_tokens = pl.tokenize_dataset(test_dataset, tokenizer, mask=mask, **tokenize_kwargs)
    print(f"Test tokenized in {time.perf_counter() - t0:.2f}s")
    summarize_lengths(test_tokens, "test")

    print("\nLoading model...")
    t0 = time.perf_counter()
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        dtype=torch.bfloat16,
        device_map="auto",
    ).eval()
    for p in model.parameters():
        p.requires_grad = False
    print(f"Model loaded in {time.perf_counter() - t0:.2f}s")

    first_param = next(model.parameters())
    n_layers = model.config.num_hidden_layers
    layer = min(args.layer, n_layers - 1)
    print(
        f"Model placement: first_param={first_param.dtype} on {first_param.device}, "
        f"num_layers={n_layers}, using_layer={layer}"
    )
    if hasattr(model, "hf_device_map"):
        print(f"hf_device_map entries: {len(model.hf_device_map)}")

    print("\nProfiling train activation collection (stream + pooled)...")
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    train_prepared, train_metrics = profile_collect_with_breakdown(
        model=model,
        tokens=train_tokens,
        layer=layer,
        batch_size=args.batch_size,
        pool_name="mean",
    )
    elapsed = time.perf_counter() - t0
    print(
        f"Train prepared shape={tuple(train_prepared.data.shape)} in {elapsed:.2f}s, "
        f"{len(train_tokens)/elapsed:.2f} samples/s, {train_tokens.total_tokens/elapsed:,.0f} tok/s"
    )
    print(
        "Train breakdown: "
        f"extract={train_metrics['total_extract_s']:.2f}s ({train_metrics['extract_share']:.1f}%), "
        f"pool={train_metrics['total_pool_s']:.2f}s ({train_metrics['pool_share']:.1f}%), "
        f"batches={train_metrics['num_batches']}, "
        f"batch_tokens(mean/max)={train_metrics['mean_batch_tokens']}/{train_metrics['max_batch_tokens']}"
    )
    print(f"  extract per-batch: {train_metrics['extract_stats']}")
    print(f"  pool per-batch:    {train_metrics['pool_stats']}")
    if torch.cuda.is_available():
        print(f"Peak GPU alloc (train collect): {torch.cuda.max_memory_allocated()/1e9:.2f} GB")

    print("\nProfiling test activation collection (stream + pooled)...")
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    test_prepared, test_metrics = profile_collect_with_breakdown(
        model=model,
        tokens=test_tokens,
        layer=layer,
        batch_size=args.batch_size,
        pool_name="mean",
    )
    elapsed = time.perf_counter() - t0
    print(
        f"Test prepared shape={tuple(test_prepared.data.shape)} in {elapsed:.2f}s, "
        f"{len(test_tokens)/elapsed:.2f} samples/s, {test_tokens.total_tokens/elapsed:,.0f} tok/s"
    )
    print(
        "Test breakdown: "
        f"extract={test_metrics['total_extract_s']:.2f}s ({test_metrics['extract_share']:.1f}%), "
        f"pool={test_metrics['total_pool_s']:.2f}s ({test_metrics['pool_share']:.1f}%), "
        f"batches={test_metrics['num_batches']}, "
        f"batch_tokens(mean/max)={test_metrics['mean_batch_tokens']}/{test_metrics['max_batch_tokens']}"
    )
    print(f"  extract per-batch: {test_metrics['extract_stats']}")
    print(f"  pool per-batch:    {test_metrics['pool_stats']}")
    if torch.cuda.is_available():
        print(f"Peak GPU alloc (test collect): {torch.cuda.max_memory_allocated()/1e9:.2f} GB")

    if args.compare_collect_api:
        print("\nComparing against pl.collect_activations(pool='mean') on train...")
        t0 = time.perf_counter()
        _ = pl.collect_activations(
            model, train_tokens, layers=[layer], batch_size=args.batch_size, pool="mean"
        )
        print(f"collect_activations() elapsed: {time.perf_counter() - t0:.2f}s")

    print("\nFreeing model...")
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print("Training logistic probe...")
    t0 = time.perf_counter()
    probe = pl.probes.Logistic(C=0.01, device="cuda" if torch.cuda.is_available() else "cpu")
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
