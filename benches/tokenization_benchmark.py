"""Benchmark tokenization and mask evaluation performance."""

from __future__ import annotations

import argparse
import logging
from typing import Any

import torch
from transformers import AutoTokenizer

try:
    from .utils import TimingResult, measure_with_warmup
except ImportError:  # pragma: no cover - supports `python benches/...`
    from utils import TimingResult, measure_with_warmup

import probelab as pl
from probelab.logger import logger as probelab_logger
from probelab.tokenization import tokenize_dataset

torch.set_float32_matmul_precision("high")
probelab_logger.setLevel(logging.WARNING)


def _sample_dataset(dataset: pl.Dataset, samples: int, *, seed: int) -> pl.Dataset:
    if samples <= 0 or samples >= len(dataset):
        return dataset
    has_both_classes = len(set(dataset.labels)) > 1
    return dataset.sample(samples, stratified=has_both_classes, seed=seed)


def _sample_exact(dataset: pl.Dataset, samples: int, *, seed: int) -> pl.Dataset:
    if samples <= 0:
        return dataset[:0]
    return _sample_dataset(dataset, samples, seed=seed)


def load_benchmark_dataset(name: str, samples: int, *, seed: int = 42) -> pl.Dataset:
    """Load a dataset using the public dataset registry."""
    if name == "harmfulness_pair":
        if samples <= 0:
            return (
                pl.datasets.load("circuit_breakers")
                + pl.datasets.load("benign_instructions")
            ).shuffle(seed)
        n_pos = samples // 2 if samples > 0 else 0
        n_neg = samples - n_pos if samples > 0 else 0
        pos = _sample_exact(
            pl.datasets.load("circuit_breakers"),
            n_pos,
            seed=seed,
        )
        neg = _sample_exact(
            pl.datasets.load("benign_instructions"),
            n_neg,
            seed=seed,
        )
        return (pos + neg).shuffle(seed)

    return _sample_dataset(pl.datasets.load(name), samples, seed=seed)


def _tokenize(
    tokenizer: Any,
    dataset: pl.Dataset,
    mask: pl.masks.Mask,
    *,
    device: str,
    chunk_size: int,
) -> pl.Tokens:
    return tokenize_dataset(
        dataset=dataset,
        tokenizer=tokenizer,
        mask=mask,
        device=device,
        chunk_size=chunk_size,
    )


def benchmark_tokenization_baseline(
    tokenizer: Any,
    dataset: pl.Dataset,
    *,
    device: str = "cpu",
    chunk_size: int = 1024,
) -> dict[str, TimingResult]:
    """Benchmark the current public tokenization API with the cheapest mask."""

    def tokenize_all_mask() -> pl.Tokens:
        return _tokenize(
            tokenizer,
            dataset,
            pl.masks.all(),
            device=device,
            chunk_size=chunk_size,
        )

    result = measure_with_warmup(
        tokenize_all_mask,
        warmup_runs=2,
        measurement_runs=10,
        name="Tokenization + mask=all",
    )
    return {"all": result}


def benchmark_mask_evaluation(
    tokenizer: Any,
    dataset: pl.Dataset,
    *,
    device: str = "cpu",
    chunk_size: int = 1024,
) -> dict[str, TimingResult]:
    """Benchmark tokenization with different mask types."""
    results = {}

    mask_configs = [
        ("none", pl.masks.none(), "No tokens selected"),
        ("all", pl.masks.all(), "All non-padding tokens selected"),
        ("assistant", pl.masks.assistant(), "Assistant messages"),
        ("user", pl.masks.user(), "User messages"),
        ("system", pl.masks.system(), "System messages"),
        (
            "assistant_no_pad",
            pl.masks.assistant(include_padding=False),
            "Assistant content without template padding",
        ),
        ("contains_the", pl.masks.contains("the"), "Contains 'the'"),
        ("contains_i", pl.masks.contains("I"), "Contains 'I'"),
        ("after_colon", pl.masks.after(":"), "After colon"),
        ("before_period", pl.masks.before("."), "Before period"),
        ("between_quotes", pl.masks.between('"', '"'), "Between quotes"),
        ("special_tokens", pl.masks.special_tokens(), "Special tokens"),
        ("nth_first", pl.masks.nth_message(0), "First message"),
        ("nth_last", pl.masks.nth_message(-1), "Last message"),
        (
            "complex_and",
            pl.masks.assistant() & pl.masks.contains("I"),
            "Assistant AND contains 'I'",
        ),
        (
            "complex_or",
            pl.masks.user() | pl.masks.assistant(),
            "User OR assistant",
        ),
        ("complex_not", ~pl.masks.user(), "NOT user"),
        (
            "complex_nested",
            (pl.masks.assistant() & pl.masks.after(":"))
            | (pl.masks.user() & pl.masks.contains("?")),
            "Nested role/text mask",
        ),
    ]

    for mask_name, mask, description in mask_configs:

        def tokenize_with_mask(mask=mask) -> pl.Tokens:
            return _tokenize(
                tokenizer,
                dataset,
                mask,
                device=device,
                chunk_size=chunk_size,
            )

        results[mask_name] = measure_with_warmup(
            tokenize_with_mask,
            warmup_runs=1,
            measurement_runs=5,
            name=f"{mask_name}: {description}",
        )

    return results


def analyze_scaling(
    tokenizer: Any,
    base_dataset: pl.Dataset,
    sizes: list[int],
    *,
    device: str = "cpu",
    chunk_size: int = 1024,
) -> dict[int, dict[str, TimingResult]]:
    """Analyze how performance scales with dataset size."""
    scaling_results = {}

    test_masks = {
        "all": pl.masks.all(),
        "assistant": pl.masks.assistant(),
        "complex": pl.masks.assistant() & pl.masks.contains("I"),
        "between": pl.masks.between("<", ">"),
    }

    for size in sizes:
        print(f"\n{'=' * 60}")
        print(f"Testing with {size} samples")
        print(f"{'=' * 60}")

        dataset = base_dataset[:size]
        size_results = {}
        for mask_name, mask in test_masks.items():

            def tokenize_fn(mask=mask) -> pl.Tokens:
                return _tokenize(
                    tokenizer,
                    dataset,
                    mask,
                    device=device,
                    chunk_size=chunk_size,
                )

            size_results[mask_name] = measure_with_warmup(
                tokenize_fn,
                warmup_runs=1,
                measurement_runs=3,
                name=f"{mask_name} ({size} samples)",
            )

        scaling_results[size] = size_results

    return scaling_results


def print_results(
    tokenization_results: dict[str, TimingResult],
    mask_results: dict[str, TimingResult],
    scaling_results: dict[int, dict[str, TimingResult]] | None = None,
    *,
    num_samples: int = 0,
    baseline_name: str = "all",
) -> None:
    """Print formatted benchmark results."""
    print("\n" + "=" * 80)
    print("TOKENIZATION PERFORMANCE BENCHMARK")
    print("=" * 80)

    if tokenization_results:
        print("\nBaseline Tokenization:")
        print("-" * 40)
        for name, result in tokenization_results.items():
            throughput = result.throughput(num_samples)
            print(f"  {name}: {throughput:.1f} samples/sec ({result.mean:.3f}s)")

    if mask_results:
        print("\nMask Evaluation Performance:")
        print("-" * 40)

        baseline = mask_results.get(baseline_name) or tokenization_results.get(
            baseline_name
        )
        baseline_throughput = baseline.throughput(num_samples) if baseline else None

        sorted_results = sorted(
            mask_results.items(),
            key=lambda x: x[1].throughput(num_samples),
            reverse=True,
        )

        for name, result in sorted_results:
            throughput = result.throughput(num_samples)
            time_str = f"{result.mean:.3f}s +/- {result.std:.3f}s"

            if baseline_throughput:
                ratio = throughput / baseline_throughput
                status = "ok" if ratio >= 0.8 else "slow"
                print(
                    f"  {name:20s}: {throughput:7.1f} samples/sec | "
                    f"{time_str:20s} | {ratio:5.1%} {status}"
                )
            else:
                print(f"  {name:20s}: {throughput:7.1f} samples/sec | {time_str}")

    if scaling_results:
        print("\nScaling Analysis:")
        print("-" * 40)

        sizes = sorted(scaling_results.keys())
        mask_names = list(next(iter(scaling_results.values())).keys())

        print(f"  {'Size':<10} ", end="")
        for mask_name in mask_names:
            print(f"{mask_name:>15} ", end="")
        print()

        for size in sizes:
            print(f"  {size:<10} ", end="")
            for mask_name in mask_names:
                result = scaling_results[size][mask_name]
                throughput = result.throughput(size)
                print(f"{throughput:>14.1f} ", end="")
            print()

        if len(sizes) > 1:
            print("\nScaling Efficiency (linear = 1.0):")
            base_size = sizes[0]
            for mask_name in mask_names:
                print(f"  {mask_name}:")
                base_throughput = scaling_results[base_size][mask_name].throughput(
                    base_size
                )

                for size in sizes[1:]:
                    result = scaling_results[size][mask_name]
                    actual_throughput = result.throughput(size)
                    efficiency = actual_throughput / base_throughput
                    print(f"    {base_size}->{size}: {efficiency:.2f}")


def _parse_sizes(raw: str) -> list[int]:
    return [int(part.strip()) for part in raw.split(",") if part.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark tokenization and mask performance"
    )
    parser.add_argument(
        "--model",
        default="google/gemma-2-2b-it",
        help="Tokenizer to use",
    )
    parser.add_argument(
        "--dataset",
        default="harmfulness_pair",
        help=(
            "Dataset registry name, or 'harmfulness_pair' for "
            "circuit_breakers + benign_instructions"
        ),
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=200,
        help="Number of samples for the main benchmark; 0 uses the full dataset",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=1024,
        help="tokenize_dataset chunk_size",
    )
    parser.add_argument(
        "--scaling",
        action="store_true",
        help="Run scaling analysis",
    )
    parser.add_argument(
        "--scaling-sizes",
        default="50,100,200,500,1000",
        help="Comma-separated sample counts for --scaling",
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to place token tensors on",
    )

    args = parser.parse_args()

    print(f"Loading tokenizer: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    print(f"Loading dataset {args.dataset!r} with {args.samples} samples...")
    dataset = load_benchmark_dataset(args.dataset, args.samples)
    print(f"Dataset loaded: {len(dataset)} samples")

    print("\nRunning tokenization baseline...")
    tokenization_results = benchmark_tokenization_baseline(
        tokenizer,
        dataset,
        device=args.device,
        chunk_size=args.chunk_size,
    )

    print("\nRunning mask evaluation benchmark...")
    mask_results = benchmark_mask_evaluation(
        tokenizer,
        dataset,
        device=args.device,
        chunk_size=args.chunk_size,
    )

    scaling_results = None
    if args.scaling:
        print("\nRunning scaling analysis...")
        sizes = _parse_sizes(args.scaling_sizes)
        max_size = max(sizes)
        large_dataset = load_benchmark_dataset(args.dataset, max_size)
        sizes = [size for size in sizes if size <= len(large_dataset)]
        if sizes:
            scaling_results = analyze_scaling(
                tokenizer,
                large_dataset,
                sizes,
                device=args.device,
                chunk_size=args.chunk_size,
            )

    print_results(
        tokenization_results,
        mask_results,
        scaling_results,
        num_samples=len(dataset),
    )


if __name__ == "__main__":
    main()
