import argparse
import logging
from typing import Any, Dict, List, Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from utils import TimingResult, measure_with_warmup, timer

import probelab as pl
from probelab import Pipeline
from probelab.masks import (
    after,
    assistant,
    between,
    contains,
    nth_message,
    regex,
    user,
)
from probelab.transforms import Pool, SelectLayer

torch.set_float32_matmul_precision("high")
pl.logger.setLevel(logging.WARNING)


def benchmark_probe_training_with_masks(
    model: Any,
    tokenizer: Any,
    dataset: pl.datasets.DialogueDataset,
    layers: List[int],
    batch_size: int = 8,
    max_samples: Optional[int] = None,
) -> Dict[str, TimingResult]:
    """Benchmark probe training performance with different mask configurations."""
    results = {}

    # Limit samples if requested
    if max_samples:
        dataset = dataset[:max_samples]

    # Test different mask configurations
    mask_configs = [
        ("default", None, "Default mask (assistant messages)"),
        ("assistant_only", assistant(), "Explicit assistant mask"),
        ("user_only", user(), "User messages only"),
        # New position-based masks
        ("between_tags", between("<", ">", inclusive=False), "Between angle brackets"),
        ("after_question", after("?", inclusive=False), "After question mark"),
    ]

    for mask_name, mask, description in mask_configs:
        print(f"\nTesting mask: {description}")

        def train_probe_with_mask():
            # Collect activations
            acts = pl.collect_activations(
                model=model,
                tokenizer=tokenizer,
                data=dataset,
                layers=layers,
                mask=mask,
                batch_size=batch_size,
                streaming=True,
            )

            pipeline = Pipeline([
                ("select", SelectLayer(layers[0])),
                ("agg", Pool(dim="sequence", method="mean")),
                ("probe", pl.probes.Logistic()),
            ])
            pipeline.fit_streaming(acts, dataset.labels)
            return pipeline

        mask_result = measure_with_warmup(
            train_probe_with_mask,
            warmup_runs=1,
            measurement_runs=3,
            name=f"Mask: {mask_name}",
        )
        results[f"mask_{mask_name}"] = mask_result

    # Also test multiple probes with complex mask using new masks
    def train_multiple_probes_complex_mask():
        # Complex mask combining new and old masks
        complex_mask = (
            (nth_message(-1) & contains("not"))
            | (user() & regex(r"\?$"))
            | (assistant() & after("The", inclusive=False))
        )

        # Collect activations
        acts = pl.collect_activations(
            model=model,
            tokenizer=tokenizer,
            data=dataset,
            layers=layers,
            mask=complex_mask,
            batch_size=batch_size,
            streaming=True,
        )

        pipelines = {
            "logistic": Pipeline([
                ("select", SelectLayer(layers[0])),
                ("agg", Pool(dim="sequence", method="mean")),
                ("probe", pl.probes.Logistic()),
            ]),
            "mlp": Pipeline([
                ("select", SelectLayer(layers[0])),
                ("agg", Pool(dim="sequence", method="mean")),
                ("probe", pl.probes.MLP()),
            ]),
            "attention": Pipeline([
                ("select", SelectLayer(layers[0])),
                ("probe", pl.probes.Attention()),
            ]),
        }

        for name, pipeline in pipelines.items():
            pipeline.fit_streaming(acts, dataset.labels)
        return pipelines

    multi_probe_result = measure_with_warmup(
        train_multiple_probes_complex_mask,
        warmup_runs=1,
        measurement_runs=3,
        name="Multiple probes with complex mask",
    )
    results["multi_probe_complex_mask"] = multi_probe_result

    return results


def print_summary(
    results: Dict[str, TimingResult],
    num_samples: int = 0,
    baseline_time: Optional[float] = None,
):
    """Print a summary of benchmark results."""
    print("\n" + "=" * 60)
    print("BENCHMARK SUMMARY")
    print("=" * 60)

    print(f"\nDataset size: {num_samples} samples")

    if baseline_time:
        print(f"Baseline throughput: {num_samples / baseline_time:.1f} samples/sec")

    print("\nMask Performance Results:")
    print("-" * 40)

    # Sort results by throughput
    sorted_results = sorted(
        results.items(), key=lambda x: num_samples / x[1].mean, reverse=True
    )

    for name, result in sorted_results:
        throughput = num_samples / result.mean
        display_name = name.replace("mask_", "").replace("_", " ")

        print(f"\n  {display_name}:")
        print(f"    Time: {result.mean:.3f}s ± {result.std:.3f}s")
        print(f"    Throughput: {throughput:.1f} samples/sec")

        if baseline_time:
            baseline_throughput = num_samples / baseline_time
            perf_ratio = throughput / baseline_throughput
            if perf_ratio >= 0.95:
                status = "✓ PASS"
                color = "\033[92m"  # Green
            elif perf_ratio >= 0.90:
                status = "⚠ WARNING"
                color = "\033[93m"  # Yellow
            else:
                status = "✗ FAIL"
                color = "\033[91m"  # Red

            print(
                f"    Performance: {color}{perf_ratio:.1%} of baseline {status}\033[0m"
            )

    # Memory usage if CUDA available
    if torch.cuda.is_available():
        print("\nGPU Memory:")
        print("-" * 40)
        print(f"  Allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
        print(f"  Reserved: {torch.cuda.memory_reserved() / 1e9:.2f} GB")
        print(f"  Max allocated: {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark probelab masking system performance"
    )
    parser.add_argument(
        "--model",
        default="google/gemma-2-2b-it",
        help="Model to use for benchmarking",
    )
    parser.add_argument(
        "--layers",
        type=int,
        nargs="+",
        default=[12, 14, 16],
        help="Layers to extract activations from",
    )
    parser.add_argument(
        "--batch-size", type=int, default=32, help="Batch size for processing"
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=200,
        help="Maximum number of samples to use",
    )
    parser.add_argument(
        "--baseline-time",
        type=float,
        default=None,
        help="Baseline time in seconds for performance comparison (e.g., 0.87 for 230 samples/sec)",
    )

    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load model and tokenizer
    with timer("Model loading"):
        tokenizer = AutoTokenizer.from_pretrained(args.model)
        model = AutoModelForCausalLM.from_pretrained(
            args.model,
            torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32,
            device_map=device,
        )
        model.eval()

    print(
        f"Model loaded: {model.config.num_hidden_layers} layers, "
        f"{model.config.hidden_size} hidden size"
    )

    # Load datasets
    with timer("Dataset loading"):
        harmful_dataset = pl.datasets.CircuitBreakersDataset()[: args.max_samples // 2]
        benign_dataset = pl.datasets.BenignInstructionsDataset()[
            : args.max_samples // 2
        ]
        dataset = harmful_dataset + benign_dataset

    print(f"\nDataset loaded: {len(dataset)} samples")

    # Run benchmark with different masks
    mask_results = benchmark_probe_training_with_masks(
        model=model,
        tokenizer=tokenizer,
        dataset=dataset,
        layers=args.layers,
        batch_size=args.batch_size,
        max_samples=None,  # Already limited above
    )

    # Calculate baseline if not provided
    if args.baseline_time is None:
        # Use the default mask result as baseline
        if "mask_default" in mask_results:
            args.baseline_time = mask_results["mask_default"].mean
            print(f"\nUsing default mask as baseline: {args.baseline_time:.3f}s")

    # Print summary
    print_summary(
        results=mask_results,
        num_samples=len(dataset),
        baseline_time=args.baseline_time,
    )


if __name__ == "__main__":
    main()
