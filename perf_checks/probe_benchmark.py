import argparse
import gc
import logging
from typing import Any, Dict, List, Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from utils import TimingResult, measure_with_warmup, timer

import probelab as pl
from probelab.models import HookedModel
from probelab.processing import tokenize_dataset

torch.set_float32_matmul_precision("high")
pl.logger.setLevel(logging.WARNING)


def benchmark_activation_collection(
    model: Any,
    tokenizer: Any,
    dataset: pl.datasets.DialogueDataset,
    layers: List[int],
    batch_size: int = 8,
    max_samples: Optional[int] = None,
) -> Dict[str, TimingResult]:
    """Benchmark activation collection performance."""
    results = {}

    # Limit samples if requested
    if max_samples:
        dataset = dataset[:max_samples]

    # Measure tokenization
    def tokenize_fn():
        return tokenize_dataset(tokenizer=tokenizer, dataset=dataset)

    tokenize_result = measure_with_warmup(
        tokenize_fn, warmup_runs=1, measurement_runs=3, name="Tokenization"
    )
    results["tokenization"] = tokenize_result

    # Get tokenized inputs for activation collection
    inputs = tokenize_dataset(dataset, tokenizer)

    # Measure activation collection with HookedModel
    def collect_with_hooks():
        with HookedModel(model, layers=layers) as hooked_model:
            activations = []

            # Process in batches
            for start_idx in range(0, len(inputs["input_ids"]), batch_size):
                end_idx = min(start_idx + batch_size, len(inputs["input_ids"]))
                batch_inputs = {
                    k: v[start_idx:end_idx].to(model.device) for k, v in inputs.items()
                }

                # Get activations
                batch_acts = hooked_model.get_activations(batch_inputs)
                # Use non_blocking for async GPU->CPU transfer
                activations.append(batch_acts.to("cpu", non_blocking=True))

            return torch.cat(activations, dim=1)

    hook_result = measure_with_warmup(
        collect_with_hooks,
        warmup_runs=1,
        measurement_runs=3,
        name="Activation Collection (HookedModel - all positions)",
    )
    results["activation_collection_hooks"] = hook_result

    # Measure using the high-level collect_activations function (batch mode)
    def collect_high_level_batch():
        return pl.collect_activations(
            model=model,
            tokenizer=tokenizer,
            data=dataset,
            layers=layers,
            mask=pl.masks.assistant(),
            batch_size=batch_size,
            streaming=False,
            verbose=False,
        )

    high_level_batch_result = measure_with_warmup(
        collect_high_level_batch,
        warmup_runs=1,
        measurement_runs=3,
        name="Activation Collection (collect_activations - batch mode)",
    )
    results["activation_collection_high_level_batch"] = high_level_batch_result

    # Measure streaming mode
    def collect_high_level_streaming():
        activation_iter = pl.collect_activations(
            model=model,
            tokenizer=tokenizer,
            data=dataset,
            layers=layers,
            mask=pl.masks.assistant(),
            batch_size=batch_size,
            streaming=True,
            verbose=False,
        )

        # Consume the iterator to measure total time
        all_batches = []
        for batch in activation_iter:
            all_batches.append(batch)

        return all_batches

    high_level_streaming_result = measure_with_warmup(
        collect_high_level_streaming,
        warmup_runs=1,
        measurement_runs=3,
        name="Activation Collection (streaming mode)",
    )
    results["activation_collection_high_level_streaming"] = high_level_streaming_result

    return results


def benchmark_probe_training(
    model: Any,
    tokenizer: Any,
    dataset: pl.datasets.DialogueDataset,
    layers: List[int],
    batch_size: int = 8,
    max_samples: Optional[int] = None,
) -> Dict[str, TimingResult]:
    """Benchmark probe training performance."""
    results = {}

    # Limit samples if requested
    if max_samples:
        dataset = dataset[:max_samples]

    layer = layers[0]

    # === FIRST: Benchmark with PRE-COLLECTED activations (probe training only) ===
    print("\n" + "=" * 60)
    print("Pre-collecting activations for probe-only benchmarks...")
    print("=" * 60)

    # Collect activations once
    pre_collected = pl.collect_activations(
        model=model,
        tokenizer=tokenizer,
        data=dataset,
        layers=layers,
        mask=pl.masks.assistant(),
        batch_size=batch_size,
        verbose=True,
    )

    # Prepare pooled activations
    pre_collected_pooled = pre_collected.select("l", layer).mean("s")
    pre_collected_tokens = pre_collected.select("l", layer)

    # Benchmark probe training ONLY with pre-collected pooled activations
    def train_probe_only_pooled():
        probe = pl.probes.Logistic().fit(pre_collected_pooled, dataset.labels)
        return probe

    probe_only_pooled_result = measure_with_warmup(
        train_probe_only_pooled,
        warmup_runs=1,
        measurement_runs=5,
        name="Probe Training ONLY (pre-collected pooled activations)",
    )
    results["probe_only_pooled"] = probe_only_pooled_result

    # Benchmark probe training ONLY with pre-collected dense activations
    def train_probe_only_dense():
        # Pool at training time
        prepared = pre_collected_tokens.pool("sequence", "mean")
        probe = pl.probes.Logistic().fit(prepared, dataset.labels)
        return probe

    probe_only_dense_result = measure_with_warmup(
        train_probe_only_dense,
        warmup_runs=1,
        measurement_runs=5,
        name="Probe Training ONLY (pre-collected dense -> pool at train)",
    )
    results["probe_only_dense"] = probe_only_dense_result

    # Benchmark 10 probes with pre-collected activations
    def train_10_probes_only():
        probes = {}
        for i in range(10):
            probes[f"logistic_{i}"] = pl.probes.Logistic().fit(pre_collected_pooled, dataset.labels)
        return probes

    probe_10_only_result = measure_with_warmup(
        train_10_probes_only,
        warmup_runs=1,
        measurement_runs=5,
        name="10 Probes Training ONLY (pre-collected pooled)",
    )
    results["probe_10_only"] = probe_10_only_result

    # === NEXT: Full pipeline benchmarks (activation collection + training) ===

    def train_full_logistic_pooled():
        acts = pl.collect_activations(
            model=model,
            tokenizer=tokenizer,
            data=dataset,
            layers=layers,
            mask=pl.masks.assistant(),
            batch_size=batch_size,
            verbose=False,
        )
        prepared = acts.select("l", layer).mean("s")
        probe = pl.probes.Logistic().fit(prepared, dataset.labels)
        return probe

    gpu_logistic_mean_result = measure_with_warmup(
        train_full_logistic_pooled,
        warmup_runs=1,
        measurement_runs=3,
        name="Full: Logistic with MEAN pooling",
    )
    results["gpu_logistic_mean_pooling"] = gpu_logistic_mean_result

    def train_full_logistic_tokens():
        acts = pl.collect_activations(
            model=model,
            tokenizer=tokenizer,
            data=dataset,
            layers=layers,
            mask=pl.masks.assistant(),
            batch_size=batch_size,
            verbose=False,
        )
        prepared = acts.select("l", layer)  # Keep tokens
        probe = pl.probes.Logistic().fit(prepared, dataset.labels)
        return probe

    gpu_logistic_token_result = measure_with_warmup(
        train_full_logistic_tokens,
        warmup_runs=1,
        measurement_runs=3,
        name="Full: Logistic token-level (no pooling)",
    )
    results["gpu_logistic_token_level"] = gpu_logistic_token_result

    def train_full_mlp_pooled():
        acts = pl.collect_activations(
            model=model,
            tokenizer=tokenizer,
            data=dataset,
            layers=layers,
            mask=pl.masks.assistant(),
            batch_size=batch_size,
            verbose=False,
        )
        prepared = acts.select("l", layer).mean("s")
        probe = pl.probes.MLP().fit(prepared, dataset.labels)
        return probe

    mlp_mean_result = measure_with_warmup(
        train_full_mlp_pooled,
        warmup_runs=1,
        measurement_runs=3,
        name="Full: MLP with MEAN pooling",
    )
    results["mlp_mean_pooling"] = mlp_mean_result

    def train_full_attention():
        acts = pl.collect_activations(
            model=model,
            tokenizer=tokenizer,
            data=dataset,
            layers=layers,
            mask=pl.masks.assistant(),
            batch_size=batch_size,
            verbose=False,
        )
        prepared = acts.select("l", layer)  # Keep tokens for attention
        probe = pl.probes.Attention().fit(prepared, dataset.labels)
        return probe

    attention_result = measure_with_warmup(
        train_full_attention,
        warmup_runs=1,
        measurement_runs=3,
        name="Full: Attention Probe",
    )
    results["attention_full"] = attention_result

    def train_10_probes_full():
        # Collect activations once
        acts = pl.collect_activations(
            model=model,
            tokenizer=tokenizer,
            data=dataset,
            layers=layers,
            mask=pl.masks.assistant(),
            batch_size=batch_size,
            verbose=False,
        )
        prepared = acts.select("l", layer).mean("s")

        probes = {}
        for i in range(10):
            probes[f"logistic_{i}"] = pl.probes.Logistic().fit(prepared, dataset.labels)
        return probes

    logistic_10_probes_result = measure_with_warmup(
        train_10_probes_full,
        warmup_runs=1,
        measurement_runs=3,
        name="Full: 10 Logistic Probes (collect once)",
    )
    results["logistic_10_probes_full"] = logistic_10_probes_result

    return results


def print_summary(
    activation_results: Dict[str, TimingResult] | None = None,
    training_results: Dict[str, TimingResult] | None = None,
    num_samples: int = 0,
):
    """Print a summary of benchmark results."""
    if activation_results:
        print("Activation Collection:")
        print("-" * 40)
        for name, result in activation_results.items():
            throughput = num_samples / result.mean
            display_name = name.replace("activation_collection_", "").replace("_", " ")
            print(f"  {display_name}:")
            print(f"    Time: {result}")
            print(f"    Throughput: {throughput:.1f} samples/sec")

    if training_results:
        print("\nProbe Training:")
        print("-" * 40)
        for name, result in training_results.items():
            throughput = num_samples / result.mean
            print(f"  {name}:")
            print(f"    Time: {result}")
            print(f"    Throughput: {throughput:.1f} samples/sec")

    # Memory usage if CUDA available
    if torch.cuda.is_available():
        print("\nGPU Memory:")
        print("-" * 40)
        print(f"  Allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
        print(f"  Reserved: {torch.cuda.memory_reserved() / 1e9:.2f} GB")
        print(f"  Max allocated: {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")


def main():
    parser = argparse.ArgumentParser(description="Benchmark probelab performance")
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
        "--add-activation-benchmark",
        action="store_true",
        help="Also benchmark activation collection",
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

    # Benchmark activation collection
    activation_results = None
    if args.add_activation_benchmark:
        activation_results = benchmark_activation_collection(
            model=model,
            tokenizer=tokenizer,
            dataset=dataset,
            layers=args.layers,
            batch_size=args.batch_size,
            max_samples=None,
        )

        # Clear cache before training benchmark
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()

    # Benchmark probe training
    training_results = benchmark_probe_training(
        model=model,
        tokenizer=tokenizer,
        dataset=dataset,
        layers=args.layers,
        batch_size=args.batch_size,
        max_samples=None,
    )

    # Print summary
    print_summary(
        activation_results=activation_results,
        training_results=training_results,
        num_samples=len(dataset),
    )


if __name__ == "__main__":
    main()
