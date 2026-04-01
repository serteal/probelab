"""Benchmark comparing probelab backends: transformers vs mirin vs mirin_server.

Usage:
    uv run python perf_checks/backend_benchmark.py [--model MODEL] [--n-samples N]
"""

from __future__ import annotations

import argparse
import gc
import json
import statistics
import time
from dataclasses import dataclass

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

import probelab as pl
from probelab.processing.tokenization import Tokens


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

@dataclass
class BenchResult:
    backend: str
    layers: list[int]
    batch_size: int
    n_samples: int
    times_s: list[float]
    activations_shape: tuple[int, ...]
    activations_dims: str

    @property
    def median_s(self) -> float:
        return statistics.median(self.times_s)

    @property
    def mean_s(self) -> float:
        return statistics.mean(self.times_s)

    @property
    def std_s(self) -> float:
        return statistics.stdev(self.times_s) if len(self.times_s) > 1 else 0.0

    @property
    def throughput(self) -> float:
        return self.n_samples / self.median_s if self.median_s > 0 else 0.0


def make_synthetic_tokens(
    tokenizer,
    n_samples: int,
    min_len: int = 32,
    max_len: int = 128,
) -> Tokens:
    """Create synthetic tokenized data with variable lengths."""
    dialogues = []
    labels = []
    for i in range(n_samples):
        # Generate variable-length prompts
        repeat = min_len + (i * 7 % (max_len - min_len))
        text = f"Sample {i}: " + "the quick brown fox jumps over the lazy dog. " * (repeat // 10 + 1)
        dialogues.append([
            pl.types.Message(role=pl.types.Role.USER, content=text),
            pl.types.Message(role=pl.types.Role.ASSISTANT, content=f"Response {i}. " * 3),
        ])
        labels.append(pl.types.Label(i % 2))
    ds = pl.datasets.base.Dataset(dialogues=dialogues, labels=labels, name="synthetic")
    tokens = pl.tokenize_dataset(ds, tokenizer, mask=pl.masks.assistant())
    return tokens


def sync_cuda():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def collect_with_backend(
    model_obj,
    tokens: Tokens,
    layers: list[int],
    batch_size: int,
    backend: str = "",
) -> pl.Activations:
    """Run collect_activations."""
    return pl.collect_activations(
        model_obj,
        tokens,
        layers=layers,
        batch_size=batch_size,
    )


def benchmark_one(
    model_obj,
    tokens: Tokens,
    layers: list[int],
    batch_size: int,
    backend: str,
    n_warmup: int = 1,
    n_runs: int = 5,
) -> BenchResult:
    """Benchmark a single configuration."""
    # Warmup
    for _ in range(n_warmup):
        acts = collect_with_backend(model_obj, tokens, layers, batch_size, backend)
        sync_cuda()
        del acts
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    times = []
    last_acts = None
    for _ in range(n_runs):
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        sync_cuda()

        t0 = time.perf_counter()
        acts = collect_with_backend(model_obj, tokens, layers, batch_size, backend)
        sync_cuda()
        t1 = time.perf_counter()

        times.append(t1 - t0)
        last_acts = acts

    return BenchResult(
        backend=backend,
        layers=layers,
        batch_size=batch_size,
        n_samples=len(tokens),
        times_s=times,
        activations_shape=tuple(last_acts.data.shape),
        activations_dims=last_acts.dims,
    )


def validate_correctness(
    hf_model,
    ti_model,
    ti_server,
    tokens: Tokens,
    layers: list[int],
    batch_size: int,
) -> dict[str, float]:
    """Validate that all backends produce numerically close results."""
    acts_tf = collect_with_backend(hf_model, tokens, layers, batch_size, "transformers")
    acts_ti = collect_with_backend(ti_model, tokens, layers, batch_size, "mirin")

    # Pool to get comparable [batch, hidden] tensors
    tf_pooled = acts_tf.mean("s").data
    ti_pooled = acts_ti.mean("s").data

    diffs = {"mirin_vs_transformers": (tf_pooled - ti_pooled).abs().max().item()}

    if ti_server is not None:
        acts_srv = collect_with_backend(ti_server, tokens, layers, batch_size, "mirin_server")
        srv_pooled = acts_srv.mean("s").data
        diffs["server_vs_transformers"] = (tf_pooled - srv_pooled).abs().max().item()

    return diffs


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Backend benchmark for probelab")
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct", help="HF model name")
    parser.add_argument("--n-samples", type=int, default=64, help="Number of synthetic samples")
    parser.add_argument("--n-runs", type=int, default=5, help="Timed runs per config")
    parser.add_argument("--n-warmup", type=int, default=2, help="Warmup runs")
    parser.add_argument("--batch-sizes", type=str, default="8,16,32", help="Comma-separated batch sizes")
    parser.add_argument("--json-output", type=str, default=None, help="Optional JSON output path")
    parser.add_argument("--skip-server", action="store_true", help="Skip mirin server backend")
    args = parser.parse_args()

    batch_sizes = [int(x) for x in args.batch_sizes.split(",")]

    print(f"{'='*72}")
    print(f"Backend Benchmark: {args.model}")
    print(f"  Samples: {args.n_samples}, Runs: {args.n_runs}, Warmup: {args.n_warmup}")
    print(f"  Batch sizes: {batch_sizes}")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  Device: {device}")
    if torch.cuda.is_available():
        print(f"  GPU: {torch.cuda.get_device_name()}")
    print(f"{'='*72}\n")

    # Load model and tokenizer
    print("Loading model and tokenizer...")
    t0 = time.perf_counter()
    hf_model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    print(f"  Model loaded in {time.perf_counter() - t0:.1f}s")

    # Wrap with mirin
    import mirin as mi

    print("Creating mirin wrappers...")
    ti_model = mi.Model(hf_model, rename=mi.renames.llm, tokenizer=tokenizer)

    ti_server = None
    if not args.skip_server:
        ti_server = mi.Server(hf_model, rename=mi.renames.llm, tokenizer=tokenizer)

    # Create synthetic data
    print("Generating synthetic tokens...")
    tokens = make_synthetic_tokens(tokenizer, args.n_samples)
    print(f"  {len(tokens)} samples, token lengths: "
          f"min={tokens.lengths.min().item()}, "
          f"max={tokens.lengths.max().item()}, "
          f"mean={tokens.lengths.float().mean().item():.0f}")

    # Determine layers to test
    n_layers = hf_model.config.num_hidden_layers
    layer_configs = [
        [n_layers // 2],                                           # single mid layer
        [n_layers // 4, n_layers // 2, 3 * n_layers // 4],        # 3 layers
    ]

    # Validate correctness first
    print("\nValidating numerical correctness...")
    for layer_cfg in layer_configs:
        diffs = validate_correctness(hf_model, ti_model, ti_server, tokens, layer_cfg, batch_sizes[0])
        for name, diff in diffs.items():
            status = "PASS" if diff < 1e-3 else ("WARN" if diff < 1e-1 else "FAIL")
            print(f"  layers={layer_cfg} {name}: max_diff={diff:.2e} [{status}]")

    # Run benchmarks
    print("\nRunning benchmarks...\n")

    backends = ["transformers", "mirin"]
    backend_models = {"transformers": hf_model, "mirin": ti_model}
    if ti_server is not None:
        backends.append("mirin_server")
        backend_models["mirin_server"] = ti_server

    all_results: list[BenchResult] = []

    for layer_cfg in layer_configs:
        layer_label = f"layers={layer_cfg}"
        print(f"--- {layer_label} ---")

        for bs in batch_sizes:
            for backend_name in backends:
                model_obj = backend_models[backend_name]
                print(f"  {backend_name:25s} bs={bs:3d} ... ", end="", flush=True)
                result = benchmark_one(
                    model_obj, tokens, layer_cfg, bs,
                    backend=backend_name,
                    n_warmup=args.n_warmup,
                    n_runs=args.n_runs,
                )
                all_results.append(result)
                print(
                    f"median={result.median_s:.3f}s  "
                    f"mean={result.mean_s:.3f}s  "
                    f"std={result.std_s:.3f}s  "
                    f"throughput={result.throughput:.0f} samples/s"
                )
            print()

    # Summary table
    print(f"\n{'='*72}")
    print("SUMMARY TABLE")
    print(f"{'='*72}")
    print(f"{'Backend':25s} {'Layers':20s} {'BS':>4s} {'Median(s)':>10s} {'Throughput':>12s} {'vs TF':>8s}")
    print("-" * 83)

    for result in all_results:
        layer_str = str(result.layers)
        # Find transformers baseline for same config
        tf_baseline = next(
            (r for r in all_results
             if r.backend == "transformers" and r.layers == result.layers and r.batch_size == result.batch_size),
            None,
        )
        speedup = ""
        if tf_baseline and result.backend != "transformers":
            ratio = tf_baseline.median_s / result.median_s
            speedup = f"{ratio:.2f}x"

        print(
            f"{result.backend:25s} {layer_str:20s} {result.batch_size:4d} "
            f"{result.median_s:10.3f} {result.throughput:10.0f}/s {speedup:>8s}"
        )

    # JSON output
    if args.json_output:
        json_data = [
            {
                "backend": r.backend,
                "layers": r.layers,
                "batch_size": r.batch_size,
                "n_samples": r.n_samples,
                "median_s": r.median_s,
                "mean_s": r.mean_s,
                "std_s": r.std_s,
                "throughput": r.throughput,
                "times_s": r.times_s,
                "activations_shape": list(r.activations_shape),
                "activations_dims": r.activations_dims,
            }
            for r in all_results
        ]
        with open(args.json_output, "w") as f:
            json.dump(json_data, f, indent=2)
        print(f"\nResults saved to {args.json_output}")

    # Cleanup
    if ti_server is not None:
        ti_server.close()
    ti_model.close()

    print("\nDone.")


if __name__ == "__main__":
    main()
