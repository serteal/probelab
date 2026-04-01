"""Large-dataset benchmark: server collector vs local backends.

Demonstrates how mirin.Server auto-batches and respects memory
budgets when extracting activations from thousands of samples, compared
with the local transformers and mirin backends.

Usage:
    uv run python perf_checks/large_dataset_benchmark.py [--model MODEL] [--n-samples N]
"""

from __future__ import annotations

import argparse
import gc
import statistics
import time
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

import mirin as mi

import probelab as pl
from probelab.processing.activations import Activations, collect_activations
from probelab.processing.tokenization import Tokens


def sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def gpu_mem_mb() -> float:
    if not torch.cuda.is_available():
        return 0.0
    return torch.cuda.max_memory_allocated() / (1024 * 1024)


def reset_peak_mem():
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()


def make_dataset(n: int) -> pl.datasets.base.Dataset:
    """Build a synthetic dataset with variable-length dialogues."""
    dialogues, labels = [], []
    for i in range(n):
        # Variable-length user messages (50–300 words)
        repeat = 5 + (i * 13 % 30)
        user_msg = (
            f"Question {i}: "
            + "Explain the significance of the following concept in modern science, "
            * repeat
        )
        # Variable-length assistant responses (20–120 words)
        resp_repeat = 2 + (i * 7 % 12)
        assistant_msg = (
            f"Answer {i}: "
            + "This is an important topic that requires careful consideration. "
            * resp_repeat
        )
        dialogues.append([
            pl.types.Message(role=pl.types.Role.USER, content=user_msg),
            pl.types.Message(role=pl.types.Role.ASSISTANT, content=assistant_msg),
        ])
        labels.append(pl.types.Label(i % 2))
    return pl.datasets.base.Dataset(dialogues=dialogues, labels=labels, name="synthetic_5k")


def collect_via_server_collector(
    server: mi.Server,
    tokens: Tokens,
    layers: list[int],
    token_budget: int | None,
) -> tuple[Activations, dict[str, Any]]:
    """Use the server's native collector interface for large-scale extraction.

    This is the 'proper' server path: compile a plan, open a collector with
    a token budget, and let it auto-chunk large batches. The server's
    scheduler handles admission control so batches stay within GPU memory.

    Instead of a fixed ``batch_size``, the caller only chooses a
    ``token_budget`` — the collector decides how many samples fit per
    forward pass based on their padded token count.  This naturally
    adapts to variable-length inputs without hand-tuned batch sizes.
    """
    layer_paths = [server._model.layers[l].path for l in layers]

    plan = server.compile(
        get=layer_paths,
        output={"activations": True, "logits": False, "activations_to_cpu": False},
    )
    collector = server.open_collector(
        plan=plan,
        stop_at_last_get=True,
        activations_to_cpu=False,
        token_budget=token_budget,
    )

    n = len(tokens)
    single_layer = len(layers) == 1

    # Sort by length for padding efficiency (same as other backends).
    lengths = tokens.lengths
    order = lengths.argsort(descending=True).tolist()

    per_sample: list[tuple[torch.Tensor, torch.Tensor] | None] = [None] * n
    n_server_chunks = 0

    # Feed length-sorted batches.  We pack as many samples as the token
    # budget allows (dynamic batching), rather than using a fixed batch size.
    cursor = 0
    while cursor < n:
        # Greedily pack samples until we hit the token budget.
        start = cursor
        cur_max_len = 0
        while cursor < n:
            sample_len = int(lengths[order[cursor]].item())
            new_max = max(cur_max_len, sample_len)
            new_count = cursor - start + 1
            padded_tokens = new_max * new_count
            # Always include at least one sample.
            if new_count > 1 and token_budget is not None and padded_tokens > token_budget:
                break
            cur_max_len = new_max
            cursor += 1

        idx_list = order[start:cursor]
        batch = tokens.pad_batch(idx_list, padding_side=tokens.padding_side)

        n_server_chunks += 1
        result = collector.collect_batch(batch)

        # result.activations[path] → [chunk_bs, seq, hidden]
        layer_acts = [result.activations[p] for p in layer_paths]
        acts_stacked = torch.stack(layer_acts, dim=2)  # [B, S, L, H]
        dev = acts_stacked.device

        attn_mask = batch["attention_mask"].to(dev).bool()
        det_mask = batch["detection_mask"].to(dev).bool()

        flat_data = acts_stacked[attn_mask]  # [T, L, H]
        flat_det = det_mask[attn_mask]

        b_size = attn_mask.shape[0]
        valid_lengths = attn_mask.sum(dim=1, dtype=torch.int64)
        offsets = torch.zeros(b_size + 1, dtype=torch.int64, device=dev)
        offsets[1:] = valid_lengths.cumsum(0)

        for j in range(offsets.shape[0] - 1):
            s, e = int(offsets[j]), int(offsets[j + 1])
            per_sample[idx_list[j]] = (flat_data[s:e], flat_det[s:e])

    collector.close()

    # Assemble into Activations
    all_data, all_det = [], []
    global_offsets = torch.zeros(n + 1, dtype=torch.int64)
    running = 0
    for i in range(n):
        if per_sample[i] is not None:
            d, dt = per_sample[i]
            all_data.append(d)
            all_det.append(dt)
            running += d.shape[0]
        global_offsets[i + 1] = running

    if all_data:
        cat_data = torch.cat(all_data, dim=0)
        cat_det = torch.cat(all_det, dim=0)
        if cat_data.is_cuda:
            cat_data = cat_data.cpu()
            cat_det = cat_det.cpu()
    else:
        cat_data = torch.zeros(0, len(layers), 0, dtype=torch.float32)
        cat_det = torch.empty(0, dtype=torch.bool)

    if single_layer:
        if cat_data.ndim == 3:
            cat_data = cat_data.squeeze(1)
        acts = Activations(data=cat_data, dims="bsh", offsets=global_offsets, det=cat_det, layers=None)
    else:
        acts = Activations(data=cat_data, dims="blsh", offsets=global_offsets, det=cat_det, layers=tuple(layers))

    meta = {
        "n_server_chunks": n_server_chunks,
        "token_budget": token_budget,
    }
    return acts, meta


def validate(acts_ref: Activations, acts_test: Activations, name: str) -> float:
    """Check numerical agreement between two Activations."""
    ref = acts_ref.mean("s").data
    test = acts_test.mean("s").data
    diff = (ref - test).abs().max().item()
    status = "PASS" if diff < 1e-3 else "FAIL"
    print(f"  {name}: max_diff={diff:.2e} [{status}]")
    return diff


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--n-samples", type=int, default=5000)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--layers", type=str, default="14", help="Comma-separated layer indices")
    parser.add_argument("--n-warmup", type=int, default=2)
    parser.add_argument("--n-runs", type=int, default=5)
    parser.add_argument("--token-budgets", type=str, default="8192,32768,131072",
                        help="Comma-separated token budgets for server collector")
    args = parser.parse_args()

    layers = [int(x) for x in args.layers.split(",")]
    token_budgets = [int(x) for x in args.token_budgets.split(",")]
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"{'=' * 80}")
    print(f"Large-Dataset Benchmark")
    print(f"  Model: {args.model}")
    print(f"  Samples: {args.n_samples}, Batch size: {args.batch_size}")
    print(f"  Layers: {layers}")
    print(f"  Runs: {args.n_runs}, Warmup: {args.n_warmup}")
    print(f"  Server token budgets: {token_budgets}")
    print(f"  Device: {device}")
    if torch.cuda.is_available():
        print(f"  GPU: {torch.cuda.get_device_name()}")
        print(f"  GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    print(f"{'=' * 80}\n")

    # Load model
    print("Loading model...")
    t0 = time.perf_counter()
    hf_model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, device_map="auto",
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    n_layers = hf_model.config.num_hidden_layers
    print(f"  Loaded in {time.perf_counter() - t0:.1f}s ({n_layers} layers, "
          f"hidden={hf_model.config.hidden_size})")

    # Create wrappers
    ti_model = mi.Model(hf_model, rename=mi.renames.llm, tokenizer=tokenizer)
    server = mi.Server(hf_model, rename=mi.renames.llm, tokenizer=tokenizer)

    # Build dataset
    print(f"\nBuilding synthetic dataset ({args.n_samples} samples)...")
    t0 = time.perf_counter()
    ds = make_dataset(args.n_samples)
    print(f"  Built in {time.perf_counter() - t0:.1f}s")

    print("Tokenizing...")
    t0 = time.perf_counter()
    tokens = pl.tokenize_dataset(ds, tokenizer, mask=pl.masks.assistant())
    lens = tokens.lengths
    total_tokens = int(lens.sum().item())
    print(f"  Tokenized in {time.perf_counter() - t0:.1f}s")
    print(f"  Token lengths: min={lens.min().item()}, max={lens.max().item()}, "
          f"mean={lens.float().mean().item():.0f}, total={total_tokens:,}")

    # -----------------------------------------------------------------------
    # Warmup all backends
    # -----------------------------------------------------------------------
    print("\nWarming up all backends...")
    small_tokens = pl.tokenize_dataset(
        ds.sample(min(64, len(ds))), tokenizer, mask=pl.masks.assistant()
    )
    for _ in range(args.n_warmup):
        collect_activations(hf_model, small_tokens, layers=layers, batch_size=args.batch_size,
                            )
        collect_activations(ti_model, small_tokens, layers=layers, batch_size=args.batch_size,
                            )
        collect_activations(server, small_tokens, layers=layers, batch_size=args.batch_size,
                            )
        for tb in token_budgets:
            collect_via_server_collector(server, small_tokens, layers, token_budget=tb)
        sync()
    print("  Done.")

    # -----------------------------------------------------------------------
    # Validate correctness on a subset
    # -----------------------------------------------------------------------
    print("\nValidating correctness (256 samples)...")
    val_tokens = pl.tokenize_dataset(
        ds.sample(256, seed=42), tokenizer, mask=pl.masks.assistant()
    )
    acts_ref = collect_activations(hf_model, val_tokens, layers=layers,
                                   batch_size=args.batch_size)
    acts_ti = collect_activations(ti_model, val_tokens, layers=layers,
                                  batch_size=args.batch_size)
    acts_srv = collect_activations(server, val_tokens, layers=layers,
                                   batch_size=args.batch_size)
    validate(acts_ref, acts_ti, "mirin vs transformers")
    validate(acts_ref, acts_srv, "mirin_server vs transformers")
    for tb in token_budgets:
        acts_col, _ = collect_via_server_collector(server, val_tokens, layers, token_budget=tb)
        validate(acts_ref, acts_col, f"server collector (budget={tb:,}) vs transformers")
    del acts_ref, acts_ti, acts_srv, acts_col, val_tokens
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # -----------------------------------------------------------------------
    # Benchmark: full dataset
    # -----------------------------------------------------------------------
    print(f"\n{'=' * 80}")
    print(f"Benchmarking on {args.n_samples} samples")
    print(f"{'=' * 80}\n")

    configs: list[tuple[str, Any]] = [
        ("transformers (probelab)", lambda: collect_activations(
            hf_model, tokens, layers=layers, batch_size=args.batch_size)),
        ("mirin local (probelab)", lambda: collect_activations(
            ti_model, tokens, layers=layers, batch_size=args.batch_size)),
        ("mirin_server (probelab)", lambda: collect_activations(
            server, tokens, layers=layers, batch_size=args.batch_size)),
    ]
    for tb in token_budgets:
        configs.append((
            f"server collector (budget={tb:,})",
            lambda tb=tb: collect_via_server_collector(server, tokens, layers, token_budget=tb),
        ))

    results: list[dict[str, Any]] = []

    for name, run_fn in configs:
        print(f"  {name}...")

        # Warmup
        for _ in range(args.n_warmup):
            r = run_fn()
            sync()
            del r
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        times = []
        peak_mems = []
        meta = None

        for run_idx in range(args.n_runs):
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            reset_peak_mem()
            sync()
            t0 = time.perf_counter()
            result = run_fn()
            sync()
            elapsed = time.perf_counter() - t0
            peak = gpu_mem_mb()
            times.append(elapsed)
            peak_mems.append(peak)

            if isinstance(result, tuple):
                acts, meta = result
            else:
                acts = result

            del result, acts
            gc.collect()

        med = statistics.median(times)
        mean_mem = statistics.mean(peak_mems)
        throughput = args.n_samples / med
        tok_throughput = total_tokens / med

        info = {
            "name": name,
            "median_s": med,
            "min_s": min(times),
            "max_s": max(times),
            "throughput_samples": throughput,
            "throughput_tokens": tok_throughput,
            "peak_gpu_mb": mean_mem,
        }
        if meta:
            info.update(meta)
        results.append(info)

        extra = ""
        if meta:
            extra = f"  chunks={meta['n_server_chunks']}"
        print(f"    median={med:.2f}s  min={min(times):.2f}s  "
              f"throughput={throughput:.0f} samples/s  "
              f"({tok_throughput:.0f} tok/s)  "
              f"peak_gpu={mean_mem:.0f}MB{extra}")

    # -----------------------------------------------------------------------
    # Summary
    # -----------------------------------------------------------------------
    print(f"\n{'=' * 80}")
    print("SUMMARY")
    print(f"{'=' * 80}")

    baseline = results[0]["median_s"]

    print(f"\n{'Method':<40s} {'Median':>8s} {'Throughput':>14s} {'Tok/s':>12s} "
          f"{'Peak GPU':>10s} {'vs base':>8s} {'Chunks':>8s}")
    print("-" * 100)
    for r in results:
        speedup = baseline / r["median_s"]
        chunks = str(r.get("n_server_chunks", "-"))
        print(f"{r['name']:<40s} {r['median_s']:>7.2f}s "
              f"{r['throughput_samples']:>10.0f}/s "
              f"{r['throughput_tokens']:>10.0f}/s "
              f"{r['peak_gpu_mb']:>8.0f}MB "
              f"{speedup:>7.2f}x "
              f"{chunks:>8s}")

    # Server stats
    print(f"\nServer stats:")
    stats = server.stats()
    for key in ["collect_batch", "call", "compile"]:
        if key in stats.get("queues", {}):
            q = stats["queues"][key]
            print(f"  {key}: completed={q.get('completed', 0)}, "
                  f"avg_service={q.get('avg_service_ms', 0):.1f}ms, "
                  f"rejected={q.get('rejected', 0)}")

    ti_model.close()
    server.close()
    print("\nDone.")


if __name__ == "__main__":
    main()
