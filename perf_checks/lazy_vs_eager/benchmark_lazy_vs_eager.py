"""Compare lazy Acts vs eager Activations on realistic probe workloads.

This benchmark is designed for GPU agents and large datasets/models.
It runs each case in an isolated subprocess to capture peak memory fairly.

Workloads:
1) train: activation extraction + pooled probe training/eval on one layer.
2) sweep: multi-layer, multi-hyperparameter probe sweep.

Methods:
- eager: explicit eager collection baseline (full tensors allocated in host memory).
- lazy: Acts pipeline with lazy transforms + optional disk cache.
"""

from __future__ import annotations

import argparse
import contextlib
import dataclasses
import json
import os
import platform
import resource
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

import probelab as pl
from probelab import pool as P
from probelab.processing.activations import Activations, Axis, LayerMeta, SequenceMeta


MB = 1024.0 * 1024.0


@dataclasses.dataclass
class BenchResult:
    task: str
    method: str
    elapsed_s: float
    extract_s: float
    train_eval_s: float
    cpu_peak_rss_mb: float
    gpu_peak_alloc_mb: float
    gpu_peak_reserved_mb: float
    metric_name: str
    metric_value: float
    extra: dict[str, Any]


def _dtype_from_str(dtype_str: str) -> torch.dtype:
    lut = {
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float32": torch.float32,
        "fp32": torch.float32,
    }
    key = dtype_str.lower()
    if key not in lut:
        raise ValueError(f"Unknown dtype {dtype_str!r}. Expected one of: {sorted(lut)}")
    return lut[key]


def _cpu_peak_rss_mb() -> float:
    # macOS uses bytes, Linux uses KB.
    raw = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if platform.system() == "Darwin":
        return raw / MB
    return raw / 1024.0


def _reset_cuda_peaks() -> None:
    if not torch.cuda.is_available():
        return
    for dev in range(torch.cuda.device_count()):
        torch.cuda.reset_peak_memory_stats(dev)


def _read_cuda_peaks_mb() -> tuple[float, float]:
    if not torch.cuda.is_available():
        return 0.0, 0.0

    alloc = 0.0
    reserved = 0.0
    for dev in range(torch.cuda.device_count()):
        torch.cuda.synchronize(dev)
        alloc = max(alloc, torch.cuda.max_memory_allocated(dev) / MB)
        reserved = max(reserved, torch.cuda.max_memory_reserved(dev) / MB)
    return alloc, reserved


def _hidden_dim(model) -> int:
    cfg = model.config
    if hasattr(cfg, "hidden_size"):
        return int(cfg.hidden_size)
    if hasattr(cfg, "text_config") and hasattr(cfg.text_config, "hidden_size"):
        return int(cfg.text_config.hidden_size)
    raise ValueError(f"Cannot determine hidden dimension for {type(model)}")


def _pooled_sum(x: torch.Tensor, mask: torch.Tensor, *, dim: int) -> torch.Tensor:
    mask_bool = mask.to(x.device).bool()
    shape = [1] * x.ndim
    shape[0] = mask_bool.shape[0]
    shape[dim] = mask_bool.shape[1]
    return (x * mask_bool.view(shape).to(x.dtype)).sum(dim=dim)


def _pool_fn(name: str):
    n = name.lower()
    if n == "mean":
        return P.mean
    if n == "max":
        return P.max
    if n in {"last", "last_token"}:
        return P.last_token
    if n == "sum":
        return _pooled_sum
    raise ValueError(f"Unknown pool {name!r}. Expected mean|max|last|sum")


def _collect_eager(
    model,
    tokens,
    layers: list[int],
    *,
    batch_size: int,
    pool: str | None,
) -> Activations:
    """Explicit eager baseline mirroring old collect_activations behavior."""
    single_layer = len(layers) == 1
    n, max_seq, d = len(tokens), tokens.seq_len, _hidden_dim(model)

    if pool is not None:
        pool_fn = _pool_fn(pool)
        out = torch.zeros(n, len(layers), d, dtype=model.dtype)

        for acts_lbsd, idx, sl in pl.processing.stream_activations(model, tokens, layers, batch_size):
            acts_blsd = acts_lbsd.transpose(0, 1)
            if tokens.padding_side == "right":
                mask = tokens.detection_mask[idx, :sl]
            else:
                mask = tokens.detection_mask[idx, -sl:]
            out[idx] = pool_fn(acts_blsd, mask, dim=2)

        if single_layer:
            return Activations(
                activations=out.squeeze(1),
                axes=(Axis.BATCH, Axis.HIDDEN),
                layer_meta=None,
                sequence_meta=None,
                batch_indices=torch.arange(n, dtype=torch.long),
            )

        return Activations(
            activations=out,
            axes=(Axis.BATCH, Axis.LAYER, Axis.HIDDEN),
            layer_meta=LayerMeta(tuple(layers)),
            sequence_meta=None,
            batch_indices=torch.arange(n, dtype=torch.long),
        )

    out = torch.zeros(n, len(layers), max_seq, d, dtype=model.dtype)
    for acts_lbsd, idx, sl in pl.processing.stream_activations(model, tokens, layers, batch_size):
        acts_blsd = acts_lbsd.transpose(0, 1)
        if tokens.padding_side == "right":
            out[idx, :, :sl] = acts_blsd
        else:
            out[idx, :, -sl:] = acts_blsd

    seq_meta = SequenceMeta(
        attention_mask=tokens.attention_mask,
        detection_mask=tokens.detection_mask,
        input_ids=tokens.input_ids,
    )

    if single_layer:
        return Activations(
            activations=out.squeeze(1),
            axes=(Axis.BATCH, Axis.SEQ, Axis.HIDDEN),
            layer_meta=None,
            sequence_meta=seq_meta,
            batch_indices=torch.arange(n, dtype=torch.long),
        )

    return Activations(
        activations=out,
        axes=(Axis.BATCH, Axis.LAYER, Axis.SEQ, Axis.HIDDEN),
        layer_meta=LayerMeta(tuple(layers)),
        sequence_meta=seq_meta,
        batch_indices=torch.arange(n, dtype=torch.long),
    )


def _parse_csv_floats(s: str) -> list[float]:
    vals = []
    for raw in s.split(","):
        t = raw.strip()
        if not t:
            continue
        vals.append(float(t))
    if not vals:
        raise ValueError("Expected at least one numeric value")
    return vals


def _load_data_and_model(args: argparse.Namespace):
    dtype = _dtype_from_str(args.dtype)

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=dtype,
        device_map=args.device_map,
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token

    ds = pl.datasets.load(args.dataset).sample(args.samples, stratified=True)
    train_ds, test_ds = ds.split(args.train_frac, stratified=True)

    train_tokens = pl.tokenize_dataset(train_ds, tokenizer, mask=pl.masks.assistant())
    test_tokens = pl.tokenize_dataset(test_ds, tokenizer, mask=pl.masks.assistant())
    return model, train_ds, test_ds, train_tokens, test_tokens


def _run_train_case(args: argparse.Namespace, method: str) -> BenchResult:
    model, train_ds, test_ds, train_tokens, test_tokens = _load_data_and_model(args)
    layer = int(args.layer)

    t0 = time.perf_counter()
    t_extract0 = time.perf_counter()

    maybe_tmpdir = tempfile.TemporaryDirectory() if (method == "lazy" and args.lazy_cache_disk) else None
    ctx = maybe_tmpdir if maybe_tmpdir is not None else contextlib.nullcontext()
    with ctx as td:
        if method == "eager":
            train_x = _collect_eager(
                model,
                train_tokens,
                [layer],
                batch_size=args.batch_size,
                pool=args.pool,
            )
            test_x = _collect_eager(
                model,
                test_tokens,
                [layer],
                batch_size=args.batch_size,
                pool=args.pool,
            )
        elif method == "lazy":
            cache_train = Path(td) / "train_layer.h5" if td is not None else None
            cache_test = Path(td) / "test_layer.h5" if td is not None else None

            train_plan = pl.collect(
                model,
                train_tokens,
                layers=[layer],
                batch_size=args.batch_size,
                dtype=_dtype_from_str(args.dtype),
            )
            test_plan = pl.collect(
                model,
                test_tokens,
                layers=[layer],
                batch_size=args.batch_size,
                dtype=_dtype_from_str(args.dtype),
            )

            if args.pool:
                train_plan = getattr(train_plan, f"{args.pool}_pool")()
                test_plan = getattr(test_plan, f"{args.pool}_pool")()

            if args.lazy_cache_disk:
                assert cache_train is not None and cache_test is not None
                train_x = train_plan.cache(cache_train)
                test_x = test_plan.cache(cache_test)
            else:
                train_x = train_plan
                test_x = test_plan
        else:
            raise ValueError(f"Unknown method {method!r}")

        extract_s = time.perf_counter() - t_extract0

        t_train0 = time.perf_counter()
        probe = pl.probes.Logistic(C=args.train_c, max_iter=args.train_max_iter)
        probe.fit(train_x, train_ds.labels)
        probs = probe.predict(test_x)
        auroc = float(pl.metrics.auroc(test_ds.labels, probs))
        train_eval_s = time.perf_counter() - t_train0

    elapsed = time.perf_counter() - t0
    gpu_alloc_mb, gpu_reserved_mb = _read_cuda_peaks_mb()

    return BenchResult(
        task="train",
        method=method,
        elapsed_s=elapsed,
        extract_s=extract_s,
        train_eval_s=train_eval_s,
        cpu_peak_rss_mb=_cpu_peak_rss_mb(),
        gpu_peak_alloc_mb=gpu_alloc_mb,
        gpu_peak_reserved_mb=gpu_reserved_mb,
        metric_name="auroc",
        metric_value=auroc,
        extra={
            "n_train": len(train_ds),
            "n_test": len(test_ds),
            "layer": layer,
            "pool": args.pool,
            "lazy_cache_disk": bool(args.lazy_cache_disk),
        },
    )


def _run_sweep_case(args: argparse.Namespace, method: str) -> BenchResult:
    model, train_ds, test_ds, train_tokens, test_tokens = _load_data_and_model(args)
    n_layers = int(args.sweep_layers)
    layers = list(range(n_layers))
    cs = _parse_csv_floats(args.sweep_cs)

    t0 = time.perf_counter()
    t_extract0 = time.perf_counter()

    if method == "eager":
        train_full = _collect_eager(
            model,
            train_tokens,
            layers,
            batch_size=args.batch_size,
            pool=None,
        )
        test_full = _collect_eager(
            model,
            test_tokens,
            layers,
            batch_size=args.batch_size,
            pool=None,
        )
        extract_s = time.perf_counter() - t_extract0

        trial_scores: list[tuple[int, float, float]] = []
        for layer in layers:
            tr = train_full.select(layer=layer)
            te = test_full.select(layer=layer)
            if args.pool:
                tr = getattr(tr, f"{args.pool}_pool")()
                te = getattr(te, f"{args.pool}_pool")()
            for c in cs:
                probe = pl.probes.Logistic(C=c, max_iter=args.sweep_max_iter)
                probe.fit(tr, train_ds.labels)
                probs = probe.predict(te)
                score = float(pl.metrics.auroc(test_ds.labels, probs))
                trial_scores.append((layer, c, score))

    elif method == "lazy":
        with tempfile.TemporaryDirectory() as td:
            cache_train = Path(td) / "train_sweep.h5"
            cache_test = Path(td) / "test_sweep.h5"

            train_plan = pl.collect(
                model,
                train_tokens,
                layers=layers,
                batch_size=args.batch_size,
                dtype=_dtype_from_str(args.dtype),
            )
            test_plan = pl.collect(
                model,
                test_tokens,
                layers=layers,
                batch_size=args.batch_size,
                dtype=_dtype_from_str(args.dtype),
            )

            if args.pool:
                train_plan = getattr(train_plan, f"{args.pool}_pool")()
                test_plan = getattr(test_plan, f"{args.pool}_pool")()

            train_acts = train_plan.cache(cache_train) if args.lazy_cache_disk else train_plan
            test_acts = test_plan.cache(cache_test) if args.lazy_cache_disk else test_plan

            extract_s = time.perf_counter() - t_extract0

            trial_scores = []
            for layer in layers:
                tr = train_acts.select_layers(layer)
                te = test_acts.select_layers(layer)
                for c in cs:
                    probe = pl.probes.Logistic(C=c, max_iter=args.sweep_max_iter)
                    probe.fit(tr, train_ds.labels)
                    probs = probe.predict(te)
                    score = float(pl.metrics.auroc(test_ds.labels, probs))
                    trial_scores.append((layer, c, score))
    else:
        raise ValueError(f"Unknown method {method!r}")

    train_eval_s = time.perf_counter() - t0 - extract_s
    elapsed = time.perf_counter() - t0
    gpu_alloc_mb, gpu_reserved_mb = _read_cuda_peaks_mb()

    best_layer, best_c, best_score = max(trial_scores, key=lambda x: x[2])
    mean_score = float(sum(x[2] for x in trial_scores) / len(trial_scores))

    return BenchResult(
        task="sweep",
        method=method,
        elapsed_s=elapsed,
        extract_s=extract_s,
        train_eval_s=train_eval_s,
        cpu_peak_rss_mb=_cpu_peak_rss_mb(),
        gpu_peak_alloc_mb=gpu_alloc_mb,
        gpu_peak_reserved_mb=gpu_reserved_mb,
        metric_name="mean_auroc",
        metric_value=mean_score,
        extra={
            "n_train": len(train_ds),
            "n_test": len(test_ds),
            "n_layers": n_layers,
            "cs": cs,
            "n_trials": len(trial_scores),
            "best_layer": best_layer,
            "best_c": best_c,
            "best_auroc": best_score,
            "pool": args.pool,
            "lazy_cache_disk": bool(args.lazy_cache_disk),
        },
    )


def _run_child(args: argparse.Namespace) -> None:
    torch.manual_seed(args.seed)
    _reset_cuda_peaks()

    if args.task == "train":
        res = _run_train_case(args, args.method)
    elif args.task == "sweep":
        res = _run_sweep_case(args, args.method)
    else:
        raise ValueError(f"Unknown task {args.task!r}")

    print(json.dumps(dataclasses.asdict(res), sort_keys=True))


def _compare_rows(rows: list[BenchResult], task: str) -> dict[str, float] | None:
    eager = next((r for r in rows if r.task == task and r.method == "eager"), None)
    lazy = next((r for r in rows if r.task == task and r.method == "lazy"), None)
    if eager is None or lazy is None:
        return None

    return {
        "speedup_lazy_vs_eager": eager.elapsed_s / lazy.elapsed_s if lazy.elapsed_s > 0 else float("nan"),
        "cpu_mem_reduction_x": eager.cpu_peak_rss_mb / lazy.cpu_peak_rss_mb if lazy.cpu_peak_rss_mb > 0 else float("nan"),
        "gpu_alloc_reduction_x": eager.gpu_peak_alloc_mb / lazy.gpu_peak_alloc_mb if lazy.gpu_peak_alloc_mb > 0 else float("nan"),
        "gpu_reserved_reduction_x": eager.gpu_peak_reserved_mb / lazy.gpu_peak_reserved_mb if lazy.gpu_peak_reserved_mb > 0 else float("nan"),
    }


def _run_parent(args: argparse.Namespace) -> None:
    tasks = [x.strip() for x in args.tasks.split(",") if x.strip()]
    methods = ["eager", "lazy"]

    jobs: list[tuple[str, str]] = []
    for task in tasks:
        if task not in {"train", "sweep"}:
            raise ValueError(f"Unknown task {task!r}; expected train,sweep")
        for method in methods:
            jobs.append((task, method))

    rows: list[BenchResult] = []
    for task, method in jobs:
        cmd = [
            sys.executable,
            __file__,
            "--child",
            "--task",
            task,
            "--method",
            method,
            "--model",
            args.model,
            "--dataset",
            args.dataset,
            "--samples",
            str(args.samples),
            "--train-frac",
            str(args.train_frac),
            "--batch-size",
            str(args.batch_size),
            "--layer",
            str(args.layer),
            "--sweep-layers",
            str(args.sweep_layers),
            "--sweep-cs",
            args.sweep_cs,
            "--pool",
            args.pool,
            "--dtype",
            args.dtype,
            "--device-map",
            args.device_map,
            "--train-c",
            str(args.train_c),
            "--train-max-iter",
            str(args.train_max_iter),
            "--sweep-max-iter",
            str(args.sweep_max_iter),
            "--seed",
            str(args.seed),
        ]
        if args.lazy_cache_disk:
            cmd.append("--lazy-cache-disk")

        env = os.environ.copy()
        env.setdefault("PYTHONPATH", ".")

        raw = subprocess.check_output(cmd, text=True, env=env)
        line = raw.strip().splitlines()[-1]
        parsed = json.loads(line)
        rows.append(BenchResult(**parsed))

    print("\nLazy vs Eager Benchmark Results")
    print(f"model={args.model} dataset={args.dataset} samples={args.samples} batch_size={args.batch_size}")
    print(f"pool={args.pool} tasks={tasks} device_map={args.device_map} dtype={args.dtype}")
    print()
    print(f"{'task':8s} {'method':6s} {'elapsed_s':>10s} {'extract_s':>10s} {'train_eval_s':>12s} {'cpu_rss_mb':>11s} {'gpu_alloc_mb':>12s} {'metric':>10s}")
    print("-" * 105)
    for r in rows:
        print(
            f"{r.task:8s} {r.method:6s} "
            f"{r.elapsed_s:10.3f} {r.extract_s:10.3f} {r.train_eval_s:12.3f} "
            f"{r.cpu_peak_rss_mb:11.1f} {r.gpu_peak_alloc_mb:12.1f} {r.metric_value:10.4f}"
        )

    summary: dict[str, Any] = {}
    for task in tasks:
        comp = _compare_rows(rows, task)
        if comp is not None:
            summary[task] = comp

    if summary:
        print("\nSpeed/Memory Comparison (lazy relative to eager baseline)")
        for task, comp in summary.items():
            print(
                f"{task:8s} speedup={comp['speedup_lazy_vs_eager']:.3f}x "
                f"cpu_mem_reduction={comp['cpu_mem_reduction_x']:.3f}x "
                f"gpu_alloc_reduction={comp['gpu_alloc_reduction_x']:.3f}x"
            )

    payload = {
        "config": {
            "model": args.model,
            "dataset": args.dataset,
            "samples": args.samples,
            "train_frac": args.train_frac,
            "batch_size": args.batch_size,
            "pool": args.pool,
            "layer": args.layer,
            "sweep_layers": args.sweep_layers,
            "sweep_cs": _parse_csv_floats(args.sweep_cs),
            "dtype": args.dtype,
            "device_map": args.device_map,
            "seed": args.seed,
            "tasks": tasks,
            "lazy_cache_disk": bool(args.lazy_cache_disk),
        },
        "rows": [dataclasses.asdict(r) for r in rows],
        "summary": summary,
    }

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, sort_keys=True))
    print(f"\nWrote JSON report: {out}")


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Benchmark lazy Acts vs eager Activations")

    p.add_argument("--child", action="store_true", help="internal")
    p.add_argument("--task", type=str, default="train", help="train or sweep")
    p.add_argument("--method", type=str, default="lazy", help="eager or lazy")

    p.add_argument("--tasks", type=str, default="train,sweep", help="comma-separated: train,sweep")

    p.add_argument("--model", type=str, default="meta-llama/Llama-3.2-1B-Instruct")
    p.add_argument("--dataset", type=str, default="repe")
    p.add_argument("--samples", type=int, default=600)
    p.add_argument("--train-frac", type=float, default=0.8)

    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--pool", type=str, default="mean", choices=["mean", "max", "last", "sum"])
    p.add_argument("--layer", type=int, default=12)
    p.add_argument("--sweep-layers", type=int, default=16)
    p.add_argument("--sweep-cs", type=str, default="0.1,1.0,10.0")

    p.add_argument("--dtype", type=str, default="bf16", help="bf16|fp16|fp32")
    p.add_argument("--device-map", type=str, default="auto")

    p.add_argument("--train-c", type=float, default=1.0)
    p.add_argument("--train-max-iter", type=int, default=100)
    p.add_argument("--sweep-max-iter", type=int, default=60)

    p.add_argument("--lazy-cache-disk", action="store_true", help="materialize lazy outputs to disk cache before training")
    p.add_argument("--seed", type=int, default=0)

    p.add_argument("--output", type=str, default="perf_checks/lazy_vs_eager/results.json")
    return p


def main() -> None:
    args = _parser().parse_args()
    if args.child:
        _run_child(args)
    else:
        _run_parent(args)


if __name__ == "__main__":
    main()
