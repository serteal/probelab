"""Benchmark eager vs lazy Acts workflows.

This benchmark uses synthetic activation tensors to isolate activation pipeline costs.
It reports:
- wall time (seconds)
- peak RSS (MB) measured in isolated subprocesses

Workloads:
1. Single-layer select from disk cache.
2. Batch subset slice from disk cache.
3. Repeated layer sweeps: eager reload+pool vs lazy pooled-cache reuse.
"""

from __future__ import annotations

import argparse
import json
import os
import resource
import statistics
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

import torch

import probelab as pl
from probelab import pool as P


@dataclass
class RunResult:
    case: str
    elapsed_s: float
    peak_rss_mb: float
    checksum: float


def _rss_mb(raw_ru_maxrss: float) -> float:
    # macOS reports bytes, Linux reports kilobytes.
    if sys.platform == "darwin":
        return raw_ru_maxrss / (1024.0 * 1024.0)
    return raw_ru_maxrss / 1024.0


def _subset_indices(n: int, frac: float = 0.25) -> list[int]:
    step = max(1, int(round(1.0 / frac)))
    return list(range(0, n, step))


def _write_dataset(path: Path, *, b: int, l: int, s: int, h: int, dtype: torch.dtype) -> None:
    g = torch.Generator().manual_seed(0)
    t = torch.randn((b, l, s, h), generator=g, dtype=torch.float32).to(dtype)
    seq_mask = (torch.rand((b, s), generator=g) > 0.2).float()
    layer_ids = list(range(l))
    pl.Acts(t, dims="blsh", seq_mask=seq_mask, layer_ids=layer_ids).save(path, compression=None)


def _case_eager_layer_select(path: Path, target_layer: int) -> float:
    import h5py

    with h5py.File(path, "r") as f:
        acts = torch.from_numpy(f["acts"][:])
    selected = acts[:, target_layer]
    return float(selected[:, 0, 0].sum().item())


def _case_lazy_layer_select(path: Path, target_layer: int) -> float:
    acts = pl.load(path).select_layers(target_layer)
    selected = acts.realize().squeeze(1)
    return float(selected[:, 0, 0].sum().item())


def _case_eager_batch_slice(path: Path, batch_frac: float) -> float:
    import h5py

    with h5py.File(path, "r") as f:
        acts = torch.from_numpy(f["acts"][:])
    idx = _subset_indices(acts.shape[0], frac=batch_frac)
    sliced = acts[idx]
    return float(sliced[:, 0, 0, 0].sum().item())


def _case_lazy_batch_slice(path: Path, batch_frac: float) -> float:
    # list[int] slice intentionally to exercise indexed batch pushdown
    n = pl.load(path).shape[0]
    idx = _subset_indices(n, frac=batch_frac)
    sliced = pl.load(path).slice_batch(idx).realize()
    return float(sliced[:, 0, 0, 0].sum().item())


def _case_eager_sweep_reload(path: Path, sweeps: int) -> float:
    import h5py

    total = 0.0
    for _ in range(sweeps):
        with h5py.File(path, "r") as f:
            acts = torch.from_numpy(f["acts"][:])
            seq_mask = torch.from_numpy(f["seq_mask"][:]).float()
        for layer in range(acts.shape[1]):
            pooled = P.mean(acts[:, layer], seq_mask, dim=1)
            total += float(pooled[:, 0].sum().item())
    return total


def _case_lazy_sweep_cached_reuse(cache_path: Path, sweeps: int) -> float:
    total = 0.0
    for _ in range(sweeps):
        pooled = pl.load(cache_path)
        for layer_id in pooled.layer_ids or []:
            x = pooled.select_layers(layer_id).realize().squeeze(1)
            total += float(x[:, 0].sum().item())
    return total


def _run_child(case: str, path: Path, cache_path: Path, target_layer: int, sweeps: int, batch_frac: float) -> None:
    torch.set_num_threads(max(1, os.cpu_count() // 2))
    t0 = time.perf_counter()

    if case == "eager_layer_select":
        checksum = _case_eager_layer_select(path, target_layer)
    elif case == "lazy_layer_select":
        checksum = _case_lazy_layer_select(path, target_layer)
    elif case == "eager_batch_slice":
        checksum = _case_eager_batch_slice(path, batch_frac)
    elif case == "lazy_batch_slice":
        checksum = _case_lazy_batch_slice(path, batch_frac)
    elif case == "eager_sweep_reload":
        checksum = _case_eager_sweep_reload(path, sweeps)
    elif case == "lazy_sweep_cached_reuse":
        checksum = _case_lazy_sweep_cached_reuse(cache_path, sweeps)
    else:
        raise ValueError(f"Unknown case: {case}")

    elapsed = time.perf_counter() - t0
    peak = _rss_mb(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)

    print(
        json.dumps(
            {
                "case": case,
                "elapsed_s": elapsed,
                "peak_rss_mb": peak,
                "checksum": checksum,
            }
        )
    )


def _run_parent(args: argparse.Namespace) -> None:
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        dataset_path = root / "acts_bench.h5"
        cache_path = root / "acts_bench_pooled.h5"

        _write_dataset(
            dataset_path,
            b=args.batch,
            l=args.layers,
            s=args.seq,
            h=args.hidden,
            dtype=torch.float16,
        )

        # Build pooled cache once outside timed sweep runs.
        pl.load(dataset_path).mean_pool().cache(cache_path)

        cases = [
            "eager_layer_select",
            "lazy_layer_select",
            "eager_batch_slice",
            "lazy_batch_slice",
            "eager_sweep_reload",
            "lazy_sweep_cached_reuse",
        ]

        out: dict[str, list[RunResult]] = {c: [] for c in cases}
        for case in cases:
            for _ in range(args.repeats):
                cmd = [
                    sys.executable,
                    __file__,
                    "--child",
                    "--case",
                    case,
                    "--path",
                    str(dataset_path),
                    "--cache-path",
                    str(cache_path),
                    "--target-layer",
                    str(args.target_layer),
                    "--sweeps",
                    str(args.sweeps),
                    "--batch-frac",
                    str(args.batch_frac),
                ]
                raw = subprocess.check_output(cmd, text=True)
                line = raw.strip().splitlines()[-1]
                parsed = json.loads(line)
                out[case].append(
                    RunResult(
                        case=parsed["case"],
                        elapsed_s=float(parsed["elapsed_s"]),
                        peak_rss_mb=float(parsed["peak_rss_mb"]),
                        checksum=float(parsed["checksum"]),
                    )
                )

        def mean_elapsed(case: str) -> float:
            return statistics.mean(r.elapsed_s for r in out[case])

        def mean_mem(case: str) -> float:
            return statistics.mean(r.peak_rss_mb for r in out[case])

        print("\nLazy Acts Benchmark")
        print("dataset:", f"b={args.batch}, l={args.layers}, s={args.seq}, h={args.hidden}, dtype=float16")
        print("repeats:", args.repeats)
        print("sweeps:", args.sweeps)
        print("batch_frac:", args.batch_frac)
        print()
        print(f"{'case':28s} {'time_s':>10s} {'peak_rss_mb':>14s}")
        print("-" * 56)
        for case in cases:
            print(f"{case:28s} {mean_elapsed(case):10.4f} {mean_mem(case):14.2f}")

        print("\nSummary")
        print(f"layer-select speedup: {mean_elapsed('eager_layer_select') / mean_elapsed('lazy_layer_select'):.2f}x")
        print(
            "layer-select peak RSS reduction: "
            f"{mean_mem('eager_layer_select') / mean_mem('lazy_layer_select'):.2f}x"
        )
        print(f"batch-slice speedup: {mean_elapsed('eager_batch_slice') / mean_elapsed('lazy_batch_slice'):.2f}x")
        print(
            "batch-slice peak RSS reduction: "
            f"{mean_mem('eager_batch_slice') / mean_mem('lazy_batch_slice'):.2f}x"
        )
        print(
            "sweep speedup (reload eager vs cached lazy): "
            f"{mean_elapsed('eager_sweep_reload') / mean_elapsed('lazy_sweep_cached_reuse'):.2f}x"
        )
        print(
            "sweep peak RSS reduction: "
            f"{mean_mem('eager_sweep_reload') / mean_mem('lazy_sweep_cached_reuse'):.2f}x"
        )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--child", action="store_true")
    p.add_argument("--case", type=str, default="")
    p.add_argument("--path", type=str, default="")
    p.add_argument("--cache-path", type=str, default="")
    p.add_argument("--target-layer", type=int, default=3)
    p.add_argument("--sweeps", type=int, default=3)
    p.add_argument("--batch-frac", type=float, default=0.25)

    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--batch", type=int, default=192)
    p.add_argument("--layers", type=int, default=12)
    p.add_argument("--seq", type=int, default=128)
    p.add_argument("--hidden", type=int, default=192)

    args = p.parse_args()

    if args.child:
        if not args.case or not args.path or not args.cache_path:
            raise ValueError("child mode requires --case, --path, --cache-path")
        _run_child(
            args.case,
            Path(args.path),
            Path(args.cache_path),
            args.target_layer,
            args.sweeps,
            args.batch_frac,
        )
    else:
        _run_parent(args)
