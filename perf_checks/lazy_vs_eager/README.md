# Lazy vs Eager Benchmarks

This folder contains reproducible benchmarks for comparing:

- **Lazy activations** (`Acts` pipeline with lazy transforms + optional disk cache), vs
- **Normal eager activations** (explicit eager baseline that materializes full activation tensors).

The benchmark targets realistic workloads on large datasets/models:

1. **Activation training workload**: extract activations, pool, train/eval a logistic probe.
2. **Layer/hyperparameter sweep workload**: collect multi-layer activations, run per-layer probe sweeps across `C` values.

---

## Why these experiments matter

1. **Prevents regressions**: verifies lazy changes actually reduce memory and do not silently degrade throughput.
2. **Finds the crossover point**: identifies when lazy streaming is preferable vs when eager/in-memory is faster.
3. **Improves planning for large runs**: tells us which execution mode to use for large-scale probe experiments.
4. **Supports deployment decisions**: validates defaults for users running on 1 GPU vs multi-GPU nodes.

---

## Files

- `benchmark_lazy_vs_eager.py`: benchmark runner (isolated subprocess per case).
- `REPORT_TEMPLATE.md`: standardized report format for the big-GPU agent.

---

## What is measured

Per case, the runner records:

- End-to-end wall time (`elapsed_s`)
- Extraction time (`extract_s`)
- Train/eval time (`train_eval_s`)
- Peak CPU RSS (`cpu_peak_rss_mb`)
- Peak GPU allocated/reserved (`gpu_peak_alloc_mb`, `gpu_peak_reserved_mb`)
- Task metric (`auroc` for train, `mean_auroc` for sweep)

It also computes lazy-vs-eager ratios:

- `speedup_lazy_vs_eager = eager_time / lazy_time`
- `cpu_mem_reduction_x = eager_cpu_peak / lazy_cpu_peak`
- `gpu_alloc_reduction_x = eager_gpu_alloc_peak / lazy_gpu_alloc_peak`

---

## Prerequisites

1. Python env with dependencies installed.
2. Access to requested HF model(s)/dataset(s).
3. Sufficient GPU memory for selected settings.

Recommended env setup from repo root:

```bash
cd /tmp/probelab-lazy-acts
export PYTHONPATH=.
```

If needed for gated models:

```bash
export HUGGING_FACE_HUB_TOKEN=...
```

---

## Run commands

Run both workloads (train + sweep):

```bash
PYTHONPATH=. /Users/alexserrano/Documents/MATS/probelab/.venv/bin/python \
  perf_checks/lazy_vs_eager/benchmark_lazy_vs_eager.py \
  --tasks train,sweep \
  --model meta-llama/Llama-3.2-1B-Instruct \
  --dataset repe \
  --samples 600 \
  --batch-size 8 \
  --pool mean \
  --sweep-layers 16 \
  --sweep-cs 0.1,1.0,10.0 \
  --dtype bf16 \
  --device-map auto \
  --lazy-cache-disk \
  --output perf_checks/lazy_vs_eager/results_1b_600.json
```

Run train workload only:

```bash
PYTHONPATH=. /Users/alexserrano/Documents/MATS/probelab/.venv/bin/python \
  perf_checks/lazy_vs_eager/benchmark_lazy_vs_eager.py \
  --tasks train \
  --model meta-llama/Llama-3.2-1B-Instruct \
  --dataset repe \
  --samples 1200 \
  --layer 12 \
  --batch-size 8 \
  --pool mean \
  --dtype bf16 \
  --device-map auto \
  --lazy-cache-disk \
  --output perf_checks/lazy_vs_eager/results_train_only.json
```

Run sweep workload only:

```bash
PYTHONPATH=. /Users/alexserrano/Documents/MATS/probelab/.venv/bin/python \
  perf_checks/lazy_vs_eager/benchmark_lazy_vs_eager.py \
  --tasks sweep \
  --model meta-llama/Llama-3.2-1B-Instruct \
  --dataset repe \
  --samples 1200 \
  --sweep-layers 24 \
  --sweep-cs 0.1,1.0,10.0 \
  --batch-size 8 \
  --pool mean \
  --dtype bf16 \
  --device-map auto \
  --lazy-cache-disk \
  --output perf_checks/lazy_vs_eager/results_sweep_only.json
```

---

## Suggested experiment matrix (big GPU agent)

Run at least 3 scales for each model:

1. `samples=2k`, `sweep-layers=16`
2. `samples=10k`, `sweep-layers=24`
3. `samples=25k+`, `sweep-layers=max feasible`

Suggested models:

1. `meta-llama/Llama-3.2-1B-Instruct`
2. `meta-llama/Llama-3.1-8B-Instruct`
3. any internal larger activation model used in production experiments

Keep all non-size settings fixed when comparing eager vs lazy.

---

## Reporting instructions

1. Save each JSON output file.
2. Fill `REPORT_TEMPLATE.md` using those JSON files.
3. Include:
   - hardware/software details,
   - exact command lines,
   - aggregated speed/memory ratios,
   - failure/oom notes,
   - recommendation for default mode (`lazy` vs `eager`) per workload scale.

