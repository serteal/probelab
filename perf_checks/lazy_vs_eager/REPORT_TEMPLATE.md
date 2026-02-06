# Lazy vs Eager Benchmark Report

## 1. Environment

- Date:
- Agent:
- Machine:
- CPU:
- RAM:
- GPU(s):
- Driver/CUDA:
- PyTorch version:
- Transformers version:
- probelab commit SHA:

## 2. Commands Run

Paste exact commands used (one block per run).

```bash
# run 1

# run 2

# ...
```

## 3. Result Files

- JSON outputs:
  - `...`
  - `...`

## 4. Summary Table

| Run | Task | Model | Samples | Layers | Pool | Lazy Speedup vs Eager (`eager/lazy`) | CPU Mem Reduction (`eager/lazy`) | GPU Alloc Reduction (`eager/lazy`) | Metric Delta (Lazy - Eager) |
|---|---|---|---:|---:|---|---:|---:|---:|---:|
| | | | | | | | | | |
| | | | | | | | | | |

## 5. Detailed Observations

### Train workload

- Time breakdown (`extract_s`, `train_eval_s`):
- Memory behavior (CPU + GPU):
- Metric parity (AUROC):
- Notable bottlenecks:

### Sweep workload

- Time breakdown (`extract_s`, `train_eval_s`):
- Memory behavior (CPU + GPU):
- Metric parity (mean AUROC / best AUROC):
- Notable bottlenecks:

## 6. Failure/OOM Notes

- Any failed runs:
- Stack traces or error types:
- Mitigations attempted:

## 7. Recommendation

- Default mode for small runs:
- Default mode for medium runs:
- Default mode for large runs:
- Suggested API/engine follow-ups:

## 8. Attachments

- Optional charts:
  - wall time vs samples
  - peak memory vs samples
  - speedup/memory reduction vs layers

