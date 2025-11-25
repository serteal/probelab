"""Microbenchmark for probe training only (no activation collection)."""
import time
import torch
import probelab as pl
from probelab import Pipeline
from probelab.preprocessing import Pool, SelectLayer
from probelab.processing.activations import Activations, Axis, SequenceMeta, LayerMeta

torch.set_float32_matmul_precision("high")

def create_mock_activations(n_samples: int, seq_len: int, hidden_dim: int, n_layers: int = 1, pooled: bool = False, device: str = "cpu"):
    """Create mock activations for benchmarking."""
    if pooled:
        # Already pooled: [layers, batch, hidden]
        acts = torch.randn(n_layers, n_samples, hidden_dim, device=device)
        return Activations(
            activations=acts,
            axes=(Axis.LAYER, Axis.BATCH, Axis.HIDDEN),
            layer_meta=LayerMeta(tuple(range(n_layers))),
            sequence_meta=None,
            batch_indices=torch.arange(n_samples, device=device),
        )
    else:
        # Dense: [layers, batch, seq, hidden]
        acts = torch.randn(n_layers, n_samples, seq_len, hidden_dim, device=device)
        attention_mask = torch.ones(n_samples, seq_len, device=device)
        detection_mask = torch.ones(n_samples, seq_len, device=device)
        input_ids = torch.zeros(n_samples, seq_len, dtype=torch.long, device=device)
        return Activations(
            activations=acts,
            axes=(Axis.LAYER, Axis.BATCH, Axis.SEQ, Axis.HIDDEN),
            layer_meta=LayerMeta(tuple(range(n_layers))),
            sequence_meta=SequenceMeta(attention_mask, detection_mask, input_ids),
            batch_indices=torch.arange(n_samples, device=device),
        )

def benchmark_probe_training(n_samples: int = 1000, seq_len: int = 512, hidden_dim: int = 2304, n_runs: int = 5):
    """Benchmark probe training with pre-created activations."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nBenchmarking with {n_samples} samples, {seq_len} seq_len, {hidden_dim} hidden_dim")
    print(f"Device: {device}\n")

    # Create mock data
    labels = torch.randint(0, 2, (n_samples,)).tolist()

    # Test 1: Pooled activations (most common case)
    print("=" * 60)
    print("Test 1: Logistic on pre-pooled activations (GPU)")
    print("=" * 60)
    pooled_acts = create_mock_activations(n_samples, seq_len, hidden_dim, pooled=True, device=device)
    
    times = []
    for i in range(n_runs):
        pipeline = Pipeline([
            ("select", SelectLayer(0)),
            ("probe", pl.probes.Logistic(device="cuda" if torch.cuda.is_available() else "cpu")),
        ])
        
        start = time.perf_counter()
        pipeline.fit(pooled_acts, labels)
        elapsed = time.perf_counter() - start
        times.append(elapsed)
        print(f"  Run {i+1}: {elapsed:.4f}s")
    
    mean_time = sum(times) / len(times)
    throughput = n_samples / mean_time
    print(f"  Mean: {mean_time:.4f}s | Throughput: {throughput:.1f} samples/sec\n")
    
    # Test 2: Dense activations with pooling in pipeline
    print("=" * 60)
    print("Test 2: Logistic on dense activations (pool in pipeline, GPU)")
    print("=" * 60)
    dense_acts = create_mock_activations(n_samples, seq_len, hidden_dim, pooled=False, device=device)
    
    times = []
    for i in range(n_runs):
        pipeline = Pipeline([
            ("select", SelectLayer(0)),
            ("pool", Pool(dim="sequence", method="mean")),
            ("probe", pl.probes.Logistic(device="cuda" if torch.cuda.is_available() else "cpu")),
        ])
        
        start = time.perf_counter()
        pipeline.fit(dense_acts, labels)
        elapsed = time.perf_counter() - start
        times.append(elapsed)
        print(f"  Run {i+1}: {elapsed:.4f}s")
    
    mean_time = sum(times) / len(times)
    throughput = n_samples / mean_time
    print(f"  Mean: {mean_time:.4f}s | Throughput: {throughput:.1f} samples/sec\n")
    
    # Test 3: 10 probes on pooled activations (parallel training)
    print("=" * 60)
    print("Test 3: 10 Logistic probes on pre-pooled activations")
    print("=" * 60)
    
    times = []
    for i in range(n_runs):
        pipelines = {
            f"probe_{j}": Pipeline([
                ("select", SelectLayer(0)),
                ("probe", pl.probes.Logistic(device="cuda" if torch.cuda.is_available() else "cpu")),
            ])
            for j in range(10)
        }
        
        start = time.perf_counter()
        pl.scripts.train_pipelines(pipelines, pooled_acts, labels, verbose=False)
        elapsed = time.perf_counter() - start
        times.append(elapsed)
        print(f"  Run {i+1}: {elapsed:.4f}s")
    
    mean_time = sum(times) / len(times)
    throughput = (n_samples * 10) / mean_time  # 10 probes
    print(f"  Mean: {mean_time:.4f}s | Throughput: {throughput:.1f} probe*samples/sec\n")
    
    # Test 4: MLP probe on pooled activations
    print("=" * 60)
    print("Test 4: MLP on pre-pooled activations")
    print("=" * 60)
    
    times = []
    for i in range(n_runs):
        pipeline = Pipeline([
            ("select", SelectLayer(0)),
            ("probe", pl.probes.MLP(device="cuda" if torch.cuda.is_available() else "cpu", n_epochs=10)),
        ])
        
        start = time.perf_counter()
        pipeline.fit(pooled_acts, labels)
        elapsed = time.perf_counter() - start
        times.append(elapsed)
        print(f"  Run {i+1}: {elapsed:.4f}s")
    
    mean_time = sum(times) / len(times)
    throughput = n_samples / mean_time
    print(f"  Mean: {mean_time:.4f}s | Throughput: {throughput:.1f} samples/sec\n")

if __name__ == "__main__":
    benchmark_probe_training(n_samples=1000, seq_len=512, hidden_dim=2304)
