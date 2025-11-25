"""Profile the pooling operation to find bottlenecks."""
import time
import torch
import probelib as pl
from probelib.processing.activations import Activations, Axis, SequenceMeta, LayerMeta

def profile_pooling():
    n_samples = 1000
    seq_len = 512
    hidden_dim = 2304
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    
    # Create activations on GPU
    print("\n=== Creating activations on GPU ===")
    acts = torch.randn(1, n_samples, seq_len, hidden_dim, device=device)
    attention_mask = torch.ones(n_samples, seq_len, device=device)
    detection_mask = torch.ones(n_samples, seq_len, device=device)
    input_ids = torch.zeros(n_samples, seq_len, dtype=torch.long, device=device)
    
    activations = Activations(
        activations=acts,
        axes=(Axis.LAYER, Axis.BATCH, Axis.SEQ, Axis.HIDDEN),
        layer_meta=LayerMeta((0,)),
        sequence_meta=SequenceMeta(attention_mask, detection_mask, input_ids),
        batch_indices=torch.arange(n_samples, device=device),
    )
    
    print(f"Activations shape: {acts.shape}")
    print(f"Activations device: {acts.device}")
    print(f"Memory: {acts.numel() * 4 / 1e9:.2f} GB")
    
    # Warmup
    print("\n=== Warmup ===")
    for _ in range(2):
        pooled = activations.pool(dim="sequence", method="mean")
    
    if device == "cuda":
        torch.cuda.synchronize()
    
    # Profile pooling
    print("\n=== Profiling pool() ===")
    times = []
    for i in range(5):
        if device == "cuda":
            torch.cuda.synchronize()
        start = time.perf_counter()
        pooled = activations.pool(dim="sequence", method="mean")
        if device == "cuda":
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        times.append(elapsed)
        print(f"  Run {i+1}: {elapsed:.4f}s")
    
    mean_time = sum(times) / len(times)
    print(f"  Mean: {mean_time:.4f}s")
    print(f"  Throughput: {n_samples / mean_time:.1f} samples/sec")
    
    # Test without detection mask (simpler path)
    print("\n=== Profiling pool() without detection mask ===")
    times = []
    for i in range(5):
        if device == "cuda":
            torch.cuda.synchronize()
        start = time.perf_counter()
        pooled = activations.pool(dim="sequence", method="mean", use_detection_mask=False)
        if device == "cuda":
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        times.append(elapsed)
        print(f"  Run {i+1}: {elapsed:.4f}s")
    
    mean_time = sum(times) / len(times)
    print(f"  Mean: {mean_time:.4f}s")
    print(f"  Throughput: {n_samples / mean_time:.1f} samples/sec")
    
    # Test raw tensor mean
    print("\n=== Profiling raw tensor mean ===")
    times = []
    for i in range(5):
        if device == "cuda":
            torch.cuda.synchronize()
        start = time.perf_counter()
        result = acts.mean(dim=2)  # Mean over seq dimension
        if device == "cuda":
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        times.append(elapsed)
        print(f"  Run {i+1}: {elapsed:.4f}s")
    
    mean_time = sum(times) / len(times)
    print(f"  Mean: {mean_time:.4f}s")
    print(f"  Throughput: {n_samples / mean_time:.1f} samples/sec")

if __name__ == "__main__":
    profile_pooling()
