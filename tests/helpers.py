"""Simple test utilities for probelab tests."""
import torch
from probelab.processing.activations import Activations
from probelab.types import Label

# =============================================================================
# Activations Helpers
# =============================================================================

def acts(n_layers=1, batch=4, seq=8, d_model=16, layer_indices=None, detection_mask=None):
  """Create test activations. Simple, explicit, no magic."""
  if layer_indices is None:
    layer_indices = list(range(n_layers))
  t = torch.randn(batch, n_layers, seq, d_model)
  det = detection_mask if detection_mask is not None else torch.ones(batch, seq)
  return Activations(
    data=t,
    dims="blsh",
    mask=det,
    layers=tuple(layer_indices),
  )

def separable_acts(n_samples=20, seq=8, d_model=8, gap=2.0):
  """Create linearly separable activations. First half positive, second half negative."""
  half = n_samples // 2
  t = torch.zeros(n_samples, 1, seq, d_model)
  t[:half, 0, :, 0] = gap  # positive class: high in dim 0
  t[half:, 0, :, 0] = -gap  # negative class: low in dim 0
  t[:, :, :, 1:] = torch.randn(n_samples, 1, seq, d_model - 1) * 0.1  # noise
  det = torch.ones(n_samples, seq)
  labels = [Label.POSITIVE] * half + [Label.NEGATIVE] * half
  return Activations(data=t, dims="blsh", mask=det, layers=(0,)), labels

# =============================================================================
# Probability Tensor Helpers
# =============================================================================

def seq_probs(batch=4, n_classes=2):
  """Create sequence-level probabilities [batch, 2]."""
  raw = torch.rand(batch, n_classes)
  return raw / raw.sum(dim=-1, keepdim=True)  # normalize

def token_probs(batch=4, seq=8, n_classes=2):
  """Create token-level probabilities [batch, seq, 2]."""
  raw = torch.rand(batch, seq, n_classes)
  return raw / raw.sum(dim=-1, keepdim=True)

# =============================================================================
# Labels Helpers
# =============================================================================

def labels_binary(n, pos_ratio=0.5):
  """Create binary labels."""
  n_pos = int(n * pos_ratio)
  return [Label.POSITIVE] * n_pos + [Label.NEGATIVE] * (n - n_pos)

def labels_int(n, pos_ratio=0.5):
  """Create integer labels 0/1."""
  n_pos = int(n * pos_ratio)
  return [1] * n_pos + [0] * (n - n_pos)

# =============================================================================
# Assertions
# =============================================================================

def assert_close(a, b, atol=1e-5, rtol=1e-5):
  """Assert two tensors are close."""
  assert torch.allclose(a, b, atol=atol, rtol=rtol), f"Tensors not close:\n{a}\nvs\n{b}"

def assert_shape(t, expected):
  """Assert tensor has expected shape."""
  assert tuple(t.shape) == expected, f"Expected shape {expected}, got {tuple(t.shape)}"

def assert_probs(t, dim=-1):
  """Assert tensor contains valid probabilities (sum to 1, in [0,1])."""
  assert torch.all(t >= 0) and torch.all(t <= 1), "Probabilities must be in [0, 1]"
  sums = t.sum(dim=dim)
  assert torch.allclose(sums, torch.ones_like(sums)), f"Probabilities must sum to 1, got {sums}"
