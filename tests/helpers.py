"""Simple test utilities for probelab tests."""
import torch
from probelab.processing.activations import Activations
from probelab.processing.scores import Scores
from probelab.types import Label

# =============================================================================
# Activations Helpers
# =============================================================================

def acts(n_layers=1, batch=4, seq=8, d_model=16, layer_indices=None, detection_mask=None):
  """Create test activations. Simple, explicit, no magic."""
  if layer_indices is None:
    layer_indices = list(range(n_layers))
  t = torch.randn(n_layers, batch, seq, d_model)
  attn = torch.ones(batch, seq)
  ids = torch.randint(0, 1000, (batch, seq))
  det = detection_mask if detection_mask is not None else torch.ones(batch, seq)
  return Activations.from_tensor(activations=t, attention_mask=attn, input_ids=ids, detection_mask=det, layer_indices=layer_indices)

def separable_acts(n_samples=20, seq=8, d_model=8, gap=2.0):
  """Create linearly separable activations. First half positive, second half negative."""
  half = n_samples // 2
  t = torch.zeros(1, n_samples, seq, d_model)
  t[0, :half, :, 0] = gap  # positive class: high in dim 0
  t[0, half:, :, 0] = -gap  # negative class: low in dim 0
  t[:, :, :, 1:] = torch.randn(1, n_samples, seq, d_model - 1) * 0.1  # noise
  attn = torch.ones(n_samples, seq)
  ids = torch.ones(n_samples, seq, dtype=torch.long)
  det = torch.ones(n_samples, seq)
  labels = [Label.POSITIVE] * half + [Label.NEGATIVE] * half
  return Activations.from_tensor(activations=t, attention_mask=attn, input_ids=ids, detection_mask=det, layer_indices=[0]), labels

# =============================================================================
# Scores Helpers
# =============================================================================

def seq_scores(batch=4, n_classes=2):
  """Create sequence-level scores [batch, 2]."""
  raw = torch.rand(batch, n_classes)
  raw = raw / raw.sum(dim=-1, keepdim=True)  # normalize
  return Scores.from_sequence_scores(raw)

def token_scores(batch=4, seq=8, n_classes=2):
  """Create token-level scores [batch, seq, 2]."""
  raw = torch.rand(batch, seq, n_classes)
  raw = raw / raw.sum(dim=-1, keepdim=True)
  tokens_per = torch.full((batch,), seq, dtype=torch.long)
  return Scores.from_token_scores(raw, tokens_per)

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
