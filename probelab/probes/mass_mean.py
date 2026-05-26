"""Mass-mean (difference-in-means) probe.

Closed-form direction probe: direction = mean(z|y=1) - mean(z|y=0).
No SGD training required.

Reference: Marks & Tegmark, "The Geometry of Truth" (COLM 2024).
"""

from pathlib import Path
from typing import Callable

import torch
import torch.nn as nn

from ..activations import Activations
from .base import BaseProbe


class _MassMeanNetwork(nn.Module):
    """Stores direction vector and bias for inference."""

    def __init__(self, d_model: int):
        super().__init__()
        self.register_buffer("direction", torch.zeros(d_model))
        self.register_buffer("bias", torch.zeros(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x @ self.direction + self.bias


class MassMean(BaseProbe):
    """Mass-mean (difference-in-means) direction probe.

    Computes direction as difference between class means. Classification
    is via projection onto this direction. No iterative training needed.

    Adapts to input dimensionality:
    - If X has SEQ axis: Trains on tokens, returns token-level scores
    - If X has no SEQ axis: Trains on sequences, returns sequence-level scores

    Args:
        normalize: L2-normalize the direction vector. Paper uses False.
        center: Center features (subtract global mean) before computing direction.
    """

    def __init__(
        self,
        normalize: bool = False,
        center: bool = True,
        *,
        seed: int | None = None,
        device: str | None = None,
        cast: str | None = None,
        # Accept but ignore these so ProbeSpec dispatch works uniformly
        optimizer_fn: Callable | None = None,
        scheduler_fn: Callable | None = None,
    ):
        super().__init__(device=device, seed=seed, cast=cast)
        self.normalize = normalize
        self.center = center
        self.net = None
        self.global_mean = None

    @property
    def fitted(self) -> bool:
        return self.net is not None and self.global_mean is not None

    def fit(self, X: Activations, y: list | torch.Tensor) -> "MassMean":
        if "l" in X.dims:
            raise ValueError(
                f"MassMean expects no LAYER axis. "
                f'Call select("l", layer) first. Current dims: {X.dims}'
            )

        if self.device is None:
            self.device = str(X.data.device)

        y_tensor = self._to_labels(y)

        if "s" in X.dims:
            features, tokens_per_sample = X.extract_tokens()
            labels = torch.repeat_interleave(y_tensor, tokens_per_sample.to(y_tensor.device))
        else:
            features = X.data
            labels = y_tensor

        if features.shape[0] == 0:
            return self

        working_dtype = self._resolve_dtype(features.dtype)
        self._training_dtype = working_dtype

        features = features.to(dtype=working_dtype)
        labels = labels.to(device=features.device, dtype=working_dtype)
        d_model = features.shape[1]

        # Center only (no z-scoring per Marks & Tegmark 2024)
        if self.center:
            center = features.mean(0)
        else:
            center = features.new_zeros(d_model)
        self.global_mean = center.to(self.device)
        z = features - center

        # Compute class means on (optionally centered) data
        pos_mask = labels > 0.5
        neg_mask = ~pos_mask
        mean_pos = z[pos_mask].mean(0) if pos_mask.any() else features.new_zeros(d_model)
        mean_neg = z[neg_mask].mean(0) if neg_mask.any() else features.new_zeros(d_model)
        direction = mean_pos - mean_neg

        if self.normalize:
            norm = direction.norm()
            if norm > 1e-8:
                direction = direction / norm

        # Bias: midpoint between class mean projections
        # With centering + balanced classes this is ~0 (paper uses no explicit bias)
        proj_pos = mean_pos @ direction
        proj_neg = mean_neg @ direction
        bias = -(proj_pos + proj_neg) / 2

        self.net = _MassMeanNetwork(d_model).to(self.device, dtype=working_dtype)
        self.net.direction.copy_(direction.to(self.device))
        self.net.bias.copy_(bias.reshape(1).to(self.device, dtype=working_dtype))
        return self

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        self._check_fitted()
        x = x.to(self.device)
        x_centered = x - self.global_mean
        return self.net(x_centered)

    def predict(self, X: Activations) -> torch.Tensor:
        self._check_fitted()
        if "l" in X.dims:
            raise ValueError(f"MassMean expects no LAYER axis. Current dims: {X.dims}")

        net_dtype = self.global_mean.dtype

        with torch.no_grad():
            if "s" in X.dims:
                features, _ = X.extract_tokens()
                flat_probs = torch.sigmoid(self(features.to(dtype=net_dtype)))
                padded_data, padded_det = X.to_padded()
                probs = torch.zeros_like(padded_det, dtype=flat_probs.dtype)
                probs[padded_det] = flat_probs
                return probs
            else:
                return torch.sigmoid(self(X.data.to(dtype=net_dtype)))

    def save(self, path: Path | str) -> None:
        self._check_fitted()
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "normalize": self.normalize,
            "center": self.center,
            "seed": self.seed,
            "device": self.device,
            "cast": self.cast,
            "training_dtype": str(self._training_dtype),
            "network_state_dict": self.net.state_dict(),
            "global_mean": self.global_mean,
        }, path)

    @classmethod
    def load(cls, path: Path | str, device: str = "cpu") -> "MassMean":
        state = torch.load(Path(path), map_location="cpu")
        probe = cls(
            normalize=state["normalize"],
            center=state["center"],
            seed=state.get("seed"),
            device=device,
            cast=state.get("cast"),
        )
        dtype_str = state.get("training_dtype")
        stored_dtype = getattr(torch, dtype_str.split(".")[-1]) if dtype_str else torch.float32
        probe._training_dtype = stored_dtype

        d_model = state["global_mean"].shape[0]
        probe.net = _MassMeanNetwork(d_model).to(device, dtype=stored_dtype)
        probe.net.load_state_dict(state["network_state_dict"])
        probe.global_mean = state["global_mean"].to(device, dtype=stored_dtype)
        return probe

    def __repr__(self) -> str:
        return f"MassMean(normalize={self.normalize}, center={self.center}, fitted={self.fitted})"
