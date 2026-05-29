"""Mass-mean difference-in-means probe."""

from __future__ import annotations

from pathlib import Path

import torch

from ..activations import Activations
from .base import BaseProbe


class MassMean(BaseProbe):
    """Closed-form direction probe."""

    def __init__(
        self,
        normalize: bool = False,
        center: bool = True,
        *,
        seed: int | None = None,
        device: str | None = None,
        cast: str | None = None,
    ):
        super().__init__(device=device, seed=seed, cast=cast)
        self.normalize = normalize
        self.center = center
        self.register_buffer("direction", torch.empty(0))
        self.register_buffer("bias", torch.empty(0))
        self.register_buffer("global_mean", torch.empty(0))

    def initialize(self, features: torch.Tensor, labels: torch.Tensor | None = None) -> "MassMean":
        if labels is None:
            raise ValueError("MassMean.initialize requires labels")
        working_dtype = self._resolve_dtype(features.dtype)
        self._training_dtype = working_dtype
        device = torch.device(self.device or features.device)
        features = features.to(dtype=working_dtype)
        labels = labels.to(dtype=working_dtype)
        d_model = features.shape[-1]
        if self.center:
            center = features.mean(0)
        else:
            center = torch.zeros(d_model, device=features.device, dtype=working_dtype)
        self.global_mean = center.to(device)
        z = features - center
        pos_mask = labels > 0.5
        neg_mask = ~pos_mask
        if not pos_mask.any() or not neg_mask.any():
            raise ValueError("MassMean requires both classes to be present in training labels.")
        mean_pos = z[pos_mask].mean(0)
        mean_neg = z[neg_mask].mean(0)
        direction = mean_pos - mean_neg
        if self.normalize:
            norm = direction.norm()
            if norm > 1e-8:
                direction = direction / norm
        proj_pos = mean_pos @ direction
        proj_neg = mean_neg @ direction
        bias = -(proj_pos + proj_neg) / 2
        self.direction = direction.to(device)
        self.bias = torch.tensor([bias], device=device, dtype=working_dtype)
        self._mark_initialized(working_dtype)
        return self

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        self._check_initialized()
        device, dtype = self._module_device_dtype()
        x = features.to(device=device, dtype=dtype)
        return (x - self.global_mean) @ self.direction + self.bias

    def fit(self, X: Activations, y: list | torch.Tensor, **kwargs) -> "MassMean":
        features, labels = self._feature_data_from_activations(X, y)
        if self.device is None:
            self.device = str(X.data.device)
        if features.shape[0] == 0:
            raise ValueError(
                f"{self.__class__.__name__}.fit received no training features "
                "(0 samples/tokens after masking). Check the activations and mask."
            )
        if features.shape[0] < 2:
            raise ValueError("MassMean requires at least two training samples or tokens.")
        features = features.to(dtype=self._resolve_dtype(features.dtype))
        labels = labels.to(dtype=features.dtype)
        return self.initialize(features, labels)

    def predict_logits(self, X: Activations, **kwargs) -> torch.Tensor:
        self._check_initialized()
        features, _ = self._feature_data_from_activations(X)
        logits = self._feature_logits_batched(features, kwargs.get("batch_size"))
        return self._feature_predict_from_flat(X, logits)

    def predict(self, X: Activations, **kwargs) -> torch.Tensor:
        with torch.no_grad():
            return torch.sigmoid(self.predict_logits(X, **kwargs))

    def save(self, path: Path | str) -> None:
        self._save_probe(path, {"normalize": self.normalize, "center": self.center})

    @classmethod
    def load(cls, path: Path | str, device: str = "cpu") -> "MassMean":
        state = torch.load(Path(path), map_location="cpu")
        probe = cls(**state["init_kwargs"], seed=state.get("seed"), device=device, cast=state.get("cast"))
        dtype = cls._stored_dtype(state)
        d_model = state["state_dict"]["direction"].shape[0]
        probe.direction = torch.zeros(d_model, dtype=dtype, device=device)
        probe.bias = torch.zeros(1, dtype=dtype, device=device)
        probe.global_mean = torch.zeros(d_model, dtype=dtype, device=device)
        probe._mark_initialized(dtype)
        probe.load_state_dict(state["state_dict"])
        probe.to(device=device, dtype=dtype)
        probe.eval()
        return probe

    def __repr__(self) -> str:
        return f"MassMean(normalize={self.normalize}, center={self.center}, fitted={self.fitted})"
