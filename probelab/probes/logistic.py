"""L2-regularized logistic regression probe."""

from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..processing.activations import Activations
from .base import BaseProbe


class _LogisticNetwork(nn.Module):
    """Simple logistic regression network."""

    def __init__(self, d_model: int):
        super().__init__()
        self.linear = nn.Linear(d_model, 1, bias=True)
        nn.init.xavier_uniform_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x).squeeze(-1)


class Logistic(BaseProbe):
    """L2-regularized logistic regression probe.

    Adapts to input dimensionality:
    - If X has SEQ axis: Trains on tokens, returns token-level scores
    - If X has no SEQ axis: Trains on sequences, returns sequence-level scores
    """

    def __init__(self, C: float = 1.0, max_iter: int = 100, device: str | None = None):
        super().__init__(device=device)
        self.C = C
        self.max_iter = max_iter
        self.net = None
        self.scaler_mean = None
        self.scaler_std = None

    @property
    def fitted(self) -> bool:
        return self.net is not None and self.scaler_mean is not None

    def fit(self, X: Activations, y: list | torch.Tensor) -> "Logistic":
        if "l" in X.dims:
            raise ValueError(
                f"Logistic expects no LAYER axis. "
                f"Call select_layers() first. Current dims: {X.dims}"
            )

        # Auto-detect device from input if not specified
        if self.device is None:
            self.device = str(X.data.device)

        y_tensor = self._to_labels(y).to(self.device)

        # Get features based on shape
        if "s" in X.dims:
            features, tokens_per_sample = X.extract_tokens()
            labels = torch.repeat_interleave(y_tensor, tokens_per_sample.to(y_tensor.device))
        else:
            features = X.data
            labels = y_tensor

        if features.shape[0] == 0:
            return self

        features = features.to(self.device)
        labels = labels.to(self.device).float()
        d_model = features.shape[1]

        # Fresh state every fit()
        self.net = _LogisticNetwork(d_model).to(self.device)
        if features.dtype != torch.float32:
            self.net = self.net.to(features.dtype)

        # Standardize
        self.scaler_mean = features.mean(0)
        self.scaler_std = features.std(0).clamp(min=1e-8)
        features_scaled = (features - self.scaler_mean) / self.scaler_std

        # Train with LBFGS
        optimizer = torch.optim.LBFGS(
            self.net.parameters(), max_iter=self.max_iter, line_search_fn="strong_wolfe"
        )
        l2_weight = 1.0 / (2.0 * self.C * len(labels)) if self.C > 0 else 0.0
        weight_param = self.net.linear.weight

        def closure():
            optimizer.zero_grad()
            loss = F.binary_cross_entropy_with_logits(self.net(features_scaled), labels)
            if l2_weight > 0:
                loss = loss + l2_weight * weight_param.pow(2).sum()
            loss.backward()
            return loss

        optimizer.step(closure)
        return self

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """Differentiable forward pass.

        Args:
            x: Features tensor [batch, hidden] or [n_tokens, hidden]

        Returns:
            Logits tensor [batch] or [n_tokens] (not probabilities)
        """
        self._check_fitted()
        x = x.to(self.device)
        x_scaled = (x - self.scaler_mean) / self.scaler_std
        return self.net(x_scaled.to(self.net.linear.weight.dtype))

    def predict(self, X: Activations) -> torch.Tensor:
        """Evaluate on activations.

        Args:
            X: Activations without LAYER axis

        Returns:
            Probability of positive class:
            - [batch, seq] if X has SEQ axis (token-level)
            - [batch] if X has no SEQ axis (sequence-level)
        """
        self._check_fitted()

        if "l" in X.dims:
            raise ValueError(f"Logistic expects no LAYER axis. Current dims: {X.dims}")

        with torch.no_grad():
            if "s" in X.dims:
                features, _ = X.extract_tokens()
                flat_probs = torch.sigmoid(self(features))

                # Scatter back to [batch, seq]
                mask = X.mask.bool()
                probs = torch.zeros_like(mask, dtype=flat_probs.dtype)
                probs[mask] = flat_probs
                return probs
            else:
                return torch.sigmoid(self(X.data))

    def save(self, path: Path | str) -> None:
        self._check_fitted()
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "C": self.C,
            "max_iter": self.max_iter,
            "device": self.device,
            "network_state": self.net.state_dict(),
            "scaler_mean": self.scaler_mean,
            "scaler_std": self.scaler_std,
        }, path)

    @classmethod
    def load(cls, path: Path | str, device: str = "cpu") -> "Logistic":
        state = torch.load(Path(path), map_location="cpu")
        probe = cls(C=state["C"], max_iter=state["max_iter"], device=device)
        d_model = state["scaler_mean"].shape[0]
        probe.net = _LogisticNetwork(d_model).to(probe.device)
        probe.net.load_state_dict(state["network_state"])
        probe.net.eval()
        probe.scaler_mean = state["scaler_mean"].to(probe.device)
        probe.scaler_std = state["scaler_std"].to(probe.device)
        return probe

    def __repr__(self) -> str:
        return f"Logistic(C={self.C}, fitted={self.fitted})"
