"""L2-regularized logistic regression probe."""

from pathlib import Path
from typing import Callable

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

    def __init__(
        self,
        C: float = 1.0,
        max_iter: int = 500,
        n_epochs: int = 100,
        *,
        optimizer_fn: Callable | None = None,
        scheduler_fn: Callable | None = None,
        seed: int | None = None,
        device: str | None = None,
        cast: str | None = None,
    ):
        super().__init__(device=device, seed=seed, optimizer_fn=optimizer_fn, scheduler_fn=scheduler_fn, cast=cast)
        self.C = C
        self.max_iter = max_iter
        self.n_epochs = n_epochs
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
                f"Call select(\"l\", layer) first. Current dims: {X.dims}"
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

        working_dtype = self._resolve_dtype(features.dtype)
        self._training_dtype = working_dtype

        features = features.to(self.device, dtype=working_dtype)
        labels = labels.to(self.device, dtype=working_dtype)
        d_model = features.shape[1]

        # Fresh state every fit()
        self._seed_everything()  # seeds weight init; no local generator needed
        self.net = _LogisticNetwork(d_model).to(self.device, dtype=working_dtype)

        # Standardize (clone to avoid mutating input data)
        self.scaler_mean = features.mean(0)
        self.scaler_std = features.std(0).clamp(min=1e-8)
        features = (features - self.scaler_mean) / self.scaler_std

        l2_weight = 1.0 / (2.0 * self.C * len(labels)) if self.C > 0 else 0.0
        weight_param = self.net.linear.weight

        if self._optimizer_fn is not None:
            # Custom optimizer: epoch-based training loop
            optimizer = self._optimizer_fn(self.net.parameters())
            scheduler = self._make_scheduler(optimizer)
            self.net.train()
            for _ in range(self.n_epochs):
                optimizer.zero_grad()
                loss = F.binary_cross_entropy_with_logits(self.net(features), labels)
                if l2_weight > 0:
                    loss = loss + l2_weight * weight_param.pow(2).sum()
                loss.backward()
                optimizer.step()
                if scheduler is not None:
                    scheduler.step()
            self.net.eval()
        else:
            # Default: single-step LBFGS
            optimizer = torch.optim.LBFGS(
                self.net.parameters(), max_iter=self.max_iter, line_search_fn="strong_wolfe"
            )

            def closure():
                optimizer.zero_grad()
                loss = F.binary_cross_entropy_with_logits(self.net(features), labels)
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
        return self.net(x_scaled)

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

        net_dtype = next(self.net.parameters()).dtype

        with torch.no_grad():
            if "s" in X.dims:
                features, _ = X.extract_tokens()
                flat_probs = torch.sigmoid(self(features.to(dtype=net_dtype)))

                # Scatter back to [batch, seq] via to_padded()
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
            "C": self.C,
            "max_iter": self.max_iter,
            "n_epochs": self.n_epochs,
            "seed": self.seed,
            "device": self.device,
            "cast": self.cast,
            "training_dtype": str(self._training_dtype),
            "network_state": self.net.state_dict(),
            "scaler_mean": self.scaler_mean,
            "scaler_std": self.scaler_std,
        }, path)

    @classmethod
    def load(cls, path: Path | str, device: str = "cpu") -> "Logistic":
        state = torch.load(Path(path), map_location="cpu")
        probe = cls(
            C=state["C"],
            max_iter=state["max_iter"],
            n_epochs=state.get("n_epochs", 100),
            seed=state.get("seed"),
            device=device,
            cast=state.get("cast"),
        )
        # Backward compat: old checkpoints without training_dtype default to float32
        dtype_str = state.get("training_dtype")
        stored_dtype = getattr(torch, dtype_str.split(".")[-1]) if dtype_str else torch.float32
        probe._training_dtype = stored_dtype

        d_model = state["scaler_mean"].shape[0]
        probe.net = _LogisticNetwork(d_model).to(probe.device, dtype=stored_dtype)
        probe.net.load_state_dict(state["network_state"])
        probe.net.eval()
        probe.scaler_mean = state["scaler_mean"].to(probe.device, dtype=stored_dtype)
        probe.scaler_std = state["scaler_std"].to(probe.device, dtype=stored_dtype)
        return probe

    def __repr__(self) -> str:
        return f"Logistic(C={self.C}, fitted={self.fitted})"
