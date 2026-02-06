"""L2-regularized logistic regression probe."""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..processing.acts import Acts
from ..processing.activations import Activations, Axis
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
    - If input has SEQ axis: trains on tokens, returns token-level scores
    - If no SEQ axis: trains on sequences, returns sequence-level scores

    Acts-specific path uses streaming updates for memory efficiency.
    """

    def __init__(
        self,
        C: float = 1.0,
        max_iter: int = 100,
        device: str | None = None,
        stream_batch_size: int = 4096,
        learning_rate: float = 3e-3,
    ):
        super().__init__(device=device)
        self.C = C
        self.max_iter = max_iter
        self.stream_batch_size = stream_batch_size
        self.learning_rate = learning_rate
        self._network: _LogisticNetwork | None = None
        self._scaler_mean: torch.Tensor | None = None
        self._scaler_std: torch.Tensor | None = None
        self._d_model: int | None = None
        self._trained_on_tokens = False
        self._tokens_per_sample: torch.Tensor | None = None

    def _normalize_batch_tensor(self, tensor: torch.Tensor, dims: str) -> tuple[torch.Tensor, str]:
        if "l" in dims:
            ldim = dims.index("l")
            if tensor.shape[ldim] != 1:
                raise ValueError("Logistic requires a single selected layer. Use select_layers(layer).")
            tensor = tensor.select(ldim, 0)
            dims = dims.replace("l", "")
        return tensor, dims

    def _fit_from_activations(self, X: Activations, y: list | torch.Tensor) -> "Logistic":
        if X.has_axis(Axis.LAYER):
            raise ValueError(
                f"Logistic expects no LAYER axis. "
                f"Add SelectLayer({X.layer_indices[0]}) to pipeline."
            )

        if self.device is None:
            self.device = str(X.activations.device)

        y_tensor = self._to_labels(y).to(self.device)

        if X.has_axis(Axis.SEQ):
            features, self._tokens_per_sample = X.extract_tokens()
            labels = torch.repeat_interleave(y_tensor, self._tokens_per_sample.to(y_tensor.device))
            self._trained_on_tokens = True
        else:
            features = X.activations
            labels = y_tensor
            self._trained_on_tokens = False
            self._tokens_per_sample = None

        if features.shape[0] == 0:
            return self

        features = features.to(self.device, dtype=torch.float32)
        labels = labels.to(self.device).float()

        self._d_model = features.shape[1]
        self._network = _LogisticNetwork(self._d_model).to(self.device)

        self._scaler_mean = features.mean(0)
        self._scaler_std = features.std(0).clamp(min=1e-8)
        features_scaled = (features - self._scaler_mean) / self._scaler_std

        optimizer = torch.optim.LBFGS(
            self._network.parameters(), max_iter=self.max_iter, line_search_fn="strong_wolfe"
        )
        l2_weight = 1.0 / (2.0 * self.C * len(labels)) if self.C > 0 else 0.0
        weight_param = self._network.linear.weight

        def closure():
            optimizer.zero_grad()
            loss = F.binary_cross_entropy_with_logits(self._network(features_scaled), labels)
            if l2_weight > 0:
                loss = loss + l2_weight * weight_param.pow(2).sum()
            loss.backward()
            return loss

        optimizer.step(closure)
        self._fitted = True
        return self

    def _iter_stream_xy(
        self,
        X: Acts,
        y_tensor: torch.Tensor,
    ):
        offset = 0
        for batch in X.iter_batches(self.stream_batch_size):
            bsize = batch.shape[batch.dims.index("b")]
            y_batch = y_tensor[offset : offset + bsize]
            offset += bsize

            t = batch.realize()
            dims = batch.dims
            t, dims = self._normalize_batch_tensor(t, dims)

            if "s" in dims:
                if batch.seq_mask is None:
                    raise ValueError("Token-level Acts requires seq_mask")
                mask = batch.seq_mask.to(t.device).bool()
                if y_batch.ndim == 1:
                    labels = torch.repeat_interleave(y_batch, mask.sum(1).to(y_batch.device))
                elif y_batch.ndim == 2:
                    labels = y_batch[mask.to(y_batch.device)]
                else:
                    raise ValueError(f"Token-level labels must have shape [b] or [b,s], got {tuple(y_batch.shape)}")
                feats = t[mask]
                yield feats, labels
            else:
                yield t, y_batch

    def _fit_from_acts(self, X: Acts, y: list | torch.Tensor) -> "Logistic":
        if "l" in X.dims and X.shape[X.dims.index("l")] != 1:
            raise ValueError("Logistic requires a single selected layer. Use acts.select_layers(layer).")

        if self.device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"

        y_tensor = self._to_labels(y)
        if y_tensor.shape[0] != X.shape[X.dims.index("b")]:
            raise ValueError(
                f"Label length {y_tensor.shape[0]} does not match batch size {X.shape[X.dims.index('b')]}"
            )

        y_tensor = y_tensor.float()
        self._trained_on_tokens = "s" in X.dims

        # Pass 1: streaming statistics (mean/std)
        n_total = 0
        mean = None
        m2 = None
        for feats, labels in self._iter_stream_xy(X, y_tensor):
            if feats.numel() == 0:
                continue
            feats = feats.to(self.device, dtype=torch.float32)
            n = feats.shape[0]
            batch_mean = feats.mean(dim=0)
            batch_m2 = ((feats - batch_mean) ** 2).sum(dim=0)

            if mean is None:
                mean = batch_mean
                m2 = batch_m2
                n_total = n
            else:
                delta = batch_mean - mean
                new_total = n_total + n
                mean = mean + delta * (n / new_total)
                m2 = m2 + batch_m2 + delta.pow(2) * (n_total * n / new_total)
                n_total = new_total

        if mean is None or n_total == 0:
            return self

        self._d_model = int(mean.shape[0])
        self._scaler_mean = mean
        self._scaler_std = torch.sqrt((m2 / max(n_total - 1, 1)).clamp(min=1e-8))

        self._network = _LogisticNetwork(self._d_model).to(self.device)
        optimizer = torch.optim.AdamW(self._network.parameters(), lr=self.learning_rate, weight_decay=0.0)
        l2_weight = 1.0 / (2.0 * self.C * n_total) if self.C > 0 else 0.0

        # Pass 2: streaming optimization
        self._network.train()
        for _ in range(self.max_iter):
            for feats, labels in self._iter_stream_xy(X, y_tensor):
                if feats.numel() == 0:
                    continue
                feats = feats.to(self.device, dtype=torch.float32)
                labels = labels.to(self.device, dtype=torch.float32)

                optimizer.zero_grad(set_to_none=True)
                x_scaled = (feats - self._scaler_mean) / self._scaler_std
                logits = self._network(x_scaled)
                loss = F.binary_cross_entropy_with_logits(logits, labels)
                if l2_weight > 0:
                    loss = loss + l2_weight * self._network.linear.weight.pow(2).sum()
                loss.backward()
                optimizer.step()

        self._network.eval()
        self._fitted = True
        return self

    def fit(self, X: Activations | Acts | torch.Tensor, y: list | torch.Tensor) -> "Logistic":
        if isinstance(X, Acts):
            return self._fit_from_acts(X, y)
        return self._fit_from_activations(self._to_activations(X), y)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """Differentiable forward pass.

        Args:
            x: Features tensor [batch, hidden] or [n_tokens, hidden]

        Returns:
            Logits tensor [batch] or [n_tokens] (not probabilities)
        """
        self._check_fitted()
        assert self._network is not None
        assert self._scaler_mean is not None and self._scaler_std is not None
        x = x.to(self.device, dtype=torch.float32)
        x_scaled = (x - self._scaler_mean) / self._scaler_std
        return self._network(x_scaled)

    def _predict_from_activations(self, X: Activations) -> torch.Tensor:
        self._check_fitted()

        if X.has_axis(Axis.LAYER):
            raise ValueError("Logistic expects no LAYER axis")

        with torch.no_grad():
            if X.has_axis(Axis.SEQ):
                features, _ = X.extract_tokens()
                flat_probs = torch.sigmoid(self(features))

                mask = X.detection_mask.bool()
                probs = torch.zeros_like(mask, dtype=flat_probs.dtype)
                probs[mask] = flat_probs
                return probs
            return torch.sigmoid(self(X.activations))

    def _predict_from_acts(self, X: Acts) -> torch.Tensor:
        self._check_fitted()

        if "l" in X.dims and X.shape[X.dims.index("l")] != 1:
            raise ValueError("Logistic requires a single selected layer. Use acts.select_layers(layer).")

        outs: list[torch.Tensor] = []
        with torch.no_grad():
            for batch in X.iter_batches(self.stream_batch_size):
                t = batch.realize()
                dims = batch.dims
                t, dims = self._normalize_batch_tensor(t, dims)

                if "s" in dims:
                    if batch.seq_mask is None:
                        raise ValueError("Token-level Acts requires seq_mask")
                    mask = batch.seq_mask.to(self.device).bool()
                    t = t.to(self.device, dtype=torch.float32)

                    batch_probs = torch.zeros(mask.shape, dtype=torch.float32, device=self.device)
                    if mask.any():
                        batch_probs[mask] = torch.sigmoid(self(t[mask]))
                    outs.append(batch_probs.cpu())
                else:
                    outs.append(torch.sigmoid(self(t)).cpu())

        if not outs:
            if "s" in X.dims:
                b = X.shape[X.dims.index("b")]
                s = X.shape[X.dims.index("s")]
                return torch.zeros((b, s), dtype=torch.float32)
            b = X.shape[X.dims.index("b")]
            return torch.zeros((b,), dtype=torch.float32)

        return torch.cat(outs, dim=0)

    def predict(self, X: Activations | Acts | torch.Tensor) -> torch.Tensor:
        if isinstance(X, Acts):
            return self._predict_from_acts(X)
        return self._predict_from_activations(self._to_activations(X))

    def save(self, path: Path | str) -> None:
        self._check_fitted()
        assert self._network is not None
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "C": self.C,
            "max_iter": self.max_iter,
            "stream_batch_size": self.stream_batch_size,
            "learning_rate": self.learning_rate,
            "device": self.device,
            "network_state": self._network.state_dict(),
            "scaler_mean": self._scaler_mean,
            "scaler_std": self._scaler_std,
            "d_model": self._d_model,
            "trained_on_tokens": self._trained_on_tokens,
        }, path)

    @classmethod
    def load(cls, path: Path | str, device: str = "cpu") -> "Logistic":
        state = torch.load(Path(path), map_location="cpu")
        probe = cls(
            C=state["C"],
            max_iter=state["max_iter"],
            device=device,
            stream_batch_size=state.get("stream_batch_size", 4096),
            learning_rate=state.get("learning_rate", 3e-3),
        )
        probe._d_model = state["d_model"]
        probe._network = _LogisticNetwork(probe._d_model).to(probe.device)
        probe._network.load_state_dict(state["network_state"])
        probe._network.eval()
        probe._scaler_mean = state["scaler_mean"].to(probe.device)
        probe._scaler_std = state["scaler_std"].to(probe.device)
        probe._trained_on_tokens = state.get("trained_on_tokens", False)
        probe._fitted = True
        return probe

    def __repr__(self) -> str:
        token_str = ", token-level" if self._trained_on_tokens else ""
        return f"Logistic(C={self.C}, fitted={self._fitted}{token_str})"
