"""G2 Reliability Learner Implementations.

Implements the three predeclared G2 candidate model families:
1. G2-LR14: Regularized binary logistic regression.
2. G2-AUC14: Pairwise ranking surrogate with training-only Platt calibration.
3. G2-Ordinal14: Cumulative-logit / proportional-odds ordinal model (4 severity bins).
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import torch
import torch.nn as nn


class G2LogisticRegression:
    """Regularized binary logistic regression for G2-LR14."""

    def __init__(self, l2_lambda: float = 1.0) -> None:
        self.l2_lambda = float(l2_lambda)
        self.weights: np.ndarray | None = None
        self.bias: float = 0.0

    def fit(self, X: np.ndarray, y: np.ndarray) -> G2LogisticRegression:
        N, D = X.shape
        X_t = torch.tensor(X, dtype=torch.float32)
        y_t = torch.tensor(y, dtype=torch.float32).unsqueeze(1)

        linear = nn.Linear(D, 1)
        nn.init.zeros_(linear.weight)
        nn.init.zeros_(linear.bias)

        optimizer = torch.optim.LBFGS(linear.parameters(), lr=0.5, max_iter=50)
        bce = nn.BCEWithLogitsLoss()

        def closure() -> torch.Tensor:
            optimizer.zero_grad()
            logits = linear(X_t)
            loss = bce(logits, y_t) + self.l2_lambda * torch.sum(linear.weight**2)
            loss.backward()
            return loss

        optimizer.step(closure)
        self.weights = linear.weight.detach().cpu().numpy().flatten()
        self.bias = float(linear.bias.detach().cpu().item())
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self.weights is None:
            raise RuntimeError("Model has not been fitted")
        z = X @ self.weights + self.bias
        z_clipped = np.clip(z, -20.0, 20.0)
        return 1.0 / (1.0 + np.exp(-z_clipped))


class G2AUCRanker:
    """Linear pairwise ranking model optimizing ROC-AUC surrogate for G2-AUC14."""

    def __init__(self, l2_lambda: float = 1.0) -> None:
        self.l2_lambda = float(l2_lambda)
        self.weights: np.ndarray | None = None
        self.platt_a: float = 1.0
        self.platt_b: float = 0.0

    def fit(self, X: np.ndarray, y: np.ndarray) -> G2AUCRanker:
        N, D = X.shape
        pos_idx = np.where(y == 1)[0]
        neg_idx = np.where(y == 0)[0]
        if len(pos_idx) == 0 or len(neg_idx) == 0:
            self.weights = np.zeros(D, dtype=np.float64)
            return self

        X_t = torch.tensor(X, dtype=torch.float32)
        w_param = nn.Parameter(torch.zeros(D, dtype=torch.float32))
        optimizer = torch.optim.LBFGS([w_param], lr=0.5, max_iter=50)

        pos_X = X_t[pos_idx]  # (P, D)
        neg_X = X_t[neg_idx]  # (M, D)

        def closure() -> torch.Tensor:
            optimizer.zero_grad()
            s_pos = torch.matmul(pos_X, w_param)  # (P,)
            s_neg = torch.matmul(neg_X, w_param)  # (M,)
            diff = s_pos.unsqueeze(1) - s_neg.unsqueeze(0)  # (P, M)
            loss = torch.log1p(torch.exp(-diff)).mean() + self.l2_lambda * torch.sum(w_param**2)
            loss.backward()
            return loss

        optimizer.step(closure)
        self.weights = w_param.detach().cpu().numpy().flatten()

        # Fit training-only Platt calibration
        raw_scores = X @ self.weights
        raw_t = torch.tensor(raw_scores, dtype=torch.float32).unsqueeze(1)
        y_t = torch.tensor(y, dtype=torch.float32).unsqueeze(1)

        platt_linear = nn.Linear(1, 1)
        platt_opt = torch.optim.LBFGS(platt_linear.parameters(), lr=0.5, max_iter=30)
        bce = nn.BCEWithLogitsLoss()

        def platt_closure() -> torch.Tensor:
            platt_opt.zero_grad()
            logits = platt_linear(raw_t)
            loss = bce(logits, y_t)
            loss.backward()
            return loss

        platt_opt.step(platt_closure)
        self.platt_a = float(platt_linear.weight.detach().cpu().item())
        self.platt_b = float(platt_linear.bias.detach().cpu().item())
        return self

    def predict_score(self, X: np.ndarray) -> np.ndarray:
        if self.weights is None:
            raise RuntimeError("Model has not been fitted")
        return X @ self.weights

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self.weights is None:
            raise RuntimeError("Model has not been fitted")
        z = (X @ self.weights) * self.platt_a + self.platt_b
        z_clipped = np.clip(z, -20.0, 20.0)
        return 1.0 / (1.0 + np.exp(-z_clipped))


class G2OrdinalClassifier:
    """Proportional-odds cumulative-logit model for G2-Ordinal14 (4 severity bins)."""

    def __init__(self, l2_lambda: float = 1.0) -> None:
        self.l2_lambda = float(l2_lambda)
        self.weights: np.ndarray | None = None
        self.theta: tuple[float, float, float] | None = None

    @staticmethod
    def map_ranks_to_bins(ranks: Sequence[int] | np.ndarray) -> np.ndarray:
        """Map true B1 ranks into 4 severity bins: 0 (<=8), 1 (9..16), 2 (17..32), 3 (>32)."""
        r = np.asarray(ranks, dtype=np.int32)
        bins = np.zeros(len(r), dtype=np.int64)
        bins[(r >= 9) & (r <= 16)] = 1
        bins[(r >= 17) & (r <= 32)] = 2
        bins[r > 32] = 3
        return bins

    def fit(self, X: np.ndarray, y_bins: np.ndarray) -> G2OrdinalClassifier:
        N, D = X.shape
        X_t = torch.tensor(X, dtype=torch.float32)
        y_t = torch.tensor(y_bins, dtype=torch.long)

        w_param = nn.Parameter(torch.zeros(D, dtype=torch.float32))
        c1 = nn.Parameter(torch.tensor(-1.0, dtype=torch.float32))
        d2 = nn.Parameter(torch.tensor(0.5, dtype=torch.float32))
        d3 = nn.Parameter(torch.tensor(0.5, dtype=torch.float32))

        optimizer = torch.optim.LBFGS([w_param, c1, d2, d3], lr=0.5, max_iter=50)
        softplus = nn.Softplus()

        def closure() -> torch.Tensor:
            optimizer.zero_grad()
            th1 = c1
            th2 = th1 + softplus(d2)
            th3 = th2 + softplus(d3)

            eta = torch.matmul(X_t, w_param)  # (N,)
            p_ge1 = torch.sigmoid(eta - th1)
            p_ge2 = torch.sigmoid(eta - th2)
            p_ge3 = torch.sigmoid(eta - th3)

            p0 = 1.0 - p_ge1
            p1 = p_ge1 - p_ge2
            p2 = p_ge2 - p_ge3
            p3 = p_ge3

            probs = torch.stack([p0, p1, p2, p3], dim=1).clamp(min=1e-7, max=1.0 - 1e-7)
            probs = probs / probs.sum(dim=1, keepdim=True)
            nll = -torch.log(probs[torch.arange(N), y_t]).mean() + self.l2_lambda * torch.sum(w_param**2)
            nll.backward()
            return nll

        optimizer.step(closure)
        self.weights = w_param.detach().cpu().numpy().flatten()
        th1 = float(c1.item())
        th2 = th1 + float(softplus(d2).item())
        th3 = th2 + float(softplus(d3).item())
        self.theta = (th1, th2, th3)
        return self

    def predict_proba_fail(self, X: np.ndarray) -> np.ndarray:
        """Return P(rank > 16 | x) = P(Y >= 2 | x) = sigma(w^T x - theta_2)."""
        if self.weights is None or self.theta is None:
            raise RuntimeError("Model has not been fitted")
        eta = X @ self.weights
        z = eta - self.theta[1]
        z_clipped = np.clip(z, -20.0, 20.0)
        return 1.0 / (1.0 + np.exp(-z_clipped))
