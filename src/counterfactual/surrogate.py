from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from scipy.stats import kendalltau, spearmanr


class SurrogateEdgeImportance(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, z: Tensor) -> Tensor:
        return torch.sigmoid(self.net(z).squeeze(-1))


def train_surrogate(
    model: SurrogateEdgeImportance,
    z: Tensor,
    y: Tensor,
    epochs: int = 100,
    lr: float = 1e-3,
) -> Dict[str, float]:
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    model.train()
    last = 0.0
    for _ in range(epochs):
        opt.zero_grad()
        pred = model(z)
        loss = F.mse_loss(pred, y)
        loss.backward()
        opt.step()
        last = float(loss.item())
    return {"mse": last}


@torch.no_grad()
def evaluate_surrogate(pred: np.ndarray, true: np.ndarray, k: int = 50) -> Dict[str, float]:
    mae = float(np.mean(np.abs(pred - true)))
    sp = float(spearmanr(pred, true).correlation or 0.0)
    kd = float(kendalltau(pred, true).correlation or 0.0)
    k = min(k, len(pred))
    top_p = set(np.argsort(-pred)[:k].tolist())
    top_t = set(np.argsort(-true)[:k].tolist())
    prec = len(top_p & top_t) / k
    rec = len(top_p & top_t) / k
    return {
        "mae": mae,
        "spearman": sp,
        "kendall": kd,
        "top_k_precision": float(prec),
        "top_k_recall": float(rec),
    }
