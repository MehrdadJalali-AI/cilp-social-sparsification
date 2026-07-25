from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


def gumbel_sigmoid(logits: Tensor, tau: float = 1.0, hard: bool = False) -> Tensor:
    """Binary Concrete / Gumbel-Sigmoid relaxation."""
    g1 = -torch.empty_like(logits).exponential_().log()
    g2 = -torch.empty_like(logits).exponential_().log()
    y = torch.sigmoid((logits + g1 - g2) / tau)
    if hard:
        y_hard = (y > 0.5).float()
        y = y_hard - y.detach() + y
    return y


class AdversarialMaskGenerator(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim + 8, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, z: Tensor, noise: Tensor | None = None, tau: float = 1.0, hard: bool = False) -> Tensor:
        if noise is None:
            noise = torch.randn(z.size(0), 8, device=z.device)
        logits = self.net(torch.cat([z, noise], dim=-1)).squeeze(-1)
        return gumbel_sigmoid(logits, tau=tau, hard=hard)
