from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class ImportanceDecoder(nn.Module):
    """Heteroscedastic decoder: predicts μ and log σ² for edge importance."""

    def __init__(self, in_dim: int, hidden_dim: int = 64, dropout: float = 0.2):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.mu_head = nn.Linear(hidden_dim, 1)
        self.logvar_head = nn.Linear(hidden_dim, 1)

    def forward(self, z: Tensor) -> tuple[Tensor, Tensor]:
        h = self.backbone(z)
        mu = torch.sigmoid(self.mu_head(h)).squeeze(-1)
        logvar = self.logvar_head(h).squeeze(-1)
        # Softplus-like bound for numerical stability on variance
        logvar = torch.clamp(logvar, min=-8.0, max=4.0)
        return mu, logvar


def heteroscedastic_nll(mu: Tensor, logvar: Tensor, target: Tensor) -> Tensor:
    """Gaussian NLL with predicted variance."""
    var = torch.exp(logvar) + 1e-6
    return (0.5 * (logvar + (target - mu) ** 2 / var)).mean()


def ranking_hinge_loss(
    scores: Tensor,
    y: Tensor,
    margin: float = 0.1,
    num_pairs: int = 256,
) -> Tensor:
    """Sample pairs where y_high > y_low and enforce score ranking."""
    device = scores.device
    n = scores.numel()
    if n < 2:
        return scores.new_zeros(())
    # Sample random pairs
    i = torch.randint(0, n, (num_pairs,), device=device)
    j = torch.randint(0, n, (num_pairs,), device=device)
    yi, yj = y[i], y[j]
    mask = yi > yj
    if mask.sum() == 0:
        return scores.new_zeros(())
    loss = F.relu(margin - scores[i][mask] + scores[j][mask])
    return loss.mean()


def conservative_remove_mask(
    mu: Tensor,
    logvar: Tensor,
    tau: float,
    kappa: float = 1.0,
    rule: str = "upper",
) -> Tensor:
    """Return boolean mask of edges eligible for removal under uncertainty rule."""
    sigma = torch.sqrt(torch.exp(logvar) + 1e-6)
    if rule == "mean":
        return mu < tau
    if rule == "lower":
        return (mu - kappa * sigma) < tau
    # upper confidence: remove only if even upper bound is low
    return (mu + kappa * sigma) < tau
