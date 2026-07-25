from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class NodeMassMLP(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, r: Tensor) -> Tensor:
        return F.softplus(self.net(r)).squeeze(-1)


def normalize_columns(r: Tensor, eps: float = 1e-8) -> Tensor:
    mn = r.min(dim=0, keepdim=True).values
    mx = r.max(dim=0, keepdim=True).values
    return (r - mn) / (mx - mn + eps)


def analytical_mass(r: Tensor, weights: Optional[Tensor] = None) -> Tensor:
    rn = normalize_columns(r)
    if weights is None:
        weights = torch.ones(rn.size(1), device=rn.device) / rn.size(1)
    return (rn * weights.view(1, -1)).sum(dim=-1)


def gravity_edge_prior(
    mass: Tensor,
    undirected_edges: Tensor,
    distance: Tensor,
    p: float = 2.0,
    eps: float = 1e-6,
) -> Tensor:
    src, dst = undirected_edges
    return (mass[src] * mass[dst]) / ((distance + eps) ** p)


def black_hole_mass_unmodified(
    degree: Tensor,
    pagerank: Tensor,
    clustering: Tensor,
) -> Tensor:
    """Baseline approximating the published Black Hole mass using common centrality proxies.

    Exact JCIM formula should replace this when the reference implementation is pinned.
    This function is explicitly marked as the 'unmodified-style' BH baseline for A32
    and must be compared against analytical/learnable variants — not assumed superior.
    """
    d = normalize_columns(degree.unsqueeze(-1)).squeeze(-1)
    pr = normalize_columns(pagerank.unsqueeze(-1)).squeeze(-1)
    cl = normalize_columns(clustering.unsqueeze(-1)).squeeze(-1)
    # Gravity-style composite used as published-inspired baseline
    return d * pr * (1.0 + cl)
