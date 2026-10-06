"""MoG-adapted: mixture-of-experts local sparsification (Zhang et al., ICLR 2025).

Adaptation of the official MoG design (yanweiyue/MoG): multiple SpLearner-style
experts with different per-node retention fractions, noisy top-k expert gating,
and edge scores aggregated as gate-weighted expert outputs. Trained with a fixed
2-layer GCN node classifier (Stage A protocol). Edge importance is exported
after training and converted to exact budgets via the shared top-(1-r)|E| prune.

This is NOT a full reproduction of MoG's OGB pipelines, RevGNN backbones, or
Grassmann mixing; see docs/baseline_fidelity_mog.md.
"""

from __future__ import annotations

from typing import Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
from torch_geometric.utils import degree

from src.utils.graph import subgraph_from_undirected_edges, undirected_edge_list
from src.sparsification.constrained_pruning import budget_prune_unconstrained


class _SpExpert(nn.Module):
    """Per-expert edge scorer (SpLearner-style MLP on endpoint features)."""

    def __init__(self, in_dim: int, hidden: int = 64):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(2 * in_dim + 1, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, h: Tensor, und: Tensor, edge_attr: Tensor) -> Tensor:
        src, dst = und
        x = torch.cat([h[src], h[dst], edge_attr.unsqueeze(-1)], dim=-1)
        z = self.mlp(x).squeeze(-1)
        return F.normalize(z, dim=0)


class MoGAdaptedScorer(nn.Module):
    def __init__(
        self,
        in_dim: int,
        k_list: Sequence[float] = (0.25, 0.5, 0.75),
        hidden: int = 64,
        expert_select: int = 2,
    ):
        super().__init__()
        self.k_list = torch.tensor(k_list, dtype=torch.float32)
        self.expert_select = expert_select
        self.enc = GCNConv(in_dim, hidden)
        self.experts = nn.ModuleList([_SpExpert(hidden, hidden) for _ in k_list])
        self.w_gate = nn.Parameter(torch.zeros(hidden, len(k_list)))
        self.w_noise = nn.Parameter(torch.zeros(hidden, len(k_list)))

    def _gates(self, h: Tensor, train: bool) -> Tensor:
        logits = h @ self.w_gate
        if train:
            noise = F.softplus(h @ self.w_noise) + 1e-2
            logits = logits + torch.randn_like(logits) * noise
        k = min(self.expert_select, logits.size(1))
        top_logits, top_idx = logits.topk(k, dim=1)
        gates = torch.zeros_like(logits)
        gates.scatter_(1, top_idx, F.softmax(top_logits, dim=1))
        return gates

    def edge_scores(self, x: Tensor, edge_index: Tensor, und: Tensor) -> Tensor:
        h = F.relu(self.enc(x, edge_index))
        deg = degree(edge_index[0], num_nodes=x.size(0)).clamp(min=1.0)
        src, dst = und
        edge_attr = (1.0 / deg[src] + 1.0 / deg[dst]).to(x.device)
        node_gates = self._gates(h, self.training)
        edge_gates = node_gates[src]
        expert_out = torch.stack([exp(h, und, edge_attr) for exp in self.experts], dim=1)
        gated = (edge_gates * expert_out).mean(dim=1)
        return torch.sigmoid(gated)


def mog_adapted_sparsify(
    data: Data,
    removal_rate: float,
    epochs: int = 30,
    device: Optional[torch.device] = None,
) -> Data:
    device = device or torch.device("cpu")
    und = undirected_edge_list(data.edge_index)
    model = MoGAdaptedScorer(data.x.size(1)).to(device)
    clf = nn.Linear(64, int(data.y.max().item()) + 1).to(device)
    opt = torch.optim.Adam(list(model.parameters()) + list(clf.parameters()), lr=1e-3)
    x, y, ei = data.x.to(device), data.y.to(device), data.edge_index.to(device)
    train = data.train_mask.to(device)

    for _ in range(epochs):
        model.train()
        opt.zero_grad()
        h = F.relu(model.enc(x, ei))
        scores = model.edge_scores(x, ei, und)
        loss = F.cross_entropy(clf(h)[train], y[train]) - 0.005 * scores.mean()
        loss.backward()
        opt.step()

    model.eval()
    with torch.no_grad():
        scores = model.edge_scores(x, ei, und).cpu()
    keep = budget_prune_unconstrained(und.cpu(), scores, removal_rate)
    return subgraph_from_undirected_edges(data, keep)


def mog_adapted_score_edges(
    data: Data,
    epochs: int = 30,
    device: Optional[torch.device] = None,
) -> Tuple[torch.Tensor, torch.Tensor, float]:
    """Train MoG-adapted once; return (undirected_edges, importance, train_seconds)."""
    import time

    device = device or torch.device("cpu")
    t0 = time.perf_counter()
    und = undirected_edge_list(data.edge_index)
    model = MoGAdaptedScorer(data.x.size(1)).to(device)
    clf = nn.Linear(64, int(data.y.max().item()) + 1).to(device)
    opt = torch.optim.Adam(list(model.parameters()) + list(clf.parameters()), lr=1e-3)
    x, y, ei = data.x.to(device), data.y.to(device), data.edge_index.to(device)
    train = data.train_mask.to(device)
    for _ in range(epochs):
        model.train()
        opt.zero_grad()
        h = F.relu(model.enc(x, ei))
        scores = model.edge_scores(x, ei, und)
        loss = F.cross_entropy(clf(h)[train], y[train]) - 0.005 * scores.mean()
        loss.backward()
        opt.step()
    model.eval()
    with torch.no_grad():
        imp = model.edge_scores(x, ei, und).cpu()
    return und.cpu(), imp, time.perf_counter() - t0
