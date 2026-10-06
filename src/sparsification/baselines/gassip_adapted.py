"""GASSIP-adapted: curriculum edge-mask sparsification without NAS (Xie et al., KDD 2024).

Faithful elements: differentiable edge importance mask trained with curriculum
increasing target sparsity; difficulty prior from endpoint degree (architecture-
agnostic proxy for edge-removing difficulty). NAS / operation pruning from GASSIP
is omitted; fixed 2-layer GCN used per Stage A protocol.

NOT the original GASSIP method — see docs/baseline_fidelity_gassip.md.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
from torch_geometric.utils import degree

from src.utils.graph import subgraph_from_undirected_edges, undirected_edge_list
from src.sparsification.constrained_pruning import budget_prune_unconstrained


class GASSIPAdaptedScorer(nn.Module):
    def __init__(self, in_dim: int, hidden: int = 64):
        super().__init__()
        self.enc = GCNConv(in_dim, hidden)
        self.edge_mlp = nn.Sequential(
            nn.Linear(2 * hidden + 2, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        und: torch.Tensor,
        difficulty: torch.Tensor,
    ) -> torch.Tensor:
        h = F.relu(self.enc(x, edge_index))
        src, dst = und
        feat = torch.cat([h[src], h[dst], difficulty.unsqueeze(-1), (1 - difficulty).unsqueeze(-1)], -1)
        return torch.sigmoid(self.edge_mlp(feat).squeeze(-1))


def _edge_difficulty(edge_index: torch.Tensor, und: torch.Tensor, num_nodes: int) -> torch.Tensor:
    deg = degree(edge_index[0], num_nodes=num_nodes).float()
    src, dst = und
    d = (deg[src] * deg[dst]).sqrt()
    d = d / (d.max() + 1e-8)
    return d


def gassip_adapted_sparsify(
    data: Data,
    removal_rate: float,
    epochs: int = 30,
    device: Optional[torch.device] = None,
) -> Data:
    device = device or torch.device("cpu")
    und = undirected_edge_list(data.edge_index)
    model = GASSIPAdaptedScorer(data.x.size(1)).to(device)
    clf = nn.Linear(64, int(data.y.max().item()) + 1).to(device)
    opt = torch.optim.Adam(list(model.parameters()) + list(clf.parameters()), lr=1e-3)
    x, y, ei = data.x.to(device), data.y.to(device), data.edge_index.to(device)
    train = data.train_mask.to(device)
    diff = _edge_difficulty(ei, und, data.num_nodes).to(device)

    for ep in range(epochs):
        model.train()
        opt.zero_grad()
        h = F.relu(model.enc(x, ei))
        scores = model(x, ei, und, diff)
        # Curriculum: gradually encourage sparser masks (easy edges first)
        target_keep = 1.0 - min(0.9, 0.1 + 0.8 * (ep + 1) / epochs)
        curriculum_loss = F.mse_loss(scores.mean(), torch.tensor(target_keep, device=device))
        loss = (
            F.cross_entropy(clf(h)[train], y[train])
            + 0.1 * curriculum_loss
            - 0.01 * (scores * (1 - diff)).mean()
        )
        loss.backward()
        opt.step()

    model.eval()
    with torch.no_grad():
        scores = model(x, ei, und, diff).cpu()
    keep = budget_prune_unconstrained(und.cpu(), scores, removal_rate)
    return subgraph_from_undirected_edges(data, keep)


def gassip_adapted_score_edges(
    data: Data,
    epochs: int = 30,
    device: Optional[torch.device] = None,
) -> Tuple[torch.Tensor, torch.Tensor, float]:
    import time

    device = device or torch.device("cpu")
    t0 = time.perf_counter()
    und = undirected_edge_list(data.edge_index)
    model = GASSIPAdaptedScorer(data.x.size(1)).to(device)
    clf = nn.Linear(64, int(data.y.max().item()) + 1).to(device)
    opt = torch.optim.Adam(list(model.parameters()) + list(clf.parameters()), lr=1e-3)
    x, y, ei = data.x.to(device), data.y.to(device), data.edge_index.to(device)
    train = data.train_mask.to(device)
    diff = _edge_difficulty(ei, und, data.num_nodes).to(device)
    for ep in range(epochs):
        model.train()
        opt.zero_grad()
        h = F.relu(model.enc(x, ei))
        scores = model(x, ei, und, diff)
        target_keep = 1.0 - min(0.9, 0.1 + 0.8 * (ep + 1) / epochs)
        curriculum_loss = F.mse_loss(scores.mean(), torch.tensor(target_keep, device=device))
        loss = (
            F.cross_entropy(clf(h)[train], y[train])
            + 0.1 * curriculum_loss
            - 0.01 * (scores * (1 - diff)).mean()
        )
        loss.backward()
        opt.step()
    model.eval()
    with torch.no_grad():
        imp = model(x, ei, und, diff).cpu()
    return und.cpu(), imp, time.perf_counter() - t0
