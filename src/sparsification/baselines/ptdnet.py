"""PTDNet-style parameterized topological denoising baseline."""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv

from src.utils.graph import subgraph_from_undirected_edges, undirected_edge_list
from src.sparsification.constrained_pruning import budget_prune_unconstrained


class PTDNetScorer(nn.Module):
    def __init__(self, in_dim: int, hidden: int = 64):
        super().__init__()
        self.conv1 = GCNConv(in_dim, hidden)
        self.conv2 = GCNConv(hidden, hidden)
        self.denoise = nn.Sequential(
            nn.Linear(2 * hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x: Tensor, edge_index: Tensor, und: Tensor) -> tuple[Tensor, Tensor]:
        h = F.relu(self.conv1(x, edge_index))
        h = self.conv2(h, edge_index)
        src, dst = und
        logits = self.denoise(torch.cat([h[src], h[dst]], -1)).squeeze(-1)
        return h, torch.sigmoid(logits)


def ptdnet_sparsify(
    data: Data,
    removal_rate: float,
    epochs: int = 30,
    device: Optional[torch.device] = None,
) -> Data:
    device = device or torch.device("cpu")
    model = PTDNetScorer(data.x.size(1)).to(device)
    n_cls = int(data.y.max().item()) + 1
    clf = nn.Linear(64, n_cls).to(device)
    opt = torch.optim.Adam(list(model.parameters()) + list(clf.parameters()), lr=1e-3)
    x, y, ei = data.x.to(device), data.y.to(device), data.edge_index.to(device)
    und = undirected_edge_list(ei)
    train = data.train_mask.to(device) if hasattr(data, "train_mask") else torch.ones(data.num_nodes, dtype=torch.bool, device=device)

    for _ in range(epochs):
        opt.zero_grad()
        h, scores = model(x, ei, und)
        task = F.cross_entropy(clf(h)[train], y[train])
        # Nuclear-norm / low-rank proxy: encourage sparse edge scores
        sparse_reg = scores.mean()
        loss = task + 0.1 * sparse_reg
        loss.backward()
        opt.step()

    with torch.no_grad():
        _, scores = model(x, ei, und)
    keep = budget_prune_unconstrained(und.cpu(), scores.cpu(), removal_rate)
    return subgraph_from_undirected_edges(data, keep)
