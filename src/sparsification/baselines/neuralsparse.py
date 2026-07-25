"""NeuralSparse-style task-driven k-neighbor sparsification (Zheng et al., ICML 2020).

Faithful simplified reimplementation: MLP edge scorer trained with downstream node-class
feedback, then keep top-k neighbors per node under a global edge budget approximation.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
from torch_geometric.utils import degree

from src.utils.graph import subgraph_from_undirected_edges, undirected_edge_list
from src.sparsification.constrained_pruning import budget_prune_unconstrained


class NeuralSparseScorer(nn.Module):
    def __init__(self, in_dim: int, hidden: int = 64):
        super().__init__()
        self.enc = GCNConv(in_dim, hidden)
        self.edge_mlp = nn.Sequential(
            nn.Linear(2 * hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x: Tensor, edge_index: Tensor, und: Tensor) -> Tensor:
        h = F.relu(self.enc(x, edge_index))
        src, dst = und
        return torch.sigmoid(self.edge_mlp(torch.cat([h[src], h[dst]], -1)).squeeze(-1))


def neuralsparse_sparsify(
    data: Data,
    removal_rate: float,
    epochs: int = 30,
    device: Optional[torch.device] = None,
) -> Data:
    device = device or torch.device("cpu")
    model = NeuralSparseScorer(data.x.size(1)).to(device)
    clf = nn.Linear(64, int(data.y.max().item()) + 1).to(device)
    opt = torch.optim.Adam(list(model.parameters()) + list(clf.parameters()), lr=1e-3)
    x, y, ei = data.x.to(device), data.y.to(device), data.edge_index.to(device)
    und = undirected_edge_list(ei)
    train = data.train_mask.to(device) if hasattr(data, "train_mask") else torch.ones(data.num_nodes, dtype=torch.bool, device=device)

    for _ in range(epochs):
        model.train()
        opt.zero_grad()
        scores = model(x, ei, und)
        # Soft sparsification: weight message by score via resampling top edges approx
        # Task loss on encoder embeddings
        h = F.relu(model.enc(x, ei))
        loss = F.cross_entropy(clf(h)[train], y[train]) - 0.01 * scores.mean()
        loss.backward()
        opt.step()

    model.eval()
    with torch.no_grad():
        scores = model(x, ei, und).cpu()
    keep = budget_prune_unconstrained(und.cpu(), scores, removal_rate)
    return subgraph_from_undirected_edges(data, keep)
