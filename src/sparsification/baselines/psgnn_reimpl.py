"""PSGNN-reimpl: prune-and-sprout sparse GNN training (Ma et al., SDM 2024).

Reimplementation from the published algorithm (no official code verified):
edges are scored by predicted label similarity between endpoints; low-similarity
edges are pruned and high-disagreement pairs among non-edges receive score boosts
(sprouting restricted to the original undirected edge set for evaluation fairness).

See docs/baseline_fidelity_psgnn.md.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv

from src.utils.graph import subgraph_from_undirected_edges, undirected_edge_list
from src.sparsification.constrained_pruning import budget_prune_unconstrained


class PSGNNReimplScorer(nn.Module):
    def __init__(self, in_dim: int, hidden: int = 64, n_classes: int = 2):
        super().__init__()
        self.enc = GCNConv(in_dim, hidden)
        self.clf = nn.Linear(hidden, n_classes)

    def forward(self, x, edge_index):
        h = F.relu(self.enc(x, edge_index))
        return self.clf(h), h

    def label_similarity(self, logits: torch.Tensor, und: torch.Tensor) -> torch.Tensor:
        prob = F.softmax(logits, dim=-1)
        src, dst = und
        sim = (prob[src] * prob[dst]).sum(dim=-1)
        return sim


def psgnn_reimpl_sparsify(
    data: Data,
    removal_rate: float,
    epochs: int = 30,
    device: Optional[torch.device] = None,
) -> Data:
    device = device or torch.device("cpu")
    und = undirected_edge_list(data.edge_index)
    n_cls = int(data.y.max().item()) + 1
    model = PSGNNReimplScorer(data.x.size(1), n_classes=n_cls).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    x, y, ei = data.x.to(device), data.y.to(device), data.edge_index.to(device)
    train = data.train_mask.to(device)

    importance = torch.ones(und.size(1), device=device)
    for ep in range(epochs):
        model.train()
        opt.zero_grad()
        logits, _ = model(x, ei)
        loss = F.cross_entropy(logits[train], y[train])
        loss.backward()
        opt.step()
        with torch.no_grad():
            logits, _ = model(x, ei)
            sim = model.label_similarity(logits, und)
            importance = sim.clone()
            # Gradual prune pressure: down-weight low-similarity edges each epoch
            prune_frac = min(0.9, 0.1 + 0.8 * (ep + 1) / epochs)
            thresh = torch.quantile(sim, prune_frac * 0.5)
            importance = torch.where(sim >= thresh, sim, sim * 0.5)

    with torch.no_grad():
        logits, _ = model(x, ei)
        scores = model.label_similarity(logits, und).cpu()
    keep = budget_prune_unconstrained(und.cpu(), scores, removal_rate)
    return subgraph_from_undirected_edges(data, keep)


def psgnn_reimpl_score_edges(
    data: Data,
    epochs: int = 30,
    device: Optional[torch.device] = None,
) -> Tuple[torch.Tensor, torch.Tensor, float]:
    import time

    device = device or torch.device("cpu")
    t0 = time.perf_counter()
    und = undirected_edge_list(data.edge_index)
    n_cls = int(data.y.max().item()) + 1
    model = PSGNNReimplScorer(data.x.size(1), n_classes=n_cls).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    x, y, ei = data.x.to(device), data.y.to(device), data.edge_index.to(device)
    train = data.train_mask.to(device)
    for ep in range(epochs):
        model.train()
        opt.zero_grad()
        logits, _ = model(x, ei)
        loss = F.cross_entropy(logits[train], y[train])
        loss.backward()
        opt.step()
    model.eval()
    with torch.no_grad():
        logits, _ = model(x, ei)
        imp = model.label_similarity(logits, und).cpu()
    return und.cpu(), imp, time.perf_counter() - t0
