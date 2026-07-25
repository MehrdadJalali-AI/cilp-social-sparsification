from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv

from src.utils.graph import undirected_edge_list
from src.sparsification.constrained_pruning import budget_prune_unconstrained


class OriginalILPGCN(nn.Module):
    """Faithful reproduction of ILP-GCN edge scoring (inverse link score + dual weight).

    W_ILP = alpha / (S_GCN + eps)
    W_final = gamma * W_initial + (1 - gamma) * W_ILP
    Low W_final ⇒ removable (we convert to importance = W_final for ranking keep-high).
    """

    def __init__(self, in_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.conv1 = GCNConv(in_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.scorer = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def encode(self, x: Tensor, edge_index: Tensor) -> Tensor:
        h = F.relu(self.conv1(x, edge_index))
        return self.conv2(h, edge_index)

    def link_score(self, h: Tensor, edges: Tensor) -> Tensor:
        src, dst = edges
        z = torch.cat([h[src], h[dst]], dim=-1)
        return torch.sigmoid(self.scorer(z).squeeze(-1))

    def ilp_importance(
        self,
        h: Tensor,
        undirected_edges: Tensor,
        initial_weights: Optional[Tensor] = None,
        alpha: float = 1.0,
        gamma: float = 0.5,
        eps: float = 1e-6,
    ) -> Tensor:
        s = self.link_score(h, undirected_edges)
        w_ilp = alpha / (s + eps)
        # Normalize
        w_ilp = (w_ilp - w_ilp.min()) / (w_ilp.max() - w_ilp.min() + eps)
        if initial_weights is None:
            initial_weights = torch.ones_like(w_ilp)
        else:
            initial_weights = (initial_weights - initial_weights.min()) / (
                initial_weights.max() - initial_weights.min() + eps
            )
        w_final = gamma * initial_weights + (1 - gamma) * w_ilp
        return w_final


def train_ilp_link_predictor(
    model: OriginalILPGCN,
    data: Data,
    epochs: int = 50,
    lr: float = 1e-3,
    device: Optional[torch.device] = None,
) -> OriginalILPGCN:
    device = device or torch.device("cpu")
    model = model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    x = data.x.to(device)
    ei = data.edge_index.to(device)
    pos = undirected_edge_list(ei)
    # Simple negative sampling
    from torch_geometric.utils import negative_sampling

    model.train()
    for _ in range(epochs):
        opt.zero_grad()
        h = model.encode(x, ei)
        neg = negative_sampling(ei, num_nodes=data.num_nodes, num_neg_samples=pos.size(1))
        # make undirected form for neg: sort endpoints
        ns, nd = neg
        swap = ns > nd
        ns, nd = ns.clone(), nd.clone()
        ns[swap], nd[swap] = nd[swap], ns[swap]
        neg_u = torch.stack([ns, nd], dim=0)
        pos_s = model.link_score(h, pos)
        neg_s = model.link_score(h, neg_u)
        loss = F.binary_cross_entropy(pos_s, torch.ones_like(pos_s)) + F.binary_cross_entropy(
            neg_s, torch.zeros_like(neg_s)
        )
        loss.backward()
        opt.step()
    return model


def sparsify_with_ilp(
    data: Data,
    removal_rate: float,
    hidden_dim: int = 64,
    epochs: int = 50,
    device: Optional[torch.device] = None,
) -> Data:
    device = device or torch.device("cpu")
    model = OriginalILPGCN(data.x.size(1), hidden_dim)
    model = train_ilp_link_predictor(model, data, epochs=epochs, device=device)
    model.eval()
    with torch.no_grad():
        h = model.encode(data.x.to(device), data.edge_index.to(device))
        und = undirected_edge_list(data.edge_index.to(device))
        imp = model.ilp_importance(h, und)
    keep = budget_prune_unconstrained(und.cpu(), imp.cpu(), removal_rate)
    from src.utils.graph import subgraph_from_undirected_edges

    return subgraph_from_undirected_edges(data, keep)
