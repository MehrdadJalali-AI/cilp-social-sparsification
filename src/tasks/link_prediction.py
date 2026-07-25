from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import average_precision_score, roc_auc_score
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv

from src.utils.splits import split_edges_for_link_prediction


class LPEncoder(nn.Module):
    def __init__(self, in_dim: int, hidden: int = 64):
        super().__init__()
        self.conv1 = GCNConv(in_dim, hidden)
        self.conv2 = GCNConv(hidden, hidden)

    def forward(self, x, edge_index):
        h = F.relu(self.conv1(x, edge_index))
        return self.conv2(h, edge_index)


class MLPDecoder(nn.Module):
    def __init__(self, hidden: int = 64):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(2 * hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, h, edges):
        src, dst = edges
        return self.mlp(torch.cat([h[src], h[dst]], -1)).squeeze(-1)


def _scores(h, edges, decoder: str, mlp: Optional[MLPDecoder] = None):
    src, dst = edges
    if decoder == "dot":
        return (h[src] * h[dst]).sum(-1)
    assert mlp is not None
    return mlp(h, edges)


def evaluate_link_prediction(
    data: Data,
    sparsified_train_edge_index: torch.Tensor,
    splits: Dict[str, torch.Tensor],
    epochs: int = 80,
    decoder: str = "dot",
    device: Optional[torch.device] = None,
) -> Dict[str, float]:
    """Train LP on sparsified training edges only; evaluate val/test with held-out edges."""
    device = device or torch.device("cpu")
    enc = LPEncoder(data.x.size(1)).to(device)
    mlp = MLPDecoder().to(device) if decoder == "mlp" else None
    params = list(enc.parameters()) + (list(mlp.parameters()) if mlp else [])
    opt = torch.optim.Adam(params, lr=1e-2)
    x = data.x.to(device)
    train_ei = sparsified_train_edge_index.to(device)
    pos = splits["train_pos"].to(device)
    neg = splits["train_neg"].to(device)

    for _ in range(epochs):
        enc.train()
        if mlp:
            mlp.train()
        opt.zero_grad()
        h = enc(x, train_ei)
        s_pos = _scores(h, pos, decoder, mlp)
        s_neg = _scores(h, neg, decoder, mlp)
        loss = F.binary_cross_entropy_with_logits(
            torch.cat([s_pos, s_neg]),
            torch.cat([torch.ones_like(s_pos), torch.zeros_like(s_neg)]),
        )
        loss.backward()
        opt.step()

    enc.eval()
    if mlp:
        mlp.eval()
    with torch.no_grad():
        h = enc(x, train_ei)
        out = {}
        for split in ("val", "test"):
            sp = _scores(h, splits[f"{split}_pos"].to(device), decoder, mlp).cpu().numpy()
            sn = _scores(h, splits[f"{split}_neg"].to(device), decoder, mlp).cpu().numpy()
            y = np.concatenate([np.ones(len(sp)), np.zeros(len(sn))])
            s = np.concatenate([sp, sn])
            out[f"{split}_auc"] = float(roc_auc_score(y, s))
            out[f"{split}_ap"] = float(average_precision_score(y, s))
            # Hits@K / MRR crude
            k = min(50, len(sp))
            ranks = []
            for i in range(len(sp)):
                rank = 1 + (sn >= sp[i]).sum()
                ranks.append(rank)
            ranks = np.array(ranks)
            out[f"{split}_mrr"] = float(np.mean(1.0 / ranks))
            out[f"{split}_hits@{k}"] = float(np.mean(ranks <= k))
    return out


def prepare_lp_splits(data: Data, seed: int = 0) -> Dict[str, torch.Tensor]:
    return split_edges_for_link_prediction(data.edge_index, data.num_nodes, seed=seed)
