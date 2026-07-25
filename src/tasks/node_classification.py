from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    cohen_kappa_score,
    f1_score,
)
from torch_geometric.data import Data
from torch_geometric.nn import GATConv, GCNConv, SAGEConv, SGConv, APPNP


class DownstreamGNN(nn.Module):
    def __init__(self, in_dim: int, n_classes: int, kind: str = "gcn", hidden: int = 64):
        super().__init__()
        self.kind = kind
        if kind == "gcn":
            self.conv1 = GCNConv(in_dim, hidden)
            self.conv2 = GCNConv(hidden, n_classes)
        elif kind == "sage":
            self.conv1 = SAGEConv(in_dim, hidden)
            self.conv2 = SAGEConv(hidden, n_classes)
        elif kind == "gatv2":
            from torch_geometric.nn import GATv2Conv

            self.conv1 = GATv2Conv(in_dim, hidden // 4, heads=4, concat=True)
            self.conv2 = GATv2Conv(hidden, n_classes, heads=1, concat=False)
        elif kind == "sgc":
            self.conv1 = SGConv(in_dim, n_classes, K=2)
            self.conv2 = None
        elif kind == "appnp":
            self.lin = nn.Linear(in_dim, n_classes)
            self.prop = APPNP(K=10, alpha=0.1)
            self.conv1 = self.conv2 = None
        elif kind == "mlp":
            self.mlp = nn.Sequential(
                nn.Linear(in_dim, hidden),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(hidden, n_classes),
            )
            self.conv1 = self.conv2 = None
        else:
            raise ValueError(kind)

    def forward(self, x, edge_index):
        if self.kind == "mlp":
            return self.mlp(x)
        if self.kind == "sgc":
            return self.conv1(x, edge_index)
        if self.kind == "appnp":
            return self.prop(self.lin(x), edge_index)
        h = F.relu(self.conv1(x, edge_index))
        h = F.dropout(h, p=0.5, training=self.training)
        return self.conv2(h, edge_index)


def train_node_classifier(
    data: Data,
    kind: str = "gcn",
    epochs: int = 100,
    lr: float = 1e-2,
    device: Optional[torch.device] = None,
) -> tuple[DownstreamGNN, Dict[str, float]]:
    device = device or torch.device("cpu")
    n_classes = int(data.y.max().item()) + 1
    model = DownstreamGNN(data.x.size(1), n_classes, kind=kind).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=5e-4)
    x, y, ei = data.x.to(device), data.y.to(device), data.edge_index.to(device)
    best_val = -1.0
    best_state = None
    for _ in range(epochs):
        model.train()
        opt.zero_grad()
        logits = model(x, ei)
        loss = F.cross_entropy(logits[data.train_mask], y[data.train_mask])
        loss.backward()
        opt.step()
        model.eval()
        with torch.no_grad():
            pred = model(x, ei).argmax(-1)
            val_f1 = f1_score(
                y[data.val_mask].cpu(),
                pred[data.val_mask].cpu(),
                average="macro",
                zero_division=0,
            )
            if val_f1 > best_val:
                best_val = val_f1
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    if best_state:
        model.load_state_dict(best_state)
    metrics = evaluate_node_classification(model, data, device)
    return model, metrics


@torch.no_grad()
def evaluate_node_classification(
    model: nn.Module,
    data: Data,
    device: Optional[torch.device] = None,
) -> Dict[str, float]:
    device = device or torch.device("cpu")
    model = model.to(device)
    model.eval()
    logits = model(data.x.to(device), data.edge_index.to(device))
    pred = logits.argmax(-1).cpu().numpy()
    y = data.y.cpu().numpy()
    out = {}
    for split, mask in (
        ("train", data.train_mask.numpy()),
        ("val", data.val_mask.numpy()),
        ("test", data.test_mask.numpy()),
    ):
        yt, pt = y[mask], pred[mask]
        out[f"{split}_accuracy"] = float(accuracy_score(yt, pt))
        out[f"{split}_macro_f1"] = float(f1_score(yt, pt, average="macro", zero_division=0))
        out[f"{split}_micro_f1"] = float(f1_score(yt, pt, average="micro", zero_division=0))
        out[f"{split}_weighted_f1"] = float(f1_score(yt, pt, average="weighted", zero_division=0))
        out[f"{split}_balanced_accuracy"] = float(balanced_accuracy_score(yt, pt))
        out[f"{split}_kappa"] = float(cohen_kappa_score(yt, pt))
        # worst-class F1
        per = f1_score(yt, pt, average=None, zero_division=0)
        out[f"{split}_worst_class_f1"] = float(np.min(per)) if len(per) else 0.0
    return out
