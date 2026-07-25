from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.utils import coalesce

from src.tasks.node_classification import train_node_classifier
from src.utils.graph import undirected_edge_list, subgraph_from_undirected_edges


def _perturb_edges(data: Data, drop_rate: float, seed: int = 0) -> Data:
    und = undirected_edge_list(data.edge_index)
    m = und.size(1)
    rng = np.random.RandomState(seed)
    keep_n = max(1, int((1 - drop_rate) * m))
    idx = rng.choice(m, size=keep_n, replace=False)
    return subgraph_from_undirected_edges(data, und[:, idx])


def robustness_evaluation(
    sparsified: Data,
    device: Optional[torch.device] = None,
    epochs: int = 50,
) -> Dict[str, float]:
    """Evaluate Macro-F1 under edge/feature perturbations of the sparsified graph."""
    device = device or torch.device("cpu")
    out: Dict[str, float] = {}
    _, base = train_node_classifier(sparsified, kind="gcn", epochs=epochs, device=device)
    out["base_test_macro_f1"] = base["test_macro_f1"]

    noisy = _perturb_edges(sparsified, drop_rate=0.1, seed=1)
    for key in ("train_mask", "val_mask", "test_mask"):
        setattr(noisy, key, getattr(sparsified, key))
    _, m_edge = train_node_classifier(noisy, kind="gcn", epochs=epochs, device=device)
    out["edge_perturb_test_macro_f1"] = m_edge["test_macro_f1"]

    feat_masked = sparsified.clone()
    x = feat_masked.x.clone()
    rng = np.random.RandomState(2)
    mask = torch.from_numpy(rng.rand(*x.shape) < 0.1)
    x[mask] = 0
    feat_masked.x = x
    _, m_feat = train_node_classifier(feat_masked, kind="gcn", epochs=epochs, device=device)
    out["feature_mask_test_macro_f1"] = m_feat["test_macro_f1"]
    return out
