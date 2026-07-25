from __future__ import annotations

from typing import Dict, Optional, Sequence

import numpy as np
import torch
from torch import Tensor
from torch_geometric.data import Data
from torch_geometric.utils import degree

from src.utils.graph import undirected_edge_list


def stratified_edge_sample(
    data: Data,
    undirected_edges: Tensor,
    n_sample: int = 200,
    seed: int = 0,
    y_train_val: Optional[Tensor] = None,
) -> Tensor:
    """Stratified subset of undirected edges for the counterfactual teacher."""
    rng = np.random.RandomState(seed)
    src, dst = undirected_edges.cpu()
    m = undirected_edges.size(1)
    if m <= n_sample:
        return torch.arange(m)

    deg = degree(data.edge_index[0], num_nodes=data.num_nodes).cpu().numpy()
    deg_u, deg_v = deg[src.numpy()], deg[dst.numpy()]
    deg_score = deg_u + deg_v
    # Feature similarity buckets
    if data.x is not None:
        x = data.x.cpu().float()
        xi, xj = x[src], x[dst]
        cos = ((xi * xj).sum(-1) / (xi.norm(dim=-1) * xj.norm(dim=-1) + 1e-8)).numpy()
    else:
        cos = np.zeros(m)

    # Simple strata via quantile bins on degree_sum and cosine
    def bins(arr: np.ndarray, k: int = 4) -> np.ndarray:
        qs = np.quantile(arr, np.linspace(0, 1, k + 1))
        qs[0] -= 1e-6
        return np.digitize(arr, qs[1:-1], right=True)

    b1 = bins(deg_score)
    b2 = bins(cos)
    strata = b1 * 10 + b2
    selected: list[int] = []
    remaining = n_sample
    keys = np.unique(strata)
    rng.shuffle(keys)
    per = max(1, n_sample // max(len(keys), 1))
    for k in keys:
        idx = np.where(strata == k)[0]
        take = min(len(idx), per, remaining)
        if take <= 0:
            continue
        chosen = rng.choice(idx, size=take, replace=False)
        selected.extend(chosen.tolist())
        remaining = n_sample - len(selected)
        if remaining <= 0:
            break
    if len(selected) < n_sample:
        extra = rng.choice(np.setdiff1d(np.arange(m), selected), size=n_sample - len(selected), replace=False)
        selected.extend(extra.tolist())
    return torch.tensor(selected[:n_sample], dtype=torch.long)


def normalize_scores(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    if x.size == 0:
        return x
    lo, hi = np.nanmin(x), np.nanmax(x)
    if hi - lo < 1e-12:
        return np.zeros_like(x)
    return (x - lo) / (hi - lo)
