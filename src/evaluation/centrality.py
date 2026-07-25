from __future__ import annotations

from typing import Dict

import networkx as nx
import numpy as np
from scipy.stats import kendalltau, spearmanr
from torch_geometric.data import Data
from torch_geometric.utils import degree, to_networkx


def _topk_overlap(a: np.ndarray, b: np.ndarray, frac: float) -> float:
    k = max(1, int(round(frac * len(a))))
    ta = set(np.argsort(-a)[:k].tolist())
    tb = set(np.argsort(-b)[:k].tolist())
    return len(ta & tb) / k


def centrality_preservation(original: Data, sparsified: Data) -> Dict[str, float]:
    G0 = to_networkx(original, to_undirected=True)
    G1 = to_networkx(sparsified, to_undirected=True)
    n = original.num_nodes
    deg0 = degree(original.edge_index[0], num_nodes=n).numpy()
    deg1 = degree(sparsified.edge_index[0], num_nodes=n).numpy()
    pr0 = np.array([nx.pagerank(G0)[i] for i in range(n)])
    pr1 = np.array([nx.pagerank(G1)[i] for i in range(n)])
    out: Dict[str, float] = {}
    for name, a, b in (("degree", deg0, deg1), ("pagerank", pr0, pr1)):
        out[f"{name}_spearman"] = float(spearmanr(a, b).correlation or 0.0)
        out[f"{name}_kendall"] = float(kendalltau(a, b).correlation or 0.0)
        for frac, tag in ((0.01, "1pct"), (0.05, "5pct"), (0.10, "10pct")):
            out[f"{name}_topk_{tag}"] = float(_topk_overlap(a, b, frac))
    return out
