from __future__ import annotations

from typing import Dict

import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.utils import degree


def fairness_metrics(original: Data, sparsified: Data) -> Dict[str, float]:
    """Class-wise connectivity preservation using labels for evaluation only."""
    y = original.y.cpu().numpy()
    classes = np.unique(y)
    deg0 = degree(original.edge_index[0], num_nodes=original.num_nodes).numpy()
    deg1 = degree(sparsified.edge_index[0], num_nodes=sparsified.num_nodes).numpy()
    out: Dict[str, float] = {}
    isol0 = (deg0 == 0)
    isol1 = (deg1 == 0)
    for c in classes:
        mask = y == c
        out[f"class_{int(c)}_degree_retention"] = float(deg1[mask].mean() / (deg0[mask].mean() + 1e-8))
        out[f"class_{int(c)}_isolation_rate"] = float(isol1[mask].mean())
    # Minority = smallest class
    sizes = [(c, (y == c).sum()) for c in classes]
    minority = min(sizes, key=lambda t: t[1])[0]
    out["minority_class"] = float(minority)
    out["minority_degree_retention"] = out[f"class_{int(minority)}_degree_retention"]
    out["minority_isolation_rate"] = out[f"class_{int(minority)}_isolation_rate"]
    return out
