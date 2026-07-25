from __future__ import annotations

from typing import Dict

import networkx as nx
import numpy as np
from torch_geometric.data import Data
from torch_geometric.utils import to_networkx


def community_metrics(original: Data, sparsified: Data, seed: int = 0) -> Dict[str, float]:
    try:
        import community as community_louvain
    except ImportError:
        return {"modularity_sparse": float("nan"), "nmi": float("nan"), "ari": float("nan")}

    from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

    G0 = to_networkx(original, to_undirected=True)
    G1 = to_networkx(sparsified, to_undirected=True)
    p0 = community_louvain.best_partition(G0, random_state=seed)
    p1 = community_louvain.best_partition(G1, random_state=seed)
    nodes = list(range(original.num_nodes))
    # Nodes missing in sparsified (shouldn't in Track A)
    labels0 = [p0.get(i, -1) for i in nodes]
    labels1 = [p1.get(i, -1) for i in nodes]
    q1 = community_louvain.modularity(p1, G1) if G1.number_of_edges() else 0.0
    return {
        "modularity_sparse": float(q1),
        "nmi": float(normalized_mutual_info_score(labels0, labels1)),
        "ari": float(adjusted_rand_score(labels0, labels1)),
        "num_communities_sparse": float(len(set(labels1))),
    }
