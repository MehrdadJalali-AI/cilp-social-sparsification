from __future__ import annotations

from typing import Dict

import networkx as nx
import numpy as np
from scipy.stats import ks_2samp, wasserstein_distance
from torch_geometric.data import Data
from torch_geometric.utils import degree, to_networkx

from src.utils.graph import graph_density, undirected_edge_list


def structural_metrics(original: Data, sparsified: Data) -> Dict[str, float]:
    und0 = undirected_edge_list(original.edge_index)
    und1 = undirected_edge_list(sparsified.edge_index)
    n = original.num_nodes
    G0 = to_networkx(original, to_undirected=True)
    G1 = to_networkx(sparsified, to_undirected=True)
    deg0 = degree(original.edge_index[0], num_nodes=n).cpu().numpy()
    deg1 = degree(sparsified.edge_index[0], num_nodes=n).cpu().numpy()
    ks = ks_2samp(deg0, deg1).statistic
    wass = wasserstein_distance(deg0, deg1)
    bridges0 = set(frozenset(e) for e in nx.bridges(G0))
    bridges1 = set(frozenset(e) for e in nx.bridges(G1))
    bridge_pres = len(bridges0 & bridges1) / max(len(bridges0), 1)
    giant0 = max(nx.connected_components(G0), key=len)
    giant1 = max(nx.connected_components(G1), key=len) if G1.number_of_nodes() else set()
    try:
        clustering1 = nx.average_clustering(G1)
    except Exception:
        clustering1 = float("nan")
    try:
        assort1 = nx.degree_assortativity_coefficient(G1)
    except Exception:
        assort1 = float("nan")
    return {
        "retained_edge_ratio": und1.size(1) / max(und0.size(1), 1),
        "density": graph_density(n, und1.size(1)),
        "average_degree": float(deg1.mean()),
        "degree_ks": float(ks),
        "degree_wasserstein": float(wass),
        "clustering": float(clustering1),
        "assortativity": float(assort1),
        "num_components": float(nx.number_connected_components(G1)),
        "giant_component_ratio": len(giant1) / n,
        "bridge_preservation": float(bridge_pres),
        "triangles_sparse": float(sum(nx.triangles(G1).values()) / 3),
    }
