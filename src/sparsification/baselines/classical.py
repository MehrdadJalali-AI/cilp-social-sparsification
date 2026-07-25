from __future__ import annotations

from typing import Callable, Dict, Optional

import networkx as nx
import numpy as np
import torch
from torch import Tensor
from torch_geometric.data import Data
from torch_geometric.utils import degree, to_networkx

from src.models.edge_encoder import structural_edge_features
from src.sparsification.constrained_pruning import budget_prune_unconstrained
from src.utils.graph import subgraph_from_undirected_edges, undirected_edge_list


def _score_and_prune(
    data: Data,
    scores: Tensor,
    removal_rate: float,
) -> Data:
    und = undirected_edge_list(data.edge_index)
    keep = budget_prune_unconstrained(und, scores, removal_rate)
    return subgraph_from_undirected_edges(data, keep)


def random_edge_removal(data: Data, removal_rate: float, seed: int = 0) -> Data:
    und = undirected_edge_list(data.edge_index)
    m = und.size(1)
    rng = np.random.RandomState(seed)
    scores = torch.from_numpy(rng.rand(m)).float()
    return _score_and_prune(data, scores, removal_rate)


def heuristic_sparsify(
    data: Data,
    removal_rate: float,
    heuristic: str = "jaccard",
) -> Data:
    und = undirected_edge_list(data.edge_index)
    feats = structural_edge_features(data.edge_index, data.num_nodes, und)
    key_map = {
        "common_neighbors": "common_neighbors",
        "jaccard": "jaccard",
        "adamic_adar": "adamic_adar",
        "resource_allocation": "resource_allocation",
        "preferential_attachment": "preferential_attachment",
        "degree_sum": "degree_sum",
        "degree_product": "degree_product",
    }
    if heuristic not in key_map:
        raise ValueError(heuristic)
    scores = feats[key_map[heuristic]].float()
    return _score_and_prune(data, scores, removal_rate)


def edge_betweenness_sparsify(data: Data, removal_rate: float) -> Data:
    G = to_networkx(data, to_undirected=True)
    # Approximate betweenness for larger graphs
    k = min(100, data.num_nodes)
    bw = nx.edge_betweenness_centrality(G, k=k, seed=0)
    und = undirected_edge_list(data.edge_index)
    scores = []
    for u, v in zip(und[0].tolist(), und[1].tolist()):
        scores.append(bw.get((u, v), bw.get((v, u), 0.0)))
    return _score_and_prune(data, torch.tensor(scores).float(), removal_rate)


def spanning_tree_backbone(data: Data, removal_rate: float) -> Data:
    """Keep a maximum spanning tree first, then fill budget with high-degree-sum edges."""
    und = undirected_edge_list(data.edge_index)
    feats = structural_edge_features(data.edge_index, data.num_nodes, und)
    scores = feats["degree_sum"].float()
    # Force tree edges to have high importance
    G = to_networkx(data, to_undirected=True)
    # Use degree product as weight
    for i, (u, v) in enumerate(zip(und[0].tolist(), und[1].tolist())):
        G[u][v]["w"] = float(scores[i])
    T = nx.maximum_spanning_tree(G, weight="w")
    tree = set(frozenset(e) for e in T.edges())
    boost = scores.clone()
    for i, (u, v) in enumerate(zip(und[0].tolist(), und[1].tolist())):
        if frozenset((u, v)) in tree:
            boost[i] = boost[i] + boost.max() + 1
    return _score_and_prune(data, boost, removal_rate)


def effective_resistance_proxy(data: Data, removal_rate: float) -> Data:
    """Degree-based effective-resistance motivated proxy (DSpar-like signal)."""
    und = undirected_edge_list(data.edge_index)
    deg = degree(data.edge_index[0], num_nodes=data.num_nodes)
    src, dst = und
    # Higher resistance edges (low deg endpoints) more likely redundant under some regimes;
    # for sparsification preserving spectral structure, prefer keeping low-resistance (high deg product)
    scores = 1.0 / (1.0 / (deg[src] + 1e-6) + 1.0 / (deg[dst] + 1e-6))
    return _score_and_prune(data, scores.float(), removal_rate)


def dropedge_export(data: Data, removal_rate: float, seed: int = 0) -> Data:
    """Persistent DropEdge-style random sparsified graph at fixed retention."""
    return random_edge_removal(data, removal_rate, seed=seed)


CLASSICAL_METHODS: Dict[str, Callable[..., Data]] = {
    "random": random_edge_removal,
    "degree_sum": lambda d, r, **kw: heuristic_sparsify(d, r, "degree_sum"),
    "degree_product": lambda d, r, **kw: heuristic_sparsify(d, r, "degree_product"),
    "common_neighbors": lambda d, r, **kw: heuristic_sparsify(d, r, "common_neighbors"),
    "jaccard": lambda d, r, **kw: heuristic_sparsify(d, r, "jaccard"),
    "adamic_adar": lambda d, r, **kw: heuristic_sparsify(d, r, "adamic_adar"),
    "resource_allocation": lambda d, r, **kw: heuristic_sparsify(d, r, "resource_allocation"),
    "preferential_attachment": lambda d, r, **kw: heuristic_sparsify(d, r, "preferential_attachment"),
    "edge_betweenness": edge_betweenness_sparsify,
    "spanning_tree": spanning_tree_backbone,
    "effective_resistance": effective_resistance_proxy,
    "dropedge": dropedge_export,
}
