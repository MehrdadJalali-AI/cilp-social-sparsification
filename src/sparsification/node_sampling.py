from __future__ import annotations

from typing import Dict, List, Optional, Set

import networkx as nx
import numpy as np
import torch
from torch import Tensor
from torch_geometric.data import Data
from torch_geometric.utils import degree, to_networkx

from src.models.node_mass import black_hole_mass_unmodified, normalize_columns


def _induced_subgraph(data: Data, nodes: Tensor) -> Data:
    node_set = set(nodes.tolist())
    mapping = {old: i for i, old in enumerate(nodes.tolist())}
    src, dst = data.edge_index
    mask = torch.tensor(
        [(int(u) in node_set and int(v) in node_set) for u, v in zip(src.tolist(), dst.tolist())],
        dtype=torch.bool,
    )
    ei = data.edge_index[:, mask]
    new_ei = torch.stack(
        [
            torch.tensor([mapping[int(u)] for u in ei[0].tolist()]),
            torch.tensor([mapping[int(v)] for v in ei[1].tolist()]),
        ]
    )
    out = Data(
        x=data.x[nodes] if data.x is not None else None,
        edge_index=new_ei,
        y=data.y[nodes] if data.y is not None else None,
    )
    return out


def sample_nodes_by_score(scores: Tensor, keep_ratio: float) -> Tensor:
    n = scores.numel()
    k = max(1, int(round(keep_ratio * n)))
    return torch.argsort(scores, descending=True)[:k]


def random_node_sample(data: Data, keep_ratio: float, seed: int = 0) -> Data:
    rng = np.random.RandomState(seed)
    n = data.num_nodes
    k = max(1, int(round(keep_ratio * n)))
    nodes = torch.from_numpy(rng.choice(n, size=k, replace=False)).long()
    return _induced_subgraph(data, nodes)


def degree_node_sample(data: Data, keep_ratio: float) -> Data:
    deg = degree(data.edge_index[0], num_nodes=data.num_nodes)
    return _induced_subgraph(data, sample_nodes_by_score(deg, keep_ratio))


def pagerank_node_sample(data: Data, keep_ratio: float) -> Data:
    G = to_networkx(data, to_undirected=True)
    pr = nx.pagerank(G)
    scores = torch.tensor([pr[i] for i in range(data.num_nodes)], dtype=torch.float)
    return _induced_subgraph(data, sample_nodes_by_score(scores, keep_ratio))


def kcore_node_sample(data: Data, keep_ratio: float) -> Data:
    G = to_networkx(data, to_undirected=True)
    core = nx.core_number(G)
    scores = torch.tensor([core[i] for i in range(data.num_nodes)], dtype=torch.float)
    return _induced_subgraph(data, sample_nodes_by_score(scores, keep_ratio))


def black_hole_node_sample(data: Data, keep_ratio: float) -> Data:
    """Track B: unmodified-style Black Hole representative node sampling."""
    G = to_networkx(data, to_undirected=True)
    deg = degree(data.edge_index[0], num_nodes=data.num_nodes)
    pr_dict = nx.pagerank(G)
    pr = torch.tensor([pr_dict[i] for i in range(data.num_nodes)], dtype=torch.float)
    cl_dict = nx.clustering(G)
    cl = torch.tensor([cl_dict[i] for i in range(data.num_nodes)], dtype=torch.float)
    mass = black_hole_mass_unmodified(deg, pr, cl)
    return _induced_subgraph(data, sample_nodes_by_score(mass, keep_ratio))


NODE_SAMPLERS = {
    "random": random_node_sample,
    "degree": degree_node_sample,
    "pagerank": pagerank_node_sample,
    "kcore": kcore_node_sample,
    "black_hole": black_hole_node_sample,
}
