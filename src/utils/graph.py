from __future__ import annotations

from typing import Optional, Tuple

import torch
from torch import Tensor
from torch_geometric.data import Data
from torch_geometric.utils import (
    coalesce,
    degree,
    remove_self_loops,
    to_undirected,
)


def clean_edge_index(
    edge_index: Tensor,
    num_nodes: Optional[int] = None,
    make_undirected: bool = True,
) -> Tuple[Tensor, dict]:
    """Remove self-loops and duplicates; optionally symmetrize.

    Returns cleaned edge_index and an audit dict.
    """
    n_raw = int(edge_index.size(1))
    ei, _ = remove_self_loops(edge_index)
    n_after_loops = int(ei.size(1))
    n_self_loops = n_raw - n_after_loops

    if make_undirected:
        ei = to_undirected(ei, num_nodes=num_nodes)

    ei, _ = coalesce(ei, None, num_nodes=num_nodes)
    n_clean = int(ei.size(1))
    # For undirected, each unique undirected edge appears twice.
    n_unique_undirected = n_clean // 2 if make_undirected else n_clean
    n_duplicates_removed = n_after_loops - (n_clean if not make_undirected else n_after_loops)
    # Approximate duplicate count on directed raw before undirected conversion:
    ei_dir, _ = coalesce(edge_index, None, num_nodes=num_nodes)
    n_dup_directed = int(edge_index.size(1)) - int(ei_dir.size(1))

    audit = {
        "directed_edge_entries_raw": n_raw,
        "self_loops": n_self_loops,
        "duplicate_directed_edges_approx": max(0, n_dup_directed),
        "directed_edge_entries_clean": n_clean,
        "unique_undirected_edges": n_unique_undirected,
    }
    return ei, audit


def undirected_edge_list(edge_index: Tensor) -> Tensor:
    """Return unique undirected edges as [2, E] with src < dst."""
    src, dst = edge_index
    mask = src < dst
    return edge_index[:, mask]


def graph_density(num_nodes: int, num_undirected_edges: int) -> float:
    if num_nodes < 2:
        return 0.0
    return (2.0 * num_undirected_edges) / (num_nodes * (num_nodes - 1))


def average_degree(num_nodes: int, num_directed_edges: int) -> float:
    if num_nodes == 0:
        return 0.0
    return num_directed_edges / float(num_nodes)


def subgraph_from_undirected_edges(
    data: Data,
    keep_undirected: Tensor,
) -> Data:
    """Build a new Data object retaining only listed undirected edges (src < dst)."""
    src, dst = keep_undirected
    edge_index = torch.stack(
        [torch.cat([src, dst]), torch.cat([dst, src])], dim=0
    )
    out = Data(
        x=data.x.clone() if data.x is not None else None,
        edge_index=edge_index,
        y=data.y.clone() if data.y is not None else None,
    )
    for key in ("train_mask", "val_mask", "test_mask"):
        if hasattr(data, key) and getattr(data, key) is not None:
            setattr(out, key, getattr(data, key).clone())
    return out


def node_degrees(edge_index: Tensor, num_nodes: int) -> Tensor:
    return degree(edge_index[0], num_nodes=num_nodes)
