from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import torch
from torch import Tensor
from torch_geometric.utils import degree

from src.utils.graph import undirected_edge_list


def _cosine(a: Tensor, b: Tensor, eps: float = 1e-8) -> Tensor:
    return (a * b).sum(dim=-1) / (a.norm(dim=-1) * b.norm(dim=-1) + eps)


def structural_edge_features(
    edge_index: Tensor,
    num_nodes: int,
    undirected_edges: Optional[Tensor] = None,
    lightweight: bool = False,
) -> Dict[str, Tensor]:
    """Compute structural features for undirected edges.

    Set ``lightweight=True`` to skip O(E·d) neighborhood overlaps (pilot/large graphs).
    """
    if undirected_edges is None:
        undirected_edges = undirected_edge_list(edge_index)
    src, dst = undirected_edges
    deg = degree(edge_index[0], num_nodes=num_nodes)
    pa = deg[src] * deg[dst]
    deg_sum = deg[src] + deg[dst]
    deg_prod = deg[src] * deg[dst]
    device = undirected_edges.device
    out = {
        "preferential_attachment": pa,
        "degree_sum": deg_sum,
        "degree_product": deg_prod,
        "deg_u": deg[src],
        "deg_v": deg[dst],
    }
    if lightweight:
        m = undirected_edges.size(1)
        zeros = torch.zeros(m, device=device)
        out.update(
            {
                "common_neighbors": zeros,
                "jaccard": zeros,
                "adamic_adar": zeros,
                "resource_allocation": zeros,
            }
        )
        return out

    adj: list[set[int]] = [set() for _ in range(num_nodes)]
    row, col = edge_index.cpu().numpy()
    for u, v in zip(row.tolist(), col.tolist()):
        if u != v:
            adj[u].add(v)

    src_np = src.cpu().numpy()
    dst_np = dst.cpu().numpy()
    cn = np.zeros(len(src_np), dtype=np.float32)
    jaccard = np.zeros(len(src_np), dtype=np.float32)
    aa = np.zeros(len(src_np), dtype=np.float32)
    ra = np.zeros(len(src_np), dtype=np.float32)
    for i, (u, v) in enumerate(zip(src_np, dst_np)):
        inter = adj[u].intersection(adj[v])
        union = adj[u].union(adj[v])
        cn[i] = len(inter)
        jaccard[i] = len(inter) / len(union) if union else 0.0
        for w in inter:
            dw = max(len(adj[w]), 1)
            aa[i] += 1.0 / np.log(dw + 1.0)
            ra[i] += 1.0 / dw

    out.update(
        {
            "common_neighbors": torch.from_numpy(cn).to(device),
            "jaccard": torch.from_numpy(jaccard).to(device),
            "adamic_adar": torch.from_numpy(aa).to(device),
            "resource_allocation": torch.from_numpy(ra).to(device),
        }
    )
    return out


def node_centric_edge_representation(
    h: Tensor,
    undirected_edges: Tensor,
    structural: Optional[Dict[str, Tensor]] = None,
    x: Optional[Tensor] = None,
) -> Tensor:
    """Build symmetric node-centric edge features z_ij^node."""
    src, dst = undirected_edges
    hi, hj = h[src], h[dst]
    parts = [
        hi + hj,
        (hi - hj).abs(),
        hi * hj,
        _cosine(hi, hj).unsqueeze(-1),
    ]
    if x is not None:
        xi, xj = x[src].float(), x[dst].float()
        parts.append(_cosine(xi, xj).unsqueeze(-1))
    if structural is not None:
        for key in (
            "common_neighbors",
            "jaccard",
            "adamic_adar",
            "resource_allocation",
            "preferential_attachment",
            "degree_sum",
            "deg_u",
            "deg_v",
        ):
            if key in structural:
                parts.append(structural[key].unsqueeze(-1).float())
    return torch.cat(parts, dim=-1)
