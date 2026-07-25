from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.nn import GCNConv


def estimate_line_graph_edges(degrees: Tensor) -> int:
    """Estimate number of line-graph edges: sum_v deg(v) choose 2 * 2 orientations approx."""
    d = degrees.double()
    # Each node of degree d contributes C(d,2) undirected line-graph edges
    return int(((d * (d - 1)) / 2).sum().item())


def should_use_line_graph(
    num_nodes: int,
    degrees: Tensor,
    max_line_edges: int = 2_000_000,
    max_memory_gb: float = 8.0,
) -> Tuple[bool, dict]:
    n_line_e = estimate_line_graph_edges(degrees)
    # Rough memory: 2 * n_line_e * 8 bytes for edge_index
    mem_gb = (2 * n_line_e * 8) / (1024**3)
    ok = n_line_e <= max_line_edges and mem_gb <= max_memory_gb
    info = {
        "estimated_line_graph_edges": n_line_e,
        "estimated_memory_gb": mem_gb,
        "use_line_graph": ok,
    }
    return ok, info


def build_line_graph(
    undirected_edges: Tensor,
    num_nodes: int,
) -> Tuple[Tensor, Tensor]:
    """Build line-graph adjacency: edge-nodes connected if they share an endpoint.

    Returns edge_index_line [2, E_L], and mapping.
    Memory-intensive for high-degree graphs — call should_use_line_graph first.
    """
    src, dst = undirected_edges.cpu()
    m = undirected_edges.size(1)
    # Incidence lists
    incident: list[list[int]] = [[] for _ in range(num_nodes)]
    for e_id, (u, v) in enumerate(zip(src.tolist(), dst.tolist())):
        incident[u].append(e_id)
        incident[v].append(e_id)

    rows, cols = [], []
    for node_edges in incident:
        L = len(node_edges)
        for i in range(L):
            for j in range(i + 1, L):
                a, b = node_edges[i], node_edges[j]
                rows.extend([a, b])
                cols.extend([b, a])
    if not rows:
        ei = torch.zeros(2, 0, dtype=torch.long)
    else:
        ei = torch.tensor([rows, cols], dtype=torch.long)
    return ei, undirected_edges


class LineGraphEncoder(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int = 64, out_dim: int = 64, dropout: float = 0.3):
        super().__init__()
        self.conv1 = GCNConv(in_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, out_dim)
        self.dropout = dropout

    def forward(self, q: Tensor, line_edge_index: Tensor) -> Tensor:
        h = F.relu(self.conv1(q, line_edge_index))
        h = F.dropout(h, p=self.dropout, training=self.training)
        return self.conv2(h, line_edge_index)


class LocalEdgeEncoder(nn.Module):
    """Scalable edge encoder: MLP on endpoint features + sampled neighbor-edge stats."""

    def __init__(self, in_dim: int, hidden_dim: int = 64, out_dim: int = 64, dropout: float = 0.3):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, q: Tensor, line_edge_index: Optional[Tensor] = None) -> Tensor:
        return self.mlp(q)


def initial_edge_features(
    x: Tensor,
    undirected_edges: Tensor,
    structural_mat: Optional[Tensor] = None,
) -> Tensor:
    src, dst = undirected_edges
    xi, xj = x[src].float(), x[dst].float()
    cos = (xi * xj).sum(-1, keepdim=True) / (xi.norm(dim=-1, keepdim=True) * xj.norm(dim=-1, keepdim=True) + 1e-8)
    parts = [((xi - xj).abs()), xi * xj, cos]
    if structural_mat is not None:
        parts.append(structural_mat)
    return torch.cat(parts, dim=-1)
