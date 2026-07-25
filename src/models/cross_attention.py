from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class CrossAttentionFusion(nn.Module):
    def __init__(self, node_dim: int, edge_dim: int, hidden_dim: int = 64, num_heads: int = 4):
        super().__init__()
        self.q = nn.Linear(node_dim, hidden_dim)
        self.k = nn.Linear(edge_dim, hidden_dim)
        self.v = nn.Linear(edge_dim, hidden_dim)
        self.attn = nn.MultiheadAttention(hidden_dim, num_heads=num_heads, batch_first=True)
        self.proj_node = nn.Linear(node_dim, hidden_dim)
        self.proj_u = nn.Linear(hidden_dim, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)
        self.out_dim = hidden_dim

    def forward(self, z_node: Tensor, z_edge: Tensor) -> Tensor:
        # Treat each edge as a sequence of length 1 for MHA API reuse
        q = self.q(z_node).unsqueeze(1)
        k = self.k(z_edge).unsqueeze(1)
        v = self.v(z_edge).unsqueeze(1)
        u, _ = self.attn(q, k, v, need_weights=False)
        u = u.squeeze(1)
        return self.norm(self.proj_node(z_node) + self.proj_u(u))


class ConcatFusion(nn.Module):
    def __init__(self, node_dim: int, edge_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(node_dim + edge_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
        )
        self.out_dim = hidden_dim

    def forward(self, z_node: Tensor, z_edge: Tensor) -> Tensor:
        return self.fc(torch.cat([z_node, z_edge], dim=-1))


class GatedFusion(nn.Module):
    def __init__(self, node_dim: int, edge_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.pn = nn.Linear(node_dim, hidden_dim)
        self.pe = nn.Linear(edge_dim, hidden_dim)
        self.gate = nn.Linear(hidden_dim * 2, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)
        self.out_dim = hidden_dim

    def forward(self, z_node: Tensor, z_edge: Tensor) -> Tensor:
        n, e = self.pn(z_node), self.pe(z_edge)
        g = torch.sigmoid(self.gate(torch.cat([n, e], dim=-1)))
        return self.norm(g * n + (1 - g) * e)


class WeightedSumFusion(nn.Module):
    def __init__(self, node_dim: int, edge_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.pn = nn.Linear(node_dim, hidden_dim)
        self.pe = nn.Linear(edge_dim, hidden_dim)
        self.alpha = nn.Parameter(torch.tensor(0.5))
        self.norm = nn.LayerNorm(hidden_dim)
        self.out_dim = hidden_dim

    def forward(self, z_node: Tensor, z_edge: Tensor) -> Tensor:
        a = torch.sigmoid(self.alpha)
        return self.norm(a * self.pn(z_node) + (1 - a) * self.pe(z_edge))


class BilinearFusion(nn.Module):
    def __init__(self, node_dim: int, edge_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.bilinear = nn.Bilinear(node_dim, edge_dim, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)
        self.out_dim = hidden_dim

    def forward(self, z_node: Tensor, z_edge: Tensor) -> Tensor:
        return self.norm(self.bilinear(z_node, z_edge))


def build_fusion(name: str, node_dim: int, edge_dim: int, hidden_dim: int = 64) -> nn.Module:
    name = name.lower()
    if name in ("cross_attention", "cross-attn", "crossattn"):
        return CrossAttentionFusion(node_dim, edge_dim, hidden_dim)
    if name in ("concat", "concatenation"):
        return ConcatFusion(node_dim, edge_dim, hidden_dim)
    if name in ("gated", "gate"):
        return GatedFusion(node_dim, edge_dim, hidden_dim)
    if name in ("weighted", "weighted_sum"):
        return WeightedSumFusion(node_dim, edge_dim, hidden_dim)
    if name in ("bilinear",):
        return BilinearFusion(node_dim, edge_dim, hidden_dim)
    raise ValueError(f"Unknown fusion: {name}")
