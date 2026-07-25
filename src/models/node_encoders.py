from __future__ import annotations

from typing import Literal, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.nn import GATConv, GATv2Conv, GCNConv, SAGEConv


class NodeEncoder(nn.Module):
    """Configurable node encoder supporting GCN, GraphSAGE, and GATv2."""

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int = 64,
        out_dim: int = 64,
        num_layers: int = 2,
        dropout: float = 0.5,
        encoder_type: Literal["gcn", "sage", "gatv2"] = "gatv2",
        heads: int = 4,
        residual: bool = True,
        norm: bool = True,
    ) -> None:
        super().__init__()
        self.encoder_type = encoder_type
        self.dropout = dropout
        self.residual = residual
        self.num_layers = num_layers
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()

        dims = [in_dim] + [hidden_dim] * (num_layers - 1) + [out_dim]
        for i in range(num_layers):
            din, dout = dims[i], dims[i + 1]
            if encoder_type == "gcn":
                self.convs.append(GCNConv(din, dout))
            elif encoder_type == "sage":
                self.convs.append(SAGEConv(din, dout))
            elif encoder_type == "gatv2":
                h = heads if i < num_layers - 1 else 1
                in_heads = 1 if i == 0 or encoder_type != "gatv2" else (heads if i > 0 else 1)
                # For intermediate GAT layers with multi-head concat, adjust dims
                if i < num_layers - 1:
                    self.convs.append(GATv2Conv(din, dout // heads, heads=heads, concat=True))
                    dout_eff = dout
                else:
                    self.convs.append(GATv2Conv(din, dout, heads=1, concat=False))
                    dout_eff = dout
            else:
                raise ValueError(encoder_type)
            if norm:
                self.norms.append(nn.LayerNorm(dout if encoder_type != "gatv2" or i == num_layers - 1 else dout))
            else:
                self.norms.append(nn.Identity())

        self.out_dim = out_dim

    def forward(self, x: Tensor, edge_index: Tensor) -> Tensor:
        h = x
        for i, conv in enumerate(self.convs):
            h_in = h
            h = conv(h, edge_index)
            h = self.norms[i](h)
            if i < self.num_layers - 1:
                h = F.relu(h)
                h = F.dropout(h, p=self.dropout, training=self.training)
                if self.residual and h_in.size(-1) == h.size(-1):
                    h = h + h_in
        return h

    def oversmoothing_diagnostic(self, h: Tensor) -> float:
        """Mean pairwise cosine similarity of node embeddings (higher ⇒ more oversmoothing)."""
        hn = F.normalize(h, dim=-1)
        # Sample for large graphs
        n = hn.size(0)
        if n > 512:
            idx = torch.randperm(n, device=hn.device)[:512]
            hn = hn[idx]
        sim = hn @ hn.t()
        # Exclude diagonal
        n2 = sim.size(0)
        off = (sim.sum() - sim.diag().sum()) / (n2 * (n2 - 1) + 1e-8)
        return float(off.item())
