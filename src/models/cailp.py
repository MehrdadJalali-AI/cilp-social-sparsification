from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.data import Data
from torch_geometric.utils import degree

from src.models.cross_attention import build_fusion
from src.models.edge_encoder import node_centric_edge_representation, structural_edge_features
from src.models.importance_decoder import ImportanceDecoder, heteroscedastic_nll, ranking_hinge_loss
from src.models.line_graph_encoder import (
    LocalEdgeEncoder,
    LineGraphEncoder,
    build_line_graph,
    initial_edge_features,
    should_use_line_graph,
)
from src.models.node_encoders import NodeEncoder
from src.utils.graph import undirected_edge_list


@dataclass
class CAILPConfig:
    encoder_type: str = "gatv2"
    hidden_dim: int = 64
    num_layers: int = 2
    dropout: float = 0.4
    heads: int = 4
    fusion: str = "cross_attention"
    use_structural: bool = True
    lambda_cf: float = 1.0
    lambda_rank: float = 0.2
    lambda_task: float = 0.5
    ranking_margin: float = 0.1


class CAILPSocial(nn.Module):
    """Core Counterfactual Attention-Based Inverse Link Prediction model."""

    def __init__(self, in_dim: int, n_classes: int, cfg: Optional[CAILPConfig] = None):
        super().__init__()
        self.cfg = cfg or CAILPConfig()
        self.node_encoder = NodeEncoder(
            in_dim=in_dim,
            hidden_dim=self.cfg.hidden_dim,
            out_dim=self.cfg.hidden_dim,
            num_layers=self.cfg.num_layers,
            dropout=self.cfg.dropout,
            encoder_type=self.cfg.encoder_type,  # type: ignore[arg-type]
            heads=self.cfg.heads,
        )
        # Placeholder dims updated on first forward after seeing structural width
        self._struct_dim = 8 if self.cfg.use_structural else 0
        node_edge_dim = self.cfg.hidden_dim * 3 + 1 + 1 + self._struct_dim  # sum,absdiff,hadamard,cos,featcos?,struct
        # feat cos adds 1
        node_edge_dim = self.cfg.hidden_dim * 3 + 2 + self._struct_dim
        # Edge-centric input: |x_i-x_j| + x_i⊙x_j + cos + struct
        self._edge_in_dim = None  # lazy
        self.local_edge_encoder: Optional[nn.Module] = None
        self.line_edge_encoder: Optional[nn.Module] = None
        self.fusion: Optional[nn.Module] = None
        self.decoder: Optional[ImportanceDecoder] = None
        self.classifier = nn.Linear(self.cfg.hidden_dim, n_classes)
        self.node_edge_proj = nn.LazyLinear(self.cfg.hidden_dim)
        self._built = False
        self.n_classes = n_classes

    def _ensure_built(self, z_node: Tensor, q: Tensor) -> None:
        if self._built:
            return
        edge_in = q.size(-1)
        self.local_edge_encoder = LocalEdgeEncoder(edge_in, self.cfg.hidden_dim, self.cfg.hidden_dim)
        self.line_edge_encoder = LineGraphEncoder(edge_in, self.cfg.hidden_dim, self.cfg.hidden_dim)
        # After projecting node-centric to hidden
        self.fusion = build_fusion(
            self.cfg.fusion,
            node_dim=self.cfg.hidden_dim,
            edge_dim=self.cfg.hidden_dim,
            hidden_dim=self.cfg.hidden_dim,
        )
        self.decoder = ImportanceDecoder(self.cfg.hidden_dim, self.cfg.hidden_dim)
        self._built = True

    def encode_nodes(self, data: Data) -> Tensor:
        return self.node_encoder(data.x, data.edge_index)

    def build_edge_views(
        self,
        data: Data,
        h: Tensor,
        undirected_edges: Optional[Tensor] = None,
        structural_cache: Optional[Dict[str, Tensor]] = None,
        lightweight_struct: bool = True,
    ) -> tuple[Tensor, Tensor, Tensor, dict]:
        if undirected_edges is None:
            undirected_edges = undirected_edge_list(data.edge_index)
        structural = structural_cache
        struct_mat = None
        if self.cfg.use_structural:
            if structural is None:
                structural = structural_edge_features(
                    data.edge_index,
                    data.num_nodes,
                    undirected_edges,
                    lightweight=lightweight_struct,
                )
            keys = [
                "common_neighbors",
                "jaccard",
                "adamic_adar",
                "resource_allocation",
                "preferential_attachment",
                "degree_sum",
                "deg_u",
                "deg_v",
            ]
            struct_mat = torch.stack([structural[k] for k in keys], dim=-1).float().to(h.device)
        z_node_raw = node_centric_edge_representation(h, undirected_edges, structural, x=data.x)
        z_node = self.node_edge_proj(z_node_raw)
        q = initial_edge_features(data.x, undirected_edges, structural_mat=struct_mat)
        self._ensure_built(z_node, q)
        assert self.local_edge_encoder is not None and self.line_edge_encoder is not None

        deg = degree(data.edge_index[0], num_nodes=data.num_nodes)
        use_line, info = should_use_line_graph(data.num_nodes, deg)
        if use_line and undirected_edges.size(1) < 50_000:
            line_ei, _ = build_line_graph(undirected_edges, data.num_nodes)
            line_ei = line_ei.to(q.device)
            z_edge = self.line_edge_encoder(q, line_ei)
        else:
            z_edge = self.local_edge_encoder(q)
            info["fallback"] = "local_edge_encoder"
            info["use_line_graph"] = False
        return z_node, z_edge, undirected_edges, info

    def score_edges(
        self,
        data: Data,
        h: Optional[Tensor] = None,
    ) -> tuple[Tensor, Tensor, Tensor, dict]:
        if h is None:
            h = self.encode_nodes(data)
        z_node, z_edge, und, info = self.build_edge_views(data, h)
        assert self.fusion is not None and self.decoder is not None
        z = self.fusion(z_node, z_edge)
        mu, logvar = self.decoder(z)
        return mu, logvar, und, info

    def forward(self, data: Data) -> Dict[str, Tensor]:
        h = self.encode_nodes(data)
        logits = self.classifier(h)
        mu, logvar, und, info = self.score_edges(data, h=h)
        return {
            "h": h,
            "logits": logits,
            "mu": mu,
            "logvar": logvar,
            "undirected_edges": und,
            "info": info,  # type: ignore[dict-item]
        }

    def compute_loss(
        self,
        out: Dict[str, Tensor],
        data: Data,
        y_cf: Optional[Tensor] = None,
        edge_idx_cf: Optional[Tensor] = None,
    ) -> Dict[str, Tensor]:
        cfg = self.cfg
        losses: Dict[str, Tensor] = {}
        train = data.train_mask
        losses["task"] = F.cross_entropy(out["logits"][train], data.y[train])
        total = cfg.lambda_task * losses["task"]
        if y_cf is not None and edge_idx_cf is not None:
            mu_sub = out["mu"][edge_idx_cf]
            logvar_sub = out["logvar"][edge_idx_cf]
            losses["cf"] = heteroscedastic_nll(mu_sub, logvar_sub, y_cf)
            losses["rank"] = ranking_hinge_loss(mu_sub, y_cf, margin=cfg.ranking_margin)
            total = total + cfg.lambda_cf * losses["cf"] + cfg.lambda_rank * losses["rank"]
        losses["total"] = total
        return losses
