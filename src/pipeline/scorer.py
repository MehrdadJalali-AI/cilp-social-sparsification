"""Surrogate + CILP scorer fitting with component timers.

The training loop, losses, epochs, optimiser and architecture are those of
``scripts/run_full_grid.py::score_cailp``. ``CachedViewCAILP`` only memoises the parts of
``CAILPSocial.build_edge_views`` that do not depend on learnable parameters (structural edge
features, the initial edge-feature matrix q, and the line graph), which the reference implementation
recomputed on every epoch. Outputs are numerically identical (tests/test_pipeline.py).
"""
from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.data import Data
from torch_geometric.utils import degree

from src.counterfactual.surrogate import SurrogateEdgeImportance, evaluate_surrogate, train_surrogate
from src.models.cailp import CAILPConfig, CAILPSocial
from src.models.edge_encoder import _cosine, node_centric_edge_representation, structural_edge_features
from src.models.importance_decoder import heteroscedastic_nll, ranking_hinge_loss
from src.models.line_graph_encoder import build_line_graph, initial_edge_features, should_use_line_graph
from src.pipeline.common import timed
from src.utils.graph import undirected_edge_list


def _sparse_edge_features(x: Tensor, und: Tensor, struct_mat: Tensor, chunk: int = 20_000) -> Tensor:
    """initial_edge_features() built chunk-wise and stored as a sparse COO tensor (same values)."""
    parts = []
    for a in range(0, und.size(1), chunk):
        sl = slice(a, a + chunk)
        parts.append(initial_edge_features(x, und[:, sl], structural_mat=struct_mat[sl]).to_sparse())
    return torch.cat(parts, dim=0).coalesce()


class CachedViewCAILP(CAILPSocial):
    _view_cache: Optional[dict] = None
    sparse_q: bool = True  # local-edge-encoder path only (graphs with >= 50k edges)

    def build_edge_views(self, data, h, undirected_edges=None, structural_cache=None, lightweight_struct=True):
        if self._view_cache is None:
            und = undirected_edge_list(data.edge_index) if undirected_edges is None else undirected_edges
            structural = structural_edge_features(data.edge_index, data.num_nodes, und, lightweight=lightweight_struct)
            keys = ["common_neighbors", "jaccard", "adamic_adar", "resource_allocation",
                    "preferential_attachment", "degree_sum", "deg_u", "deg_v"]
            struct_mat = torch.stack([structural[k] for k in keys], dim=-1).float().to(h.device)
            deg = degree(data.edge_index[0], num_nodes=data.num_nodes)
            use_line, info = should_use_line_graph(data.num_nodes, deg)
            line_ei = None
            if use_line and und.size(1) < 50_000:
                q = initial_edge_features(data.x, und, structural_mat=struct_mat)
                line_ei, _ = build_line_graph(und, data.num_nodes)
                line_ei = line_ei.to(q.device)
            else:
                info["fallback"] = "local_edge_encoder"
                info["use_line_graph"] = False
                q = _sparse_edge_features(data.x, und, struct_mat) if self.sparse_q else \
                    initial_edge_features(data.x, und, structural_mat=struct_mat)
            # parameter-independent tail of node_centric_edge_representation: cos(x_i, x_j) and structural columns
            xi, xj = data.x[und[0]].float(), data.x[und[1]].float()
            xcos = _cosine(xi, xj).unsqueeze(-1)
            del xi, xj
            tail = torch.cat([xcos] + [structural[k].unsqueeze(-1).float() for k in keys], dim=-1)
            self._view_cache = {"und": und, "structural": structural, "q": q, "line_ei": line_ei, "info": info,
                                "node_tail": tail}
        c = self._view_cache
        src, dst = c["und"]
        hi, hj = h[src], h[dst]
        z_raw = torch.cat([hi + hj, (hi - hj).abs(), hi * hj, _cosine(hi, hj).unsqueeze(-1), c["node_tail"]], dim=-1)
        z_node = self.node_edge_proj(z_raw)
        self._ensure_built(z_node, c["q"])
        if c["line_ei"] is not None:
            z_edge = self.line_edge_encoder(c["q"], c["line_ei"])
        elif c["q"].layout != torch.strided:
            mlp = self.local_edge_encoder.mlp  # Linear -> ReLU -> Dropout -> Linear
            first = torch.sparse.mm(c["q"], mlp[0].weight.t()) + mlp[0].bias
            z_edge = mlp[3](mlp[2](mlp[1](first)))
        else:
            z_edge = self.local_edge_encoder(c["q"])
        return z_node, z_edge, c["und"], c["info"]


def teacher_features(data: Data, encoder: torch.nn.Module, und: Tensor) -> Tensor:
    with torch.no_grad():
        encoder.eval()
        h_t = encoder(data.x, data.edge_index)
    struct = structural_edge_features(data.edge_index, data.num_nodes, und, lightweight=True)
    return node_centric_edge_representation(h_t, und, struct, x=data.x)


def fit_surrogate(z_all: Tensor, idx: Tensor, y_np: np.ndarray, epochs: int = 60,
                  eval_mode: bool = True) -> Tuple[SurrogateEdgeImportance, Tensor, Dict[str, float]]:
    """eval_mode=True predicts with dropout disabled (used for all reported results). eval_mode=False
    reproduces the Stage A behaviour, which predicted y_full with the surrogate still in train mode
    (dropout p=0.2 active), making the scorer's regression targets stochastic."""
    y_cf = torch.tensor(y_np, dtype=torch.float32)
    sur = SurrogateEdgeImportance(z_all.size(1), 64)
    train_surrogate(sur, z_all[idx], y_cf, epochs=epochs)
    if eval_mode:
        sur.eval()
    with torch.no_grad():
        y_hat = sur(z_all[idx]).numpy()
        y_full = sur(z_all).detach()
    m = evaluate_surrogate(y_hat, y_np, k=min(50, len(y_np)))
    return sur, y_full, {f"insample_{k}": v for k, v in m.items()}


def fit_scorer(data: Data, y_full: Tensor, train_epochs: int, timings: Dict[str, float]) -> Tensor:
    n_cls = int(data.y.max().item()) + 1
    model = CachedViewCAILP(data.x.size(1), n_cls,
                            CAILPConfig(encoder_type="gcn", fusion="concat", hidden_dim=64, num_layers=2, dropout=0.4))
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    with timed(timings, "scorer_view_build"):
        _ = model(data)
    with timed(timings, "scorer_fit", memory=True):
        for _ in range(train_epochs):
            model.train()
            opt.zero_grad()
            out = model(data)
            loss = (0.5 * F.cross_entropy(out["logits"][data.train_mask], data.y[data.train_mask])
                    + 1.0 * heteroscedastic_nll(out["mu"], out["logvar"], y_full)
                    + 0.2 * ranking_hinge_loss(out["mu"], y_full))
            loss.backward()
            opt.step()
    with timed(timings, "scorer_rank_inference"):
        model.eval()
        with torch.no_grad():
            mu = model(data)["mu"].detach()
    return mu
