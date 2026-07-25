from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set

import networkx as nx
import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.data import Data
from torch_geometric.utils import to_networkx

from src.counterfactual.sampling import normalize_scores
from src.utils.graph import undirected_edge_list


@dataclass
class CFCoefficients:
    task: float = 1.0
    comm: float = 0.5
    conn: float = 1.0
    spec: float = 0.3
    repr: float = 0.5
    group: float = 0.5


@dataclass
class CFEffects:
    task: List[float] = field(default_factory=list)
    comm: List[float] = field(default_factory=list)
    conn: List[float] = field(default_factory=list)
    spec: List[float] = field(default_factory=list)
    repr: List[float] = field(default_factory=list)
    group: List[float] = field(default_factory=list)

    def as_dict(self) -> Dict[str, List[float]]:
        return {
            "task": self.task,
            "comm": self.comm,
            "conn": self.conn,
            "spec": self.spec,
            "repr": self.repr,
            "group": self.group,
        }


def _to_nx(data: Data) -> nx.Graph:
    return to_networkx(data, to_undirected=True)


def _remove_edge_nx(G: nx.Graph, u: int, v: int) -> nx.Graph:
    H = G.copy()
    if H.has_edge(u, v):
        H.remove_edge(u, v)
    return H


def connectivity_effect(
    G: nx.Graph,
    u: int,
    v: int,
    bridges: Optional[set] = None,
    n0: Optional[int] = None,
    giant0: Optional[int] = None,
) -> float:
    """Higher = more harmful removal (bridges, component splits)."""
    if not G.has_edge(u, v):
        return 0.0
    if bridges is None:
        bridges = set(frozenset(e) for e in nx.bridges(G))
    bridge = 1.0 if frozenset((u, v)) in bridges else 0.0
    if n0 is None:
        n0 = nx.number_connected_components(G)
    if giant0 is None:
        giant0 = max(len(c) for c in nx.connected_components(G))
    H = _remove_edge_nx(G, u, v)
    n1 = nx.number_connected_components(H)
    giant1 = max(len(c) for c in nx.connected_components(H))
    return float(bridge + (n1 - n0) + max(0, giant0 - giant1) / max(G.number_of_nodes(), 1))


def community_effect(
    G: nx.Graph,
    u: int,
    v: int,
    partition: Optional[dict] = None,
    q0: Optional[float] = None,
) -> float:
    """Local community harm without re-running Louvain per edge when possible."""
    try:
        import community as community_louvain  # python-louvain

        if partition is None:
            partition = community_louvain.best_partition(G)
        if q0 is None:
            q0 = community_louvain.modularity(partition, G)
        same0 = int(partition.get(u) == partition.get(v))
        # Local modularity contribution proxy (Newman): avoid full re-cluster
        # Harm higher for inter-community edges and high-degree endpoints
        du, dv = G.degree(u), G.degree(v)
        m = max(G.number_of_edges(), 1)
        a_uv = 1.0
        expected = (du * dv) / (2.0 * m)
        local = (a_uv - expected) if same0 else abs(a_uv - expected)
        # Extra harm if endpoints in different communities (inter-community connector)
        inter = 0.0 if same0 else 1.0
        return float(max(0.0, local) + inter)
    except Exception:
        cu0 = nx.clustering(G, u)
        cv0 = nx.clustering(G, v)
        H = _remove_edge_nx(G, u, v)
        cu1 = nx.clustering(H, u) if H.degree(u) else 0.0
        cv1 = nx.clustering(H, v) if H.degree(v) else 0.0
        return float(abs(cu0 - cu1) + abs(cv0 - cv1))


def spectral_effect(G: nx.Graph, u: int, v: int, k: int = 3) -> float:
    try:
        n = G.number_of_nodes()
        if n > 400:
            # Algebraic connectivity proxy via endpoint degrees / resistance-style term
            du, dv = max(G.degree(u), 1), max(G.degree(v), 1)
            return float(1.0 / du + 1.0 / dv)
        L0 = nx.normalized_laplacian_matrix(G).astype(float)
        H = _remove_edge_nx(G, u, v)
        L1 = nx.normalized_laplacian_matrix(H).astype(float)
        ev0 = np.sort(np.linalg.eigvalsh(L0.toarray()))[1 : k + 1]
        ev1 = np.sort(np.linalg.eigvalsh(L1.toarray()))[1 : k + 1]
        return float(np.linalg.norm(ev0 - ev1))
    except Exception:
        return 0.0


@torch.no_grad()
def task_and_repr_and_group_effects(
    data: Data,
    encoder: torch.nn.Module,
    classifier: torch.nn.Module,
    u: int,
    v: int,
    device: torch.device,
) -> tuple[float, float, float]:
    """Approximate Δ_task, Δ_repr, Δ_group using a frozen downstream head on train/val only."""
    encoder.eval()
    classifier.eval()
    x = data.x.to(device)
    y = data.y.to(device)
    ei = data.edge_index.to(device)

    def forward(edge_index: Tensor):
        h = encoder(x, edge_index)
        logits = classifier(h)
        return h, logits

    # Masks: never use test
    train = data.train_mask
    val = data.val_mask
    tv = train | val

    h0, logits0 = forward(ei)
    loss0 = F.cross_entropy(logits0[tv], y[tv])

    # Remove both directions of undirected edge
    src, dst = ei
    mask = ~(((src == u) & (dst == v)) | ((src == v) & (dst == u)))
    ei1 = ei[:, mask]
    h1, logits1 = forward(ei1)
    loss1 = F.cross_entropy(logits1[tv], y[tv])
    d_task = float((loss1 - loss0).item())

    d_repr = float((h0 - h1).norm(dim=-1).mean().item())

    # Group: worst-class F1 deterioration on val only
    def worst_f1(logits: Tensor) -> float:
        pred = logits[val].argmax(-1).cpu()
        yt = y[val].cpu()
        f1s = []
        for c in yt.unique():
            tp = ((pred == c) & (yt == c)).sum().item()
            fp = ((pred == c) & (yt != c)).sum().item()
            fn = ((pred != c) & (yt == c)).sum().item()
            prec = tp / (tp + fp + 1e-8)
            rec = tp / (tp + fn + 1e-8)
            f1s.append(2 * prec * rec / (prec + rec + 1e-8))
        return float(min(f1s)) if f1s else 0.0

    w0 = worst_f1(logits0)
    w1 = worst_f1(logits1)
    d_group = float(max(0.0, w0 - w1))
    return d_task, d_repr, d_group


class ExactCounterfactualTeacher:
    """Computes six-component counterfactual targets for a subset of edges."""

    def __init__(
        self,
        coefficients: Optional[CFCoefficients] = None,
        single_objective: bool = False,
    ) -> None:
        self.beta = coefficients or CFCoefficients()
        self.single_objective = single_objective

    def score_edges(
        self,
        data: Data,
        edge_indices: Tensor,
        undirected_edges: Tensor,
        encoder: torch.nn.Module,
        classifier: torch.nn.Module,
        device: torch.device,
    ) -> tuple[np.ndarray, CFEffects]:
        G = _to_nx(data)
        bridges = set(frozenset(e) for e in nx.bridges(G))
        n0 = nx.number_connected_components(G)
        giant0 = max(len(c) for c in nx.connected_components(G))
        partition = None
        q0 = None
        try:
            import community as community_louvain

            partition = community_louvain.best_partition(G)
            q0 = community_louvain.modularity(partition, G)
        except Exception:
            pass
        effects = CFEffects()
        srcs = undirected_edges[0].tolist()
        dsts = undirected_edges[1].tolist()
        for idx in edge_indices.tolist():
            u, v = int(srcs[idx]), int(dsts[idx])
            d_task, d_repr, d_group = task_and_repr_and_group_effects(
                data, encoder, classifier, u, v, device
            )
            d_comm = community_effect(G, u, v, partition=partition, q0=q0)
            d_conn = connectivity_effect(G, u, v, bridges=bridges, n0=n0, giant0=giant0)
            d_spec = spectral_effect(G, u, v)
            effects.task.append(d_task)
            effects.comm.append(d_comm)
            effects.conn.append(d_conn)
            effects.spec.append(d_spec)
            effects.repr.append(d_repr)
            effects.group.append(d_group)

        if self.single_objective:
            # CF-GNNExplainer-style fidelity/prediction-change only (RQ11 / A31)
            y = normalize_scores(np.array(effects.task))
        else:
            y = normalize_scores(
                self.beta.task * normalize_scores(np.array(effects.task))
                + self.beta.comm * normalize_scores(np.array(effects.comm))
                + self.beta.conn * normalize_scores(np.array(effects.conn))
                + self.beta.spec * normalize_scores(np.array(effects.spec))
                + self.beta.repr * normalize_scores(np.array(effects.repr))
                + self.beta.group * normalize_scores(np.array(effects.group))
            )
        return y, effects
