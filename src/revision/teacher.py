"""Per-edge counterfactual criteria Δ_k(e) with caching (R1 revision).

Numerically this reproduces ``src/counterfactual/exact_teacher.py`` with three documented
differences:

1. Label access (REVISED protocol): ``task_rev`` uses cross-entropy on *training* nodes only and
   ``group_rev`` uses worst-class F1 on *training* nodes only. The ORIGINAL quantities
   (``task_orig`` on train∪validation, ``group_orig`` on validation) are cached alongside for
   provenance and are never used to build REVISED targets.
2. The Louvain partition behind the community-boundary proxy is seeded (``random_state=seed``);
   the ORIGINAL call was unseeded and therefore not reproducible.
3. Efficiency only: the unperturbed forward pass is computed once instead of once per edge, and
   Δ_conn is evaluated with the original function only for bridges (for a non-bridge edge every
   term of Δ_conn is exactly zero). ``tests/test_revision.py`` checks equivalence.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Dict, List, Optional

import networkx as nx
import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.utils import to_networkx

from src.counterfactual.exact_teacher import community_effect, connectivity_effect, spectral_effect
from src.counterfactual.sampling import normalize_scores

RAW_KEYS = ["task_rev", "task_orig", "repr", "group_rev", "group_orig", "comm", "conn", "spec"]


@dataclass
class GraphContext:
    G: nx.Graph
    bridges: set
    n0: int
    giant0: int
    partition: dict
    q0: float
    seconds: Dict[str, float]


def graph_context(data: Data, louvain_seed: int) -> GraphContext:
    import community as community_louvain

    sec: Dict[str, float] = {}
    t = time.perf_counter()
    G = to_networkx(data, to_undirected=True)
    sec["to_networkx"] = time.perf_counter() - t
    t = time.perf_counter()
    bridges = set(frozenset(e) for e in nx.bridges(G))
    n0 = nx.number_connected_components(G)
    giant0 = max(len(c) for c in nx.connected_components(G))
    sec["bridges_components"] = time.perf_counter() - t
    t = time.perf_counter()
    partition = community_louvain.best_partition(G, random_state=louvain_seed)
    q0 = community_louvain.modularity(partition, G)
    sec["louvain"] = time.perf_counter() - t
    return GraphContext(G, bridges, n0, giant0, partition, q0, sec)


def _worst_f1(pred: torch.Tensor, yt: torch.Tensor) -> float:
    # identical to the ORIGINAL nested worst_f1 (classes present in yt; epsilon-smoothed)
    f1s = []
    for c in yt.unique():
        tp = ((pred == c) & (yt == c)).sum().item()
        fp = ((pred == c) & (yt != c)).sum().item()
        fn = ((pred != c) & (yt == c)).sum().item()
        prec = tp / (tp + fp + 1e-8)
        rec = tp / (tp + fn + 1e-8)
        f1s.append(2 * prec * rec / (prec + rec + 1e-8))
    return float(min(f1s)) if f1s else 0.0


@torch.no_grad()
def compute_deltas(
    data: Data,
    encoder: torch.nn.Module,
    classifier: torch.nn.Module,
    und: torch.Tensor,
    edge_ids: List[int],
    ctx: GraphContext,
) -> tuple[Dict[str, np.ndarray], Dict[str, float]]:
    encoder.eval()
    classifier.eval()
    x, y, ei = data.x, data.y, data.edge_index
    tr = data.train_mask
    tv = data.train_mask | data.val_mask
    val = data.val_mask
    h0 = encoder(x, ei)
    lg0 = classifier(h0)
    ce0_tv = F.cross_entropy(lg0[tv], y[tv])
    ce0_tr = F.cross_entropy(lg0[tr], y[tr])
    p0 = lg0.argmax(-1)
    wv0 = _worst_f1(p0[val], y[val])
    wt0 = _worst_f1(p0[tr], y[tr])

    out = {k: np.zeros(len(edge_ids), dtype=np.float64) for k in RAW_KEYS}
    sec = {"probe_forward": 0.0, "conn": 0.0, "comm": 0.0, "spec": 0.0}
    src, dst = ei
    for j, idx in enumerate(edge_ids):
        u, v = int(und[0, idx]), int(und[1, idx])
        t = time.perf_counter()
        mask = ~(((src == u) & (dst == v)) | ((src == v) & (dst == u)))
        h1 = encoder(x, ei[:, mask])
        lg1 = classifier(h1)
        out["task_orig"][j] = float((F.cross_entropy(lg1[tv], y[tv]) - ce0_tv).item())
        out["task_rev"][j] = float((F.cross_entropy(lg1[tr], y[tr]) - ce0_tr).item())
        out["repr"][j] = float((h0 - h1).norm(dim=-1).mean().item())
        p1 = lg1.argmax(-1)
        out["group_orig"][j] = max(0.0, wv0 - _worst_f1(p1[val], y[val]))
        out["group_rev"][j] = max(0.0, wt0 - _worst_f1(p1[tr], y[tr]))
        sec["probe_forward"] += time.perf_counter() - t

        t = time.perf_counter()
        if frozenset((u, v)) in ctx.bridges:
            out["conn"][j] = connectivity_effect(ctx.G, u, v, bridges=ctx.bridges, n0=ctx.n0, giant0=ctx.giant0)
        else:
            out["conn"][j] = 0.0
        sec["conn"] += time.perf_counter() - t
        t = time.perf_counter()
        out["comm"][j] = community_effect(ctx.G, u, v, partition=ctx.partition, q0=ctx.q0)
        sec["comm"] += time.perf_counter() - t
        t = time.perf_counter()
        out["spec"][j] = spectral_effect(ctx.G, u, v)
        sec["spec"] += time.perf_counter() - t
    return out, sec


def composite_target(raw: Dict[str, np.ndarray], beta: Dict[str, float], protocol: str = "REVISED") -> np.ndarray:
    """y = minmax( Σ_k β_k minmax(Δ_k) ) over the given edge set (ORIGINAL aggregation rule)."""
    key = {"task": "task_rev" if protocol == "REVISED" else "task_orig",
           "group": "group_rev" if protocol == "REVISED" else "group_orig"}
    acc = None
    for k, b in beta.items():
        if b == 0.0:
            continue
        arr = normalize_scores(raw[key.get(k, k)])
        acc = b * arr if acc is None else acc + b * arr
    if acc is None:
        return np.zeros(len(next(iter(raw.values()))))
    return normalize_scores(acc)
