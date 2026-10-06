"""Downstream evaluation at exact budgets with structural, community, and cost metrics.

Metric definitions (identical in manuscript, supplement, and code)
------------------------------------------------------------------
giant-component ratio      |largest connected component of G_r| / |V|  (all nodes retained)
bridge retention           |B(G) ∩ E_r| / |B(G)|, B(G) = bridges of the original graph
                            (an original bridge kept in E_r is necessarily still a bridge)
minority-degree retention  mean degree in G_r of nodes in the minority class / same in G,
                            minority class = class with the fewest nodes in the full label vector
                            (labels are used for this evaluation metric only)
conductance (reference P)  size-weighted mean over communities C∈P with vol>0 of
                            cut(C)/min(vol(C), vol(V)-vol(C)); P = Leiden partition of G
NMI / ARI                  between Leiden(G) and Leiden(G_r) (same algorithm, same seed)
modularity retention       Q(P, G_r) / Q(P, G)
messages / layer / epoch   2|E_r| + |V|  (GCNConv message passing incl. self-loops)
"""
from __future__ import annotations

import time
from typing import Dict, List, Optional

import numpy as np
import scipy.sparse as sp
import torch
from scipy.sparse.csgraph import connected_components
from scipy.stats import ks_2samp
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from torch_geometric.data import Data

from src.revision.common import PeakRSS, gcn_messages, set_seed_all
from src.sparsification.constrained_pruning import budget_prune_unconstrained
from src.tasks.node_classification import evaluate_node_classification, train_node_classifier
from src.utils.graph import subgraph_from_undirected_edges


class GraphReference:
    """Per-dataset quantities of the original graph, computed once."""

    def __init__(self, data: Data, und: torch.Tensor, leiden_seed: int = 0):
        import igraph as ig
        import leidenalg
        import networkx as nx
        from torch_geometric.utils import to_networkx

        self.n = data.num_nodes
        self.m = und.size(1)
        self.und = und
        src, dst = und.numpy()
        self.deg0 = np.bincount(np.concatenate([src, dst]), minlength=self.n).astype(float)
        G = to_networkx(data, to_undirected=True)
        bset = set(frozenset(e) for e in nx.bridges(G))
        self.bridge_mask = np.array([frozenset((int(a), int(b))) in bset for a, b in zip(src, dst)])
        self.n_bridges = int(self.bridge_mask.sum())
        y = data.y.numpy()
        classes, counts = np.unique(y, return_counts=True)
        self.minority_class = int(classes[np.argmin(counts)])
        self.minority_mask = y == self.minority_class
        self.leiden_seed = leiden_seed
        self.g0 = ig.Graph(n=self.n, edges=list(zip(src.tolist(), dst.tolist())))
        t = time.perf_counter()
        self.ref_part = np.array(
            leidenalg.find_partition(self.g0, leidenalg.ModularityVertexPartition, seed=leiden_seed).membership)
        self.leiden_seconds = time.perf_counter() - t
        self.q0 = self.g0.modularity(self.ref_part.tolist())
        self.phi0 = self.conductance(src, dst)
        self.n_communities = int(self.ref_part.max() + 1)

    def conductance(self, src: np.ndarray, dst: np.ndarray) -> float:
        P = self.ref_part
        k = P.max() + 1
        deg = np.bincount(np.concatenate([src, dst]), minlength=self.n)
        vol = np.bincount(P, weights=deg, minlength=k)
        cross = P[src] != P[dst]
        cut = np.bincount(P[src[cross]], minlength=k) + np.bincount(P[dst[cross]], minlength=k)
        total = vol.sum()
        denom = np.minimum(vol, total - vol)
        ok = denom > 0
        sizes = np.bincount(P, minlength=k)
        return float(np.sum(sizes[ok] * cut[ok] / denom[ok]) / max(sizes[ok].sum(), 1))


def structural_and_community(ref: GraphReference, keep: torch.Tensor, community: bool = True) -> Dict[str, float]:
    import igraph as ig
    import leidenalg

    src, dst = keep.numpy()
    n = ref.n
    A = sp.coo_matrix((np.ones(len(src)), (src, dst)), shape=(n, n))
    ncomp, lab = connected_components(A, directed=False)
    giant = np.bincount(lab).max() / n
    deg1 = np.bincount(np.concatenate([src, dst]), minlength=n).astype(float)
    # retained bridges: map kept edges back to original ids via a hash of (u,v)
    key0 = ref.und[0].numpy().astype(np.int64) * n + ref.und[1].numpy()
    key1 = src.astype(np.int64) * n + dst
    kept_mask = np.isin(key0, key1, assume_unique=True)
    out = {
        "retained_edge_ratio": len(src) / ref.m,
        "giant_component_ratio": float(giant),
        "num_components": int(ncomp),
        "bridge_retention": float((kept_mask & ref.bridge_mask).sum() / max(ref.n_bridges, 1)),
        "minority_degree_retention": float(deg1[ref.minority_mask].mean() / (ref.deg0[ref.minority_mask].mean() + 1e-8)),
        "minority_isolation_rate": float((deg1[ref.minority_mask] == 0).mean()),
        "degree_ks": float(ks_2samp(ref.deg0, deg1).statistic),
        "messages_per_layer": gcn_messages(len(src), n),
    }
    if community:
        g1 = ig.Graph(n=n, edges=list(zip(src.tolist(), dst.tolist())))
        t = time.perf_counter()
        part1 = np.array(leidenalg.find_partition(g1, leidenalg.ModularityVertexPartition, seed=ref.leiden_seed).membership)
        out["leiden_seconds"] = time.perf_counter() - t
        phi = ref.conductance(src, dst)
        q_ref = g1.modularity(ref.ref_part.tolist()) if len(src) else 0.0
        out.update({
            "conductance_ref": phi,
            "conductance_ratio": phi / ref.phi0 if ref.phi0 > 0 else float("nan"),
            "nmi_leiden": float(normalized_mutual_info_score(ref.ref_part, part1)),
            "ari_leiden": float(adjusted_rand_score(ref.ref_part, part1)),
            "modularity_ref_partition": float(q_ref),
            "modularity_retention": float(q_ref / ref.q0) if ref.q0 else float("nan"),
            "modularity_redetected": float(g1.modularity(part1.tolist())) if len(src) else 0.0,
        })
    return out


def downstream(sparse: Data, epochs: int, init_seed: int, with_test: bool) -> Dict[str, float]:
    """Train the fixed 2-layer GCN (validation Macro-F1 selects the epoch) and time it."""
    set_seed_all(init_seed)
    rec: Dict[str, float] = {}
    with PeakRSS() as mem:
        t = time.perf_counter()
        model, metrics = train_node_classifier(sparse, kind="gcn", epochs=epochs, device=torch.device("cpu"))
        rec["downstream_train_seconds"] = time.perf_counter() - t
    rec["downstream_peak_rss_mb"] = mem.peak_mb
    rec["downstream_delta_rss_mb"] = mem.delta_mb
    model.eval()
    times = []
    with torch.no_grad():
        for _ in range(5):
            t = time.perf_counter()
            model(sparse.x, sparse.edge_index)
            times.append(time.perf_counter() - t)
    rec["downstream_inference_seconds"] = float(np.median(times))
    rec["val_macro_f1"] = metrics["val_macro_f1"]
    rec["val_accuracy"] = metrics["val_accuracy"]
    if with_test:
        for k in ("test_macro_f1", "test_accuracy", "test_worst_class_f1"):
            rec[k] = metrics[k]
    return rec


def evaluate_scores(
    data: Data,
    ref: GraphReference,
    importance: Optional[torch.Tensor],
    budgets: List[float],
    down_epochs: int,
    seed: int,
    with_test: bool,
    community: bool = True,
) -> List[Dict[str, float]]:
    """importance=None evaluates the full graph (r=0 reference)."""
    rows = []
    for r in budgets:
        t = time.perf_counter()
        keep = ref.und if importance is None else budget_prune_unconstrained(ref.und, importance, r)
        sparse = subgraph_from_undirected_edges(data, keep)
        prune_s = time.perf_counter() - t
        # common random numbers: identical GCN init for every method at a given (seed, budget)
        rec = downstream(sparse, down_epochs, init_seed=seed * 1000 + int(round(r * 100)), with_test=with_test)
        rec.update(structural_and_community(ref, keep, community=community))
        rec.update({"removal_rate": r, "prune_seconds": prune_s})
        rows.append(rec)
    return rows
