from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

import networkx as nx
import numpy as np
import torch
from torch import Tensor
from torch_geometric.data import Data
from torch_geometric.utils import to_networkx

from src.utils.graph import undirected_edge_list


@dataclass
class PruningConstraints:
    preserve_bridges: bool = True
    preserve_giant_component: bool = True
    min_degree: int = 1
    preserve_msf: bool = True
    protect_minority_edges: bool = True
    minority_frac: float = 0.2


@dataclass
class PruningResult:
    keep_undirected: Tensor
    removed_undirected: Tensor
    achieved_removal_rate: float
    requested_removal_rate: float
    budget_shortfall: bool
    notes: List[str] = field(default_factory=list)


def _minority_classes(y: Tensor, frac: float = 0.2) -> Set[int]:
    vals, counts = torch.unique(y, return_counts=True)
    order = torch.argsort(counts)
    n_min = max(1, int(np.ceil(frac * len(vals))))
    return set(vals[order[:n_min]].tolist())


def constrained_prune(
    data: Data,
    undirected_edges: Tensor,
    importance: Tensor,
    removal_rate: float,
    constraints: Optional[PruningConstraints] = None,
    use_train_val_labels_only: bool = True,
) -> PruningResult:
    """Rank edges ascending by importance; remove if constraints allow until budget."""
    constraints = constraints or PruningConstraints()
    m = undirected_edges.size(1)
    target_remove = int(round(removal_rate * m))
    order = torch.argsort(importance)  # least important first

    G = to_networkx(data, to_undirected=True)
    # Maximum spanning forest on importance as weights for backbone protection
    msf_edges: Set[frozenset] = set()
    if constraints.preserve_msf:
        # Kruskal on descending importance
        H = nx.Graph()
        H.add_nodes_from(range(data.num_nodes))
        pairs = []
        srcs, dsts = undirected_edges[0].tolist(), undirected_edges[1].tolist()
        for i in range(m):
            pairs.append((float(importance[i]), srcs[i], dsts[i]))
        pairs.sort(reverse=True)
        uf = {i: i for i in range(data.num_nodes)}

        def find(a: int) -> int:
            while uf[a] != a:
                uf[a] = uf[uf[a]]
                a = uf[a]
            return a

        for w, u, v in pairs:
            ru, rv = find(u), find(v)
            if ru != rv:
                uf[ru] = rv
                msf_edges.add(frozenset((u, v)))

    bridges = set(frozenset(e) for e in nx.bridges(G)) if constraints.preserve_bridges else set()
    giant0 = max(nx.connected_components(G), key=len)

    minority: Set[int] = set()
    if constraints.protect_minority_edges and data.y is not None:
        # Use train+val labels only
        if use_train_val_labels_only and hasattr(data, "train_mask"):
            mask = data.train_mask | data.val_mask
            y_obs = data.y.clone()
            # Only compute class frequencies on train/val nodes
            minority = _minority_classes(data.y[mask], constraints.minority_frac)
        else:
            minority = _minority_classes(data.y, constraints.minority_frac)

    kept = set(range(m))
    removed: List[int] = []
    notes: List[str] = []
    srcs = undirected_edges[0].tolist()
    dsts = undirected_edges[1].tolist()

    working = G.copy()
    for idx in order.tolist():
        if len(removed) >= target_remove:
            break
        u, v = srcs[idx], dsts[idx]
        key = frozenset((u, v))
        if constraints.preserve_bridges and key in bridges:
            continue
        if constraints.preserve_msf and key in msf_edges:
            continue
        if constraints.min_degree > 0:
            if working.degree(u) <= constraints.min_degree or working.degree(v) <= constraints.min_degree:
                continue
        if constraints.protect_minority_edges and data.y is not None:
            yu, yv = int(data.y[u]), int(data.y[v])
            if yu in minority or yv in minority:
                # Allow removal only if both endpoints remain degree > min after
                if working.degree(u) <= constraints.min_degree + 1 or working.degree(v) <= constraints.min_degree + 1:
                    continue

        # Tentative removal
        working.remove_edge(u, v)
        if constraints.preserve_giant_component:
            giant1 = max(nx.connected_components(working), key=len)
            if len(giant1) < len(giant0):
                working.add_edge(u, v)
                continue
        kept.discard(idx)
        removed.append(idx)

    shortfall = len(removed) < target_remove
    if shortfall:
        notes.append(
            f"Constraints prevented full budget: removed {len(removed)}/{target_remove}"
        )

    keep_idx = sorted(kept)
    rem_idx = sorted(removed)
    return PruningResult(
        keep_undirected=undirected_edges[:, keep_idx] if keep_idx else undirected_edges[:, []] ,
        removed_undirected=undirected_edges[:, rem_idx] if rem_idx else undirected_edges[:, []],
        achieved_removal_rate=len(removed) / m if m else 0.0,
        requested_removal_rate=removal_rate,
        budget_shortfall=shortfall,
        notes=notes,
    )


def budget_prune_unconstrained(
    undirected_edges: Tensor,
    importance: Tensor,
    removal_rate: float,
) -> Tensor:
    """Keep top-(1-r) edges by importance (exactly equal budget)."""
    m = undirected_edges.size(1)
    n_keep = max(0, m - int(round(removal_rate * m)))
    order = torch.argsort(importance, descending=True)
    keep = order[:n_keep]
    return undirected_edges[:, keep.sort().values]
