from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np
import torch
from torch import Tensor
from torch_geometric.data import Data


class LeakageGuard:
    """Tracks forbidden accesses to test labels/edges during sparsification."""

    def __init__(self) -> None:
        self.test_label_accesses: int = 0
        self.test_edge_accesses: int = 0
        self._locked: bool = False
        self._phase: str = "init"

    def set_phase(self, phase: str) -> None:
        self._phase = phase

    def lock_test(self) -> None:
        self._locked = True

    def unlock_test_for_eval(self) -> None:
        self._locked = False
        self._phase = "final_eval"

    def access_test_labels(self, reason: str = "") -> None:
        self.test_label_accesses += 1
        if self._locked:
            raise RuntimeError(
                f"Forbidden test-label access during phase={self._phase}: {reason}"
            )

    def access_test_edges(self, reason: str = "") -> None:
        self.test_edge_accesses += 1
        if self._locked:
            raise RuntimeError(
                f"Forbidden test-edge access during phase={self._phase}: {reason}"
            )

    def summary(self) -> Dict[str, object]:
        return {
            "phase": self._phase,
            "locked": self._locked,
            "test_label_accesses": self.test_label_accesses,
            "test_edge_accesses": self.test_edge_accesses,
        }


def stratified_node_splits(
    y: Tensor,
    train_ratio: float = 0.6,
    val_ratio: float = 0.2,
    seed: int = 0,
) -> Tuple[Tensor, Tensor, Tensor]:
    """Create stratified train/val/test boolean masks."""
    y_np = y.cpu().numpy()
    n = len(y_np)
    rng = np.random.RandomState(seed)
    train_mask = np.zeros(n, dtype=bool)
    val_mask = np.zeros(n, dtype=bool)
    test_mask = np.zeros(n, dtype=bool)

    for c in np.unique(y_np):
        idx = np.where(y_np == c)[0]
        rng.shuffle(idx)
        n_train = int(round(train_ratio * len(idx)))
        n_val = int(round(val_ratio * len(idx)))
        # Ensure at least one test if possible
        if n_train + n_val >= len(idx) and len(idx) >= 3:
            n_val = max(1, len(idx) // 5)
            n_train = max(1, len(idx) - n_val - 1)
        train_idx = idx[:n_train]
        val_idx = idx[n_train : n_train + n_val]
        test_idx = idx[n_train + n_val :]
        train_mask[train_idx] = True
        val_mask[val_idx] = True
        test_mask[test_idx] = True

    return (
        torch.from_numpy(train_mask),
        torch.from_numpy(val_mask),
        torch.from_numpy(test_mask),
    )


def split_edges_for_link_prediction(
    edge_index: Tensor,
    num_nodes: int,
    val_ratio: float = 0.05,
    test_ratio: float = 0.1,
    seed: int = 0,
) -> Dict[str, Tensor]:
    """Split undirected positive edges; return train graph edges + val/test pos/neg.

    Leakage rule: sparsifier may only see train_edge_index.
    """
    from torch_geometric.utils import negative_sampling, to_undirected

    rng = np.random.RandomState(seed)
    # Unique undirected
    row, col = edge_index
    mask = row < col
    pos = torch.stack([row[mask], col[mask]], dim=0)
    n_pos = pos.size(1)
    perm = rng.permutation(n_pos)
    n_test = int(round(test_ratio * n_pos))
    n_val = int(round(val_ratio * n_pos))
    test_idx = perm[:n_test]
    val_idx = perm[n_test : n_test + n_val]
    train_idx = perm[n_test + n_val :]

    def take(idx: np.ndarray) -> Tensor:
        return pos[:, torch.from_numpy(idx).long()]

    train_pos = take(train_idx)
    val_pos = take(val_idx)
    test_pos = take(test_idx)

    train_edge_index = to_undirected(train_pos, num_nodes=num_nodes)

    # Separate negatives per split
    neg_val = negative_sampling(
        edge_index, num_nodes=num_nodes, num_neg_samples=val_pos.size(1)
    )
    neg_test = negative_sampling(
        edge_index, num_nodes=num_nodes, num_neg_samples=test_pos.size(1)
    )
    neg_train = negative_sampling(
        train_edge_index, num_nodes=num_nodes, num_neg_samples=train_pos.size(1)
    )

    return {
        "train_edge_index": train_edge_index,
        "train_pos": train_pos,
        "train_neg": neg_train,
        "val_pos": val_pos,
        "val_neg": neg_val,
        "test_pos": test_pos,
        "test_neg": neg_test,
    }


def attach_masks(data: Data, train: Tensor, val: Tensor, test: Tensor) -> Data:
    data.train_mask = train
    data.val_mask = val
    data.test_mask = test
    return data
