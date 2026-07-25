#!/usr/bin/env python3
"""Reproduce original ILP-GCN sparsification under equal edge-retention budgets."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.evaluation.structural import structural_metrics
from src.sparsification.original_ilp import sparsify_with_ilp
from src.tasks.node_classification import train_node_classifier
from src.utils.io import get_device, save_json, set_seed, setup_logging
from src.utils.splits import LeakageGuard


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="lastfm")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--removal-rates", nargs="+", type=float, default=[0.3, 0.5, 0.7])
    parser.add_argument("--epochs", type=int, default=40)
    args = parser.parse_args()
    setup_logging()
    set_seed(args.seed)
    device = get_device()
    data = torch.load(ROOT / "data" / "processed" / f"{args.dataset}.pt", weights_only=False)
    split = torch.load(ROOT / "data" / "splits" / f"{args.dataset}_seed{args.seed}.pt", weights_only=False)
    data.train_mask, data.val_mask, data.test_mask = split["train_mask"], split["val_mask"], split["test_mask"]

    guard = LeakageGuard()
    guard.set_phase("ilp_sparsify")
    guard.lock_test()

    results = []
    for r in args.removal_rates:
        sparse = sparsify_with_ilp(data, removal_rate=r, epochs=args.epochs, device=device)
        sparse.train_mask, sparse.val_mask, sparse.test_mask = data.train_mask, data.val_mask, data.test_mask
        guard.unlock_test_for_eval()
        _, metrics = train_node_classifier(sparse, kind="gcn", epochs=80, device=device)
        struct = structural_metrics(data, sparse)
        row = {"method": "original_ilp_gcn", "removal_rate": r, **metrics, **struct, "leakage": guard.summary()}
        results.append(row)
        print(row)
        guard.lock_test()
        guard.set_phase("ilp_sparsify")

    out = ROOT / "results" / "raw" / f"original_ilp_{args.dataset}_seed{args.seed}.json"
    save_json(results, out)
    print("Wrote", out)


if __name__ == "__main__":
    main()
