#!/usr/bin/env python3
"""Run classical and learned edge-sparsification baselines under equal budgets."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.evaluation.structural import structural_metrics
from src.sparsification.baselines.classical import CLASSICAL_METHODS
from src.sparsification.baselines.dspar import dspar_sparsify
from src.sparsification.baselines.neuralsparse import neuralsparse_sparsify
from src.sparsification.baselines.ptdnet import ptdnet_sparsify
from src.sparsification.original_ilp import sparsify_with_ilp
from src.tasks.node_classification import train_node_classifier
from src.utils.io import get_device, save_json, set_seed, setup_logging
from src.utils.splits import LeakageGuard


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", default=["lastfm"])
    parser.add_argument("--budgets", nargs="+", type=float, default=[0.3, 0.5, 0.7])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--methods",
        nargs="+",
        default=[
            "random",
            "jaccard",
            "degree_sum",
            "dropedge",
            "effective_resistance",
            "dspar",
            "neuralsparse",
            "ptdnet",
            "original_ilp",
        ],
    )
    args = parser.parse_args()
    setup_logging()
    set_seed(args.seed)
    device = get_device()

    results = []
    for ds in args.datasets:
        data = torch.load(ROOT / "data" / "processed" / f"{ds}.pt", weights_only=False)
        split = torch.load(ROOT / "data" / "splits" / f"{ds}_seed{args.seed}.pt", weights_only=False)
        data.train_mask, data.val_mask, data.test_mask = (
            split["train_mask"],
            split["val_mask"],
            split["test_mask"],
        )
        guard = LeakageGuard()
        guard.lock_test()
        guard.set_phase("baseline_sparsify")

        # Original graph control
        guard.unlock_test_for_eval()
        _, m0 = train_node_classifier(data, kind="gcn", epochs=80, device=device)
        results.append({"dataset": ds, "method": "original", "removal_rate": 0.0, **m0})
        guard.lock_test()

        for method in args.methods:
            for r in args.budgets:
                if method in CLASSICAL_METHODS:
                    sparse = CLASSICAL_METHODS[method](data, r, seed=args.seed) if method in ("random", "dropedge") else CLASSICAL_METHODS[method](data, r)
                elif method == "dspar":
                    sparse = dspar_sparsify(data, r)
                elif method == "neuralsparse":
                    sparse = neuralsparse_sparsify(data, r, device=device)
                elif method == "ptdnet":
                    sparse = ptdnet_sparsify(data, r, device=device)
                elif method == "original_ilp":
                    sparse = sparsify_with_ilp(data, r, epochs=30, device=device)
                else:
                    raise ValueError(method)
                sparse.train_mask, sparse.val_mask, sparse.test_mask = (
                    data.train_mask,
                    data.val_mask,
                    data.test_mask,
                )
                guard.unlock_test_for_eval()
                _, metrics = train_node_classifier(sparse, kind="gcn", epochs=80, device=device)
                struct = structural_metrics(data, sparse)
                row = {
                    "dataset": ds,
                    "method": method,
                    "removal_rate": r,
                    **{k: metrics[k] for k in metrics if k.startswith("test_")},
                    **struct,
                    "leakage": guard.summary(),
                }
                results.append(row)
                print(row)
                guard.lock_test()
                guard.set_phase("baseline_sparsify")

    ds_tag = '_'.join(args.datasets)
    out = ROOT / "results" / "raw" / f"baselines_{ds_tag}_seed{args.seed}.json"
    save_json(results, out)
    print("Wrote", out)


if __name__ == "__main__":
    main()
