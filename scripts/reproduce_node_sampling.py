#!/usr/bin/env python3
"""Track B: reproduce node-sampling methods including Black Hole."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.sparsification.node_sampling import NODE_SAMPLERS
from src.tasks.node_classification import train_node_classifier
from src.utils.io import get_device, save_json, set_seed, setup_logging
from src.utils.splits import stratified_node_splits


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="lastfm")
    parser.add_argument("--method", default="black_hole", choices=list(NODE_SAMPLERS))
    parser.add_argument("--keep-ratios", nargs="+", type=float, default=[0.5, 0.3])
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    setup_logging()
    set_seed(args.seed)
    device = get_device()
    data = torch.load(ROOT / "data" / "processed" / f"{args.dataset}.pt", weights_only=False)
    sampler = NODE_SAMPLERS[args.method]
    results = []
    for kr in args.keep_ratios:
        if args.method == "random":
            sub = sampler(data, kr, seed=args.seed)
        else:
            sub = sampler(data, kr)
        # New stratified splits on induced subgraph labels
        tr, va, te = stratified_node_splits(sub.y, seed=args.seed)
        sub.train_mask, sub.val_mask, sub.test_mask = tr, va, te
        _, metrics = train_node_classifier(sub, kind="gcn", epochs=80, device=device)
        row = {
            "track": "B",
            "method": args.method,
            "keep_ratio_nodes": kr,
            "num_nodes": sub.num_nodes,
            "num_edges": int(sub.edge_index.size(1)),
            **metrics,
        }
        results.append(row)
        print(row)
    out = ROOT / "results" / "raw" / f"node_sample_{args.method}_{args.dataset}_seed{args.seed}.json"
    save_json(results, out)
    print("Wrote", out)


if __name__ == "__main__":
    main()
