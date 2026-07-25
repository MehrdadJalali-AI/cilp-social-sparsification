#!/usr/bin/env python3
"""Build deterministic stratified node splits for all seeds."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.utils.io import ensure_dir, setup_logging
from src.utils.splits import stratified_node_splits


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", default=["facebook", "lastfm", "github"])
    parser.add_argument("--seeds", nargs="+", type=int, default=list(range(10)))
    args = parser.parse_args()
    setup_logging()
    out_dir = ensure_dir(ROOT / "data" / "splits")
    for name in args.datasets:
        path = ROOT / "data" / "processed" / f"{name}.pt"
        if not path.exists():
            raise FileNotFoundError(f"Missing {path}; run download/audit first")
        data = torch.load(path, weights_only=False)
        for seed in args.seeds:
            train, val, test = stratified_node_splits(data.y, seed=seed)
            torch.save(
                {"train_mask": train, "val_mask": val, "test_mask": test, "seed": seed},
                out_dir / f"{name}_seed{seed}.pt",
            )
            print(f"{name} seed={seed} train={int(train.sum())} val={int(val.sum())} test={int(test.sum())}")


if __name__ == "__main__":
    main()
