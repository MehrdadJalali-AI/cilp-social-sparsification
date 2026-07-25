#!/usr/bin/env python3
"""Core ablations including A31 (single vs multi-objective CF) and A32 (Black Hole mass)."""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def run(cmd: list[str]) -> None:
    print(">>", " ".join(cmd))
    subprocess.check_call(cmd, cwd=str(ROOT))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", default=["lastfm"])
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    py = sys.executable
    for ds in args.datasets:
        # A1 / core
        run([py, "scripts/train_cailp.py", "--dataset", ds, "--seed", str(args.seed)])
        # A31 single-objective
        run([py, "scripts/train_cailp.py", "--dataset", ds, "--seed", str(args.seed), "--single-objective"])
        # A2 original ILP
        run([py, "scripts/reproduce_original_ilp.py", "--dataset", ds, "--seed", str(args.seed)])
        # A32 / Track B black hole
        run([py, "scripts/reproduce_node_sampling.py", "--dataset", ds, "--method", "black_hole", "--seed", str(args.seed)])
        run([py, "scripts/reproduce_node_sampling.py", "--dataset", ds, "--method", "degree", "--seed", str(args.seed)])
        run([py, "scripts/reproduce_node_sampling.py", "--dataset", ds, "--method", "pagerank", "--seed", str(args.seed)])


if __name__ == "__main__":
    main()
