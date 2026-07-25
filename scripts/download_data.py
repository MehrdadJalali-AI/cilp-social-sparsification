#!/usr/bin/env python3
"""Download social-network benchmarks via PyTorch Geometric."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.datasets import download_dataset, preprocess_graph, save_processed
from src.utils.io import setup_logging


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", default=["facebook", "lastfm", "github"])
    parser.add_argument("--all", action="store_true")
    args = parser.parse_args()
    setup_logging()
    names = ["facebook", "lastfm", "github"] if args.all else args.datasets
    for name in names:
        print(f"Downloading {name}...")
        data = download_dataset(name, raw_dir=ROOT / "data" / "raw")
        data, audit = preprocess_graph(data)
        path = save_processed(data, name, processed_dir=ROOT / "data" / "processed")
        print(f"  saved {path} | nodes={data.num_nodes} edges={data.edge_index.size(1)} audit={audit}")


if __name__ == "__main__":
    main()
