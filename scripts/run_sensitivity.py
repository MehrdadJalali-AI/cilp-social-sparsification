#!/usr/bin/env python3
"""Lightweight sensitivity sweeps on LastFM."""
from __future__ import annotations

import argparse
import copy
import sys
from pathlib import Path

import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.utils.io import save_json, setup_logging


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="lastfm")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    setup_logging()
    base = yaml.safe_load((ROOT / "configs/experiments/pilot.yaml").read_text())
    grid = {
        "hidden_dim": [32, 64],
        "dropout": [0.2, 0.5],
        "lambda_cf": [0.5, 1.0, 2.0],
        "fusion": ["concat", "gated", "cross_attention"],
    }
    jobs = []
    for key, values in grid.items():
        for v in values:
            cfg = copy.deepcopy(base)
            cfg["dataset"] = args.dataset
            cfg["seed"] = args.seed
            if key in ("hidden_dim", "dropout"):
                cfg["encoder"][key if key != "hidden_dim" else "hidden_dim"] = v
                if key == "hidden_dim":
                    cfg["encoder"]["hidden_dim"] = v
                else:
                    cfg["encoder"]["dropout"] = v
            elif key == "lambda_cf":
                cfg["train"]["lambda_cf"] = v
            elif key == "fusion":
                cfg["fusion"] = v
            path = ROOT / "configs" / "experiments" / f"sens_{key}_{v}.yaml"
            path.write_text(yaml.dump(cfg), encoding="utf-8")
            jobs.append({"key": key, "value": v, "config": str(path.relative_to(ROOT))})
    save_json(jobs, ROOT / "results" / "processed" / "sensitivity_jobs.json")
    print(f"Wrote {len(jobs)} sensitivity configs. Run train_cailp.py --config <path> for each.")
    # Execute a minimal subset immediately
    import subprocess

    for job in jobs[:3]:
        subprocess.check_call(
            [
                sys.executable,
                "scripts/train_cailp.py",
                "--dataset",
                args.dataset,
                "--config",
                job["config"],
                "--seed",
                str(args.seed),
            ],
            cwd=str(ROOT),
        )


if __name__ == "__main__":
    main()
