#!/usr/bin/env python3
"""Run Stage B recent baselines into the provisional result store only."""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.run_full_grid import BUDGETS, GRID_CFG, eval_importance, load_data
from src.sparsification.baselines.gassip_adapted import gassip_adapted_score_edges
from src.sparsification.baselines.mog_adapted import mog_adapted_score_edges
from src.sparsification.baselines.psgnn_reimpl import psgnn_reimpl_score_edges
from src.utils.io import ensure_dir, get_device, save_json, set_seed, setup_logging

STAGE_B_METHODS = {
    "gassip_adapted": gassip_adapted_score_edges,
    "mog_adapted": mog_adapted_score_edges,
    "psgnn_reimpl": psgnn_reimpl_score_edges,
}

METHOD_DISPLAY = {
    "gassip_adapted": "GASSIP-adapted",
    "mog_adapted": "MoG-adapted",
    "psgnn_reimpl": "PSGNN-reimpl",
}

PROV_DIR = ROOT / "results" / "processed" / "recent_baselines"
RAW_DIR = ROOT / "results" / "raw" / "stage_b"


def score_stage_b(data, method: str, device, epochs: int):
    fn = STAGE_B_METHODS[method]
    return fn(data, epochs=epochs, device=device)


def _row_key(r: Dict[str, Any]) -> tuple:
    return (r["dataset"], r["method"], int(r["seed"]), float(r["edge_removal_rate"]))


def load_existing_provisional() -> List[Dict[str, Any]]:
    csv_path = PROV_DIR / "stage_b_results.csv"
    if not csv_path.exists():
        return []
    with csv_path.open() as f:
        return list(csv.DictReader(f))


def merge_provisional(existing: List[Dict[str, Any]], new_rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    merged = {_row_key(r): r for r in existing if "test_macro_f1" in r}
    for r in new_rows:
        if "test_macro_f1" in r:
            merged[_row_key(r)] = r
    return list(merged.values())


def rebuild_from_raw() -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if not RAW_DIR.exists():
        return rows
    for path in sorted(RAW_DIR.glob("*/*/*.json")):
        data = json.loads(path.read_text())
        if isinstance(data, list):
            rows.extend(data)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", default=["lastfm", "facebook", "github"])
    parser.add_argument("--seeds", nargs="+", type=int, default=list(range(10)))
    parser.add_argument("--budgets", nargs="+", type=float, default=BUDGETS)
    parser.add_argument("--methods", nargs="+", default=list(STAGE_B_METHODS.keys()))
    parser.add_argument("--rebuild-from-raw", action="store_true", help="Rebuild provisional CSV from raw JSON only")
    args = parser.parse_args()
    setup_logging()

    if args.rebuild_from_raw:
        ensure_dir(PROV_DIR)
        merged = rebuild_from_raw()
        for r in merged:
            if "removal_rate" in r and "edge_removal_rate" not in r:
                r["edge_removal_rate"] = r.pop("removal_rate")
        _write_provisional(merged)
        print(f"Rebuilt {len(merged)} rows from raw JSON", flush=True)
        return

    device = get_device()
    ensure_dir(PROV_DIR)
    ensure_dir(RAW_DIR)

    all_rows: List[Dict[str, Any]] = []
    for ds in args.datasets:
        cfg = GRID_CFG[ds]
        down_epochs = cfg["down_epochs"]
        train_epochs = cfg.get("train_epochs", 25)
        for seed in args.seeds:
            set_seed(seed)
            data = load_data(ds, seed)
            data.name = ds
            for method in args.methods:
                if method not in STAGE_B_METHODS:
                    raise ValueError(f"Unknown Stage B method: {method}")
                print(f"[Stage B] {method} {ds} seed={seed}", flush=True)
                try:
                    und, imp, train_s = score_stage_b(data, method, device, epochs=train_epochs)
                    extra = {"train_seconds": train_s, "stage": "B", "method_display": METHOD_DISPLAY[method]}
                    rows = eval_importance(
                        data,
                        und,
                        imp,
                        args.budgets,
                        device,
                        down_epochs,
                        method,
                        seed,
                        extra=extra,
                    )
                    for r in rows:
                        r["dataset"] = ds
                        r["edge_removal_rate"] = r.pop("removal_rate")
                    raw_path = RAW_DIR / method / ds / f"seed_{seed}.json"
                    ensure_dir(raw_path.parent)
                    save_json(rows, raw_path)
                    all_rows.extend(rows)
                except Exception as exc:
                    err = {
                        "dataset": ds,
                        "method": method,
                        "seed": seed,
                        "error": str(exc),
                        "stage": "B",
                    }
                    print(f"  ERROR: {exc}", flush=True)
                    all_rows.append(err)

    existing = load_existing_provisional()
    valid_new = [r for r in all_rows if "test_macro_f1" in r]
    merged = merge_provisional(existing, valid_new)
    _write_provisional(merged)
    print(f"Wrote {len(merged)} provisional Stage B rows ({len(valid_new)} new/updated)", flush=True)


def _write_provisional(valid: List[Dict[str, Any]]) -> None:
    csv_path = PROV_DIR / "stage_b_results.csv"
    fieldnames = [
        "dataset",
        "method",
        "method_display",
        "seed",
        "edge_removal_rate",
        "test_macro_f1",
        "val_macro_f1",
        "test_accuracy",
        "test_worst_class_f1",
        "train_seconds",
        "prune_seconds",
        "giant_component_ratio",
        "bridge_preservation",
        "minority_degree_retention",
        "retained_edge_ratio",
        "stage",
    ]
    with csv_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        for r in sorted(valid, key=lambda x: (x["dataset"], x["method"], int(x["seed"]), float(x["edge_removal_rate"]))):
            w.writerow(r)
    (PROV_DIR / "stage_b_results.json").write_text(json.dumps(valid, indent=2))


if __name__ == "__main__":
    main()
