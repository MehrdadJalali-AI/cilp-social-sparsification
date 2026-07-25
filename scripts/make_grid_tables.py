#!/usr/bin/env python3
"""Export publication tables from grid results (no novelty claims)."""
from __future__ import annotations

import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.analyze_grid import load_grid_rows, sparsity_auc
from src.evaluation.statistics import summarize
from src.utils.io import ensure_dir


def main() -> None:
    rows = load_grid_rows()
    out = ensure_dir(ROOT / "results" / "tables")
    # Node classification wide table
    path = out / "table3_multiseed_macro_f1.csv"
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        budgets = [i / 10 for i in range(1, 10)]
        w.writerow(["dataset", "method", "n_seeds"] + [f"rem_{b:.1f}_mean" for b in budgets] + [f"rem_{b:.1f}_std" for b in budgets] + ["auc_mean", "auc_std"])
        datasets = sorted({r.get("dataset") for r in rows if r.get("dataset")})
        methods = sorted({r.get("method") for r in rows if r.get("method") and not str(r.get("method")).startswith("ablation") and "teacher" not in str(r.get("method", ""))})
        for ds in datasets:
            for method in methods:
                # filter diagnostic
                if "cailp_multi_teacher" in method or method.startswith("A"):
                    continue
                aucs = []
                by_seed = defaultdict(list)
                for r in rows:
                    if r.get("dataset") != ds or r.get("method") != method:
                        continue
                    if r.get("test_macro_f1") is None:
                        continue
                    by_seed[r["seed"]].append((float(r["removal_rate"]), float(r["test_macro_f1"])))
                if not by_seed:
                    continue
                means, stds = [], []
                for b in budgets:
                    vals = []
                    for seed, pts in by_seed.items():
                        for rem, f1 in pts:
                            if abs(rem - b) < 1e-9:
                                vals.append(f1)
                    s = summarize(vals)
                    means.append(s["mean"])
                    stds.append(s["std"])
                for seed, pts in by_seed.items():
                    aucs.append(sparsity_auc([p[0] for p in pts], [p[1] for p in pts]))
                as_ = summarize(aucs)
                w.writerow([ds, method, len(by_seed)] + means + stds + [as_["mean"], as_["std"]])
    print("Wrote", path)

    # Surrogate table
    path2 = out / "table_surrogate_quality.csv"
    with open(path2, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["dataset", "method", "seed", "mae", "spearman", "kendall", "top_k_precision", "top_k_recall", "calibration_ece", "teacher_n"])
        seen = set()
        for r in rows:
            sur = r.get("surrogate")
            if not sur:
                continue
            key = (r.get("dataset"), r.get("method"), r.get("seed"))
            if key in seen:
                continue
            seen.add(key)
            w.writerow([r.get("dataset"), r.get("method"), r.get("seed"), sur.get("mae"), sur.get("spearman"), sur.get("kendall"), sur.get("top_k_precision"), sur.get("top_k_recall"), sur.get("calibration_ece"), sur.get("teacher_n")])
    print("Wrote", path2)

    # Paired tests export
    pt = ROOT / "results" / "processed" / "paired_tests.json"
    if pt.exists():
        rows_pt = json.loads(pt.read_text())
        path3 = out / "table14_paired_tests.csv"
        with open(path3, "w", newline="", encoding="utf-8") as f:
            if rows_pt:
                keys = list(rows_pt[0].keys())
                w = csv.DictWriter(f, fieldnames=keys)
                w.writeheader()
                w.writerows(rows_pt)
        print("Wrote", path3)


if __name__ == "__main__":
    main()
