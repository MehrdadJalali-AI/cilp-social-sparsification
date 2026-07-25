#!/usr/bin/env python3
"""Facebook negative-result investigation: teacher size, stratification, surrogate quality.

Uses validation Macro-F1 only for any comparative selection among diagnostic variants.
Test metrics are reported for all variants without selection on test.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.run_full_grid import eval_importance, load_data, score_cailp
from src.utils.io import get_device, save_json, set_seed, setup_logging


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--budgets", nargs="+", type=float, default=[0.3, 0.5, 0.7])
    args = parser.parse_args()
    setup_logging()
    device = get_device()
    set_seed(args.seed)
    data = load_data("facebook", args.seed)
    data.name = "facebook"

    variants = [
        {"name": "teacher20", "teacher_n": 20, "teacher_epochs": 20, "train_epochs": 20},
        {"name": "teacher40", "teacher_n": 40, "teacher_epochs": 20, "train_epochs": 20},
        {"name": "teacher80", "teacher_n": 80, "teacher_epochs": 25, "train_epochs": 25},
        {"name": "teacher120", "teacher_n": 120, "teacher_epochs": 25, "train_epochs": 30},
    ]
    all_rows = []
    for v in variants:
        print(f"Facebook diagnostic: {v['name']}", flush=True)
        und, imp, sur, ts = score_cailp(
            data,
            device,
            args.seed,
            single_objective=False,
            teacher_n=v["teacher_n"],
            teacher_epochs=v["teacher_epochs"],
            train_epochs=v["train_epochs"],
        )
        rows = eval_importance(
            data,
            und,
            imp,
            args.budgets,
            device,
            down_epochs=40,
            method=f"cailp_multi_{v['name']}",
            seed=args.seed,
            extra={"train_seconds": ts, "surrogate": sur, "diagnostic": v},
        )
        all_rows.extend(rows)
        # Report surrogate quality explicitly
        print(
            f"  surrogate MAE={sur['mae']:.4f} spearman={sur['spearman']:.4f} "
            f"kendall={sur['kendall']:.4f} ece={sur.get('calibration_ece', float('nan')):.4f}",
            flush=True,
        )

    out = ROOT / "results" / "raw" / "grid" / f"facebook_diagnostic_seed{args.seed}.json"
    save_json(all_rows, out)
    # Rank variants by VALIDATION macro-F1 at 50% only (no test selection)
    at50 = [r for r in all_rows if r["removal_rate"] == 0.5]
    at50_sorted = sorted(at50, key=lambda r: r["val_macro_f1"], reverse=True)
    report = {
        "note": "Variants ranked by validation Macro-F1 at 50% removal; test reported for all.",
        "ranked_by_val_at_50": [
            {
                "method": r["method"],
                "val_macro_f1": r["val_macro_f1"],
                "test_macro_f1": r["test_macro_f1"],
                "surrogate": r.get("surrogate"),
            }
            for r in at50_sorted
        ],
    }
    save_json(report, ROOT / "results" / "processed" / f"facebook_diagnostic_seed{args.seed}_summary.json")
    print("Wrote", out)


if __name__ == "__main__":
    main()
