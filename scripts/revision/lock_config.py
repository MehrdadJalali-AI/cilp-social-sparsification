#!/usr/bin/env python3
"""Apply the pre-declared teacher-size rule (docs/revision/PREREGISTRATION.md §2) and write
configs/revised/locked.json. Reads only selection-phase outputs (validation metrics; no test data exist).

Also writes results/revised/tables/selection_n_sensitivity.csv: validation Macro-F1 at r = 0.5 and
held-out surrogate/scorer–teacher rank agreement (Spearman, Kendall) as a function of n.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.revision.common import N_GRID, REVISED_DIR, git_commit, sha_of  # noqa: E402

LOCK = ROOT / "configs" / "revised" / "locked.json"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=["lastfm", "facebook", "github", "pubmed"])
    ap.add_argument("--seeds", type=int, default=10)
    a = ap.parse_args()
    agg = defaultdict(lambda: defaultdict(list))
    for p in (REVISED_DIR / "select").glob("*/seed*/*.json"):
        r = json.loads(p.read_text())
        assert all("test_macro_f1" not in row for row in r["rows"]), f"test metric found in {p}"
        k = (r["dataset"], r["variant"], r["teacher_n"])
        agg[k]["val"].append(r["rows"][0]["val_macro_f1"])
        for m in ("holdout_surrogate_spearman", "holdout_surrogate_kendall", "holdout_scorer_spearman", "holdout_scorer_kendall"):
            agg[k][m].append(r["agreement"][m])
        agg[k]["insample_spearman"].append(r["surrogate"]["insample_spearman"])
    out_rows, locked, incomplete = [], {}, []
    for ds in a.datasets:
        locked[ds] = {}
        for t in ("cilp_full", "task_only"):
            means = {}
            for n in N_GRID:
                v = agg.get((ds, t, n))
                if not v or len(v["val"]) < a.seeds:
                    incomplete.append((ds, t, n, 0 if not v else len(v["val"])))
                    continue
                means[n] = float(np.mean(v["val"]))
                row = {"dataset": ds, "teacher": t, "n": n, "seeds": len(v["val"]),
                       "val_macro_f1_r0.5_mean": means[n], "val_macro_f1_r0.5_sd": float(np.std(v["val"], ddof=1))}
                for m in ("holdout_surrogate_spearman", "holdout_surrogate_kendall", "holdout_scorer_spearman",
                          "holdout_scorer_kendall", "insample_spearman"):
                    x = np.array(v[m], dtype=float)
                    row[f"{m}_mean"] = float(np.nanmean(x))
                    row[f"{m}_sd"] = float(np.nanstd(x, ddof=1))
                out_rows.append(row)
            if len(means) == len(N_GRID):
                best = max(means.values())
                locked[ds][t] = min(n for n, m in means.items() if best - m <= 1e-4)  # ties → smaller n
    tab = REVISED_DIR / "tables" / "selection_n_sensitivity.csv"
    tab.parent.mkdir(parents=True, exist_ok=True)
    with tab.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(out_rows[0].keys()))
        w.writeheader()
        w.writerows(out_rows)
    if incomplete:
        print("INCOMPLETE selection cells (no lock written):", incomplete)
        return
    body = {"protocol": "REVISED", "rule": "PREREGISTRATION.md §2", "teacher_n": locked,
            "selection_commit": git_commit(), "selection_table": str(tab.relative_to(ROOT))}
    body["sha"] = sha_of(body)
    LOCK.parent.mkdir(parents=True, exist_ok=True)
    LOCK.write_text(json.dumps(body, indent=2))
    print(json.dumps(body, indent=2))


if __name__ == "__main__":
    main()
