#!/usr/bin/env python3
"""Generate markdown/CSV tables from aggregated results."""
from __future__ import annotations

import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.utils.io import ensure_dir, setup_logging


def main() -> None:
    setup_logging()
    path = ROOT / "results" / "processed" / "all_results.json"
    if not path.exists():
        print("No aggregated results; run evaluate_all.py first")
        return
    rows = json.loads(path.read_text())
    out_dir = ensure_dir(ROOT / "results" / "tables")

    # Table 3-like: node classification macro-F1 by method/budget
    key_f1 = None
    for cand in ("test_macro_f1", "metrics.gcn.test_macro_f1"):
        if any(cand in r for r in rows):
            key_f1 = cand
            break

    table_path = out_dir / "table_node_classification.csv"
    with open(table_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["dataset", "method", "removal_rate", "test_macro_f1", "source_file"])
        for r in rows:
            f1 = r.get("test_macro_f1", r.get("metrics.gcn.test_macro_f1"))
            if f1 is None:
                continue
            w.writerow([
                r.get("dataset", ""),
                r.get("method", ""),
                r.get("removal_rate", ""),
                f1,
                r.get("source_file", ""),
            ])
    print("Wrote", table_path)

    # RQ11 summary if both teachers present
    rq11 = [r for r in rows if r.get("teacher") in ("multi_objective", "single_objective")]
    md = ["# Auto-generated Tables", "", "## Node classification (extract)", ""]
    md.append(f"Rows with F1: see `{table_path.name}`")
    md.append("")
    md.append("## RQ11 rows")
    md.append("")
    md.append(f"Found {len(rq11)} CAILP teacher-comparison rows.")
    (out_dir / "README.md").write_text("\n".join(md), encoding="utf-8")

    # Also write results/tables stub
    paper_tab = ensure_dir(ROOT / "paper" / "tables")
    (paper_tab / "table3_node_classification.csv").write_text(table_path.read_text(), encoding="utf-8")


if __name__ == "__main__":
    main()
