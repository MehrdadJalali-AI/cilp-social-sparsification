#!/usr/bin/env python3
"""Aggregate raw JSON results into a unified evaluation table."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.utils.io import save_json, setup_logging


def flatten(obj, prefix=""):
    rows = []
    if isinstance(obj, list):
        for item in obj:
            rows.extend(flatten(item, prefix))
        return rows
    if isinstance(obj, dict):
        # If nested metrics
        flat = {}
        for k, v in obj.items():
            if isinstance(v, dict) and k in ("metrics", "structural", "community", "centrality", "fairness", "surrogate", "leakage"):
                for kk, vv in v.items():
                    if isinstance(vv, dict):
                        for k3, v3 in vv.items():
                            flat[f"{k}.{kk}.{k3}"] = v3
                    else:
                        flat[f"{k}.{kk}"] = vv
            else:
                flat[k] = v
        rows.append(flat)
    return rows


def main() -> None:
    setup_logging()
    raw = ROOT / "results" / "raw"
    all_rows = []
    for path in sorted(raw.glob("*.json")):
        try:
            data = json.loads(path.read_text())
        except Exception:
            continue
        for row in flatten(data):
            row["source_file"] = path.name
            all_rows.append(row)
    out = ROOT / "results" / "processed" / "all_results.json"
    save_json(all_rows, out)
    print(f"Aggregated {len(all_rows)} rows -> {out}")


if __name__ == "__main__":
    main()
