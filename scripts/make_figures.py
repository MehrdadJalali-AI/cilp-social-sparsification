#!/usr/bin/env python3
"""Generate publication-oriented figures from results (matplotlib)."""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.utils.io import ensure_dir, setup_logging


def main() -> None:
    setup_logging()
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available")
        return

    path = ROOT / "results" / "processed" / "all_results.json"
    if not path.exists():
        print("No results yet")
        return
    rows = json.loads(path.read_text())
    fig_dir = ensure_dir(ROOT / "results" / "figures")
    paper_fig = ensure_dir(ROOT / "paper" / "figures")

    # Sparsity-performance curves
    series = defaultdict(list)
    for r in rows:
        f1 = r.get("test_macro_f1", r.get("metrics.gcn.test_macro_f1"))
        method = r.get("method")
        rem = r.get("removal_rate")
        if f1 is None or method is None or rem is None:
            continue
        series[method].append((float(rem), float(f1)))

    plt.figure(figsize=(7, 4))
    for method, pts in sorted(series.items()):
        pts = sorted(pts)
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        plt.plot(xs, ys, marker="o", label=method)
    plt.xlabel("Edge removal rate")
    plt.ylabel("Test Macro-F1")
    plt.title("Sparsity–performance (Figure 6 draft)")
    plt.legend(fontsize=7, loc="best")
    plt.tight_layout()
    for d in (fig_dir, paper_fig):
        plt.savefig(d / "fig6_sparsity_performance.png", dpi=200)
    plt.close()
    print("Wrote fig6_sparsity_performance.png")

    # Simple architecture placeholder diagram text note
    (fig_dir / "FIGURE_NOTES.md").write_text(
        "Figures 1-5, 8-15 require full runs; fig6 generated from available raw results.\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
