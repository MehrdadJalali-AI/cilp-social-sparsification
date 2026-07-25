#!/usr/bin/env python3
"""Assemble paper stubs and copy tables/figures; does not invent novelty claims."""
from __future__ import annotations

import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    paper = ROOT / "paper"
    # Copy figures/tables if present
    src_fig = ROOT / "results" / "figures"
    dst_fig = paper / "figures"
    dst_fig.mkdir(parents=True, exist_ok=True)
    if src_fig.exists():
        for p in src_fig.glob("*.png"):
            shutil.copy2(p, dst_fig / p.name)
    src_tab = ROOT / "results" / "tables"
    dst_tab = paper / "tables"
    dst_tab.mkdir(parents=True, exist_ok=True)
    if src_tab.exists():
        for p in src_tab.glob("*.csv"):
            shutil.copy2(p, dst_tab / p.name)
    print("Paper assets synced. Compile with: cd paper && pdflatex main.tex")


if __name__ == "__main__":
    main()
