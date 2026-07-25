#!/usr/bin/env python3
"""Fail if reader-facing manuscript files contain banned terminology."""
from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

SCAN_GLOBS = [
    "paper/*.tex",
    "paper/sections/*.tex",
    "paper/figures/*.tikz",
    "paper/tables/*.tex",
]

ALLOW_INTERNAL_ID_FILES = {
    ROOT / "paper" / "tables" / "internal_id_mapping.tex",
    ROOT / "paper" / "tables" / "internal_id_mapping.csv",
}

BANNED = [
    r"CAILP-Social",
    r"Counterfactual-Aware",
    r"Attention-Based Inverse Link Prediction",
    r"\bA31\b",
    r"\bRQ11\b",
    r"multi-obj\.",
    r"[Ff]idelity-only",
    r"concat multi",
    r"DSpar/ER",
    r"resistance-style proxy",
    r"CAILP multi",
    r"CAILP-Social multi",
    r"Project Team",
    r"authoritative store",
    r"multi-objective CAILP",
    r"Original ILP-GCN",
]

INTERNAL_ID_PATTERNS = [
    r"\bA1\b",
    r"\bA2\b",
    r"\bA8\b",
    r"\bA9\b",
    r"\bA10\b",
    r"\bA19\b",
    r"\bA20\b",
    r"\bA23\b",
    r"\bA24\b",
    r"\bA25\b",
    r"\bA26\b",
    r"\bA27\b",
    r"\bA28\b",
    r"\bA32\b",
    r"A26--A28",
    r"A1--A32",
]


def scan_files() -> list[Path]:
    files: list[Path] = []
    for g in SCAN_GLOBS:
        files.extend(ROOT.glob(g))
    # Exclude archives
    return sorted({p.resolve() for p in files if p.is_file() and "_archive" not in str(p)})


def main() -> int:
    failures: list[str] = []
    for path in scan_files():
        if path in ALLOW_INTERNAL_ID_FILES:
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        rel = path.relative_to(ROOT)
        for pat in BANNED:
            for m in re.finditer(pat, text):
                line = text.count("\n", 0, m.start()) + 1
                failures.append(f"{rel}:{line}: banned '{m.group(0)}' (/{pat}/)")
        for pat in INTERNAL_ID_PATTERNS:
            for m in re.finditer(pat, text):
                line = text.count("\n", 0, m.start()) + 1
                failures.append(f"{rel}:{line}: internal ID '{m.group(0)}' outside mapping table")
    if failures:
        print("Terminology validation FAILED:")
        for f in failures:
            print(" ", f)
        return 1
    print(f"Terminology validation OK ({len(scan_files())} files).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
