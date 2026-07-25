#!/usr/bin/env python3
"""Validate authoritative results consistency. Exit nonzero on failure."""
from __future__ import annotations

import csv
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
AUTH = ROOT / "results" / "processed" / "authoritative_results.csv"
SEED = ROOT / "paper" / "tables" / "seed_audit.csv"


def main() -> int:
    errors = []
    if not AUTH.exists():
        print("FAIL: missing authoritative_results.csv", file=sys.stderr)
        return 1
    rows = list(csv.DictReader(AUTH.open()))
    keys = [(r["dataset"], r["method"], r["seed"], r["edge_removal_rate"]) for r in rows]
    counts = Counter(keys)
    dups = [k for k, c in counts.items() if c > 1]
    if dups:
        errors.append(f"duplicate unique keys: {len(dups)} e.g. {dups[:3]}")

    # resistance proxy seed counts
    for ds in sorted({r["dataset"] for r in rows}):
        seeds = sorted({int(r["seed"]) for r in rows if r["dataset"] == ds and r["method"] == "resistance_style_proxy"})
        if ds == "lastfm" and len(seeds) != 10:
            errors.append(f"lastfm resistance_style_proxy n={len(seeds)} expected 10")
        if ds == "facebook" and len(seeds) != 10:
            errors.append(f"facebook resistance_style_proxy n={len(seeds)} expected 10 (got {seeds})")
        # no method should exceed 10 seeds
        for m in sorted({r["method"] for r in rows if r["dataset"] == ds}):
            n = len({r["seed"] for r in rows if r["dataset"] == ds and r["method"] == m})
            if n > 10:
                errors.append(f"{ds} {m} has n_seeds={n} > 10 (alias inflation?)")

    # budgets complete 0.1-0.9
    for ds in sorted({r["dataset"] for r in rows}):
        for m in sorted({r["method"] for r in rows if r["dataset"] == ds}):
            for seed in sorted({int(r["seed"]) for r in rows if r["dataset"] == ds and r["method"] == m}):
                budgets = {
                    float(r["edge_removal_rate"])
                    for r in rows
                    if r["dataset"] == ds and r["method"] == m and int(r["seed"]) == seed
                }
                expected = {round(0.1 * i, 1) for i in range(1, 10)}
                if budgets != expected:
                    errors.append(f"missing budgets {ds} {m} seed{seed}: {sorted(expected-budgets)}")

    # CAILP multi and A31 same seeds on facebook for paired work
    if SEED.exists():
        audits = list(csv.DictReader(SEED.open()))
        fb_multi = next((a for a in audits if a["dataset"] == "facebook" and a["method"] == "cailp_social_multi"), None)
        fb_a31 = next((a for a in audits if a["dataset"] == "facebook" and a["method"] == "cailp_a31"), None)
        if fb_multi and fb_a31 and fb_multi["seed_ids"] != fb_a31["seed_ids"]:
            errors.append(
                f"Facebook multi/A31 seed mismatch: {fb_multi['seed_ids']} vs {fb_a31['seed_ids']}"
            )

    # raw method keys for proxy must not both appear as separate methods
    methods = {r["method"] for r in rows}
    if "dspar" in methods or "effective_resistance" in methods:
        errors.append("raw dspar/effective_resistance present as method — should be resistance_style_proxy")

    # Fig5 / Table4 structural consistency
    t4 = ROOT / "paper" / "tables" / "table4_structure.csv"
    if t4.exists():
        t4_rows = list(csv.DictReader(t4.open()))
        methods_t4 = {(r["dataset"], r["method"]) for r in t4_rows}
        expected = {
            (ds, m)
            for ds in ("lastfm", "facebook", "github")
            for m in (
                "cailp_social_multi",
                "cailp_a31",
                "original_ilp_gcn",
                "ptdnet",
                "random",
            )
            if any(r["dataset"] == ds for r in rows)
        }
        if methods_t4 != expected:
            errors.append(f"Table4 method set mismatch: {sorted(methods_t4 ^ expected)}")
        for r in t4_rows:
            vals = []
            for row in rows:
                if (
                    row["dataset"] == r["dataset"]
                    and row["method"] == r["method"]
                    and abs(float(row["edge_removal_rate"]) - 0.5) < 1e-9
                    and row.get("giant_component_ratio") not in (None, "")
                ):
                    vals.append(float(row["giant_component_ratio"]))
            if not vals:
                errors.append(f"Table4 missing auth data for {r['dataset']} {r['method']}")
                continue
            mu = sum(vals) / len(vals)
            if abs(mu - float(r["giant_component_ratio"])) > 1e-6:
                errors.append(
                    f"Table4 GC mismatch {r['dataset']} {r['method']}: table={r['giant_component_ratio']} auth={mu}"
                )
            if int(r["n"]) != len({int(row["seed"]) for row in rows if row["dataset"]==r["dataset"] and row["method"]==r["method"] and abs(float(row["edge_removal_rate"])-0.5)<1e-9}):
                errors.append(f"Table4 n mismatch {r['dataset']} {r['method']}")

    if errors:
        print("VALIDATION FAILED:", file=sys.stderr)
        for e in errors:
            print(" -", e, file=sys.stderr)
        return 1
    print(f"OK: {len(rows)} authoritative rows, no duplicate keys, budgets complete")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
