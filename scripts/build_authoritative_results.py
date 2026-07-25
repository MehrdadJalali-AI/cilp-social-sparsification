#!/usr/bin/env python3
"""Build authoritative_results from raw grid JSON (dedupe aliases)."""
from __future__ import annotations

import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

GRID = ROOT / "results" / "raw" / "grid"
OUT = ROOT / "results" / "processed"
TAB = ROOT / "paper" / "tables"

CANON = {
    "cailp_multi": "cailp_social_multi",
    "cailp_a31": "cailp_a31",
    "original_ilp": "original_ilp_gcn",
    "ptdnet": "ptdnet",
    "neuralsparse": "neuralsparse",
    "random": "random",
    "dspar": "resistance_style_proxy",
    "effective_resistance": "resistance_style_proxy",
}
DISPLAY = {
    "cailp_social_multi": "CILP",
    "cailp_a31": "Task-only",
    "original_ilp_gcn": "ILP-GCN",
    "ptdnet": "PTDNet",
    "neuralsparse": "NeuralSparse",
    "random": "Random",
    "resistance_style_proxy": "Resistance proxy",
}
CORE_ORDER = list(DISPLAY.keys())


def sparsity_auc(pts):
    pts = sorted(pts)
    if len(pts) < 2:
        return float("nan")
    xs = np.array([p[0] for p in pts])
    ys = np.array([p[1] for p in pts])
    return float(np.trapz(ys, xs))


def paired_stats(a, b):
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    d = a - b
    n = len(d)
    if n < 2:
        return {
            "n": n,
            "mean_diff": float(d.mean()) if n else float("nan"),
            "ci_low": float("nan"),
            "ci_high": float("nan"),
            "t_stat": float("nan"),
            "t_p": float("nan"),
            "wilcoxon_stat": float("nan"),
            "wilcoxon_p": float("nan"),
            "dz": float("nan"),
            "wins": int((d > 0).sum()) if n else 0,
            "losses": int((d < 0).sum()) if n else 0,
            "ties": int((d == 0).sum()) if n else 0,
        }
    mean_d = float(d.mean())
    sd = float(d.std(ddof=1))
    sem = sd / np.sqrt(n)
    tcrit = float(stats.t.ppf(0.975, n - 1))
    t_stat, t_p = stats.ttest_rel(a, b)
    try:
        w = stats.wilcoxon(d)
        w_stat, w_p = float(w.statistic), float(w.pvalue)
    except Exception:
        w_stat, w_p = float("nan"), float("nan")
    dz = mean_d / sd if sd > 1e-12 else float("nan")
    return {
        "n": n,
        "mean_diff": mean_d,
        "ci_low": mean_d - tcrit * sem,
        "ci_high": mean_d + tcrit * sem,
        "t_stat": float(t_stat),
        "t_p": float(t_p),
        "wilcoxon_stat": w_stat,
        "wilcoxon_p": w_p,
        "dz": float(dz),
        "wins": int((d > 0).sum()),
        "losses": int((d < 0).sum()),
        "ties": int((d == 0).sum()),
    }


def holm(pvals):
    m = len(pvals)
    order = np.argsort(pvals)
    adj_sorted = []
    running = 0.0
    for i, idx in enumerate(order):
        val = min(1.0, pvals[idx] * (m - i))
        running = max(running, val)
        adj_sorted.append(running)
    out = np.empty(m)
    for i, idx in enumerate(order):
        out[idx] = adj_sorted[i]
    return out


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    TAB.mkdir(parents=True, exist_ok=True)
    raw = []
    for p in sorted(GRID.glob("*.json")):
        if p.name.startswith("ablation_") or p.name.endswith(".error.json"):
            continue
        if "teacher" in p.name:
            continue
        data = json.loads(p.read_text())
        items = data if isinstance(data, list) else ([data] if isinstance(data, dict) and "test_macro_f1" in data else [])
        for r in items:
            if r.get("method") not in CANON:
                continue
            raw.append({**r, "_src": p.name, "_canon": CANON[r["method"]], "_raw": r["method"]})

    auth_map = {}
    # Prefer dspar over effective_resistance for proxy
    for prefer in ("dspar", "effective_resistance"):
        for r in raw:
            if r["_canon"] != "resistance_style_proxy" or r["_raw"] != prefer:
                continue
            key = (r["dataset"], r["_canon"], int(r["seed"]), round(float(r["removal_rate"]), 1))
            if key not in auth_map:
                auth_map[key] = r
    for r in raw:
        if r["_canon"] == "resistance_style_proxy":
            continue
        key = (r["dataset"], r["_canon"], int(r["seed"]), round(float(r["removal_rate"]), 1))
        if key in auth_map:
            if abs(auth_map[key]["test_macro_f1"] - r["test_macro_f1"]) > 1e-10:
                raise SystemExit(f"CONFLICT {key}")
            continue
        auth_map[key] = r

    rows = []
    for r in sorted(auth_map.values(), key=lambda x: (x["dataset"], x["_canon"], x["seed"], float(x["removal_rate"]))):
        rows.append(
            {
                "dataset": r["dataset"],
                "method": r["_canon"],
                "method_display": DISPLAY[r["_canon"]],
                "raw_method_key": r["_raw"],
                "seed": int(r["seed"]),
                "edge_removal_rate": round(float(r["removal_rate"]), 1),
                "test_macro_f1": float(r["test_macro_f1"]),
                "val_macro_f1": r.get("val_macro_f1"),
                "test_accuracy": r.get("test_accuracy"),
                "test_worst_class_f1": r.get("test_worst_class_f1"),
                "train_seconds": r.get("train_seconds"),
                "prune_seconds": r.get("prune_seconds"),
                "giant_component_ratio": r.get("giant_component_ratio"),
                "bridge_preservation": r.get("bridge_preservation"),
                "minority_degree_retention": r.get("minority_degree_retention"),
                "minority_isolation_rate": r.get("minority_isolation_rate"),
                "clustering": r.get("clustering"),
                "num_components": r.get("num_components"),
                "degree_ks": r.get("degree_ks"),
                "retained_edge_ratio": r.get("retained_edge_ratio"),
                "source_file": r["_src"],
            }
        )

    fields = list(rows[0].keys())
    with (OUT / "authoritative_results.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    (OUT / "authoritative_results.json").write_text(json.dumps(rows, indent=2))
    print(f"Wrote {len(rows)} authoritative rows (from {len(raw)} raw)")

    # seed audit
    audit = []
    for ds in sorted({r["dataset"] for r in rows}):
        for m in CORE_ORDER:
            seeds = sorted({r["seed"] for r in rows if r["dataset"] == ds and r["method"] == m})
            if not seeds:
                continue
            audit.append(
                {
                    "dataset": ds,
                    "method": m,
                    "method_display": DISPLAY[m],
                    "n_seeds": len(seeds),
                    "seed_ids": " ".join(map(str, seeds)),
                }
            )
    with (TAB / "facebook_seed_audit.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["dataset", "method", "method_display", "n_seeds", "seed_ids"])
        w.writeheader()
        w.writerows([a for a in audit if a["dataset"] == "facebook"])
    with (TAB / "seed_audit.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["dataset", "method", "method_display", "n_seeds", "seed_ids"])
        w.writeheader()
        w.writerows(audit)

    # paired comparisons
    common_rows = []
    for ds in sorted({r["dataset"] for r in rows}):
        by_m = defaultdict(dict)
        by_m50 = defaultdict(dict)
        for r in rows:
            if r["dataset"] != ds:
                continue
            by_m[r["method"]].setdefault(r["seed"], []).append((r["edge_removal_rate"], r["test_macro_f1"]))
            if abs(r["edge_removal_rate"] - 0.5) < 1e-9:
                by_m50[r["method"]][r["seed"]] = r["test_macro_f1"]
        aucs = {m: {s: sparsity_auc(pts) for s, pts in sm.items()} for m, sm in by_m.items()}
        multi = "cailp_social_multi"
        for other in CORE_ORDER:
            if other == multi:
                continue
            common = sorted(set(aucs.get(multi, {})) & set(aucs.get(other, {})))
            if len(common) < 2:
                continue
            st = paired_stats([aucs[multi][s] for s in common], [aucs[other][s] for s in common])
            common_rows.append(
                {
                    "dataset": ds,
                    "metric": "sparsity_macro_f1_auc",
                    "method_a": multi,
                    "method_b": other,
                    "seed_ids": " ".join(map(str, common)),
                    **st,
                }
            )
        for other in ["cailp_a31", "original_ilp_gcn", "ptdnet", "random"]:
            common = sorted(set(by_m50.get(multi, {})) & set(by_m50.get(other, {})))
            if len(common) < 2:
                continue
            st = paired_stats([by_m50[multi][s] for s in common], [by_m50[other][s] for s in common])
            common_rows.append(
                {
                    "dataset": ds,
                    "metric": "macro_f1_at_0.5",
                    "method_a": multi,
                    "method_b": other,
                    "seed_ids": " ".join(map(str, common)),
                    **st,
                }
            )
    for ds in sorted({r["dataset"] for r in common_rows}):
        idxs = [i for i, r in enumerate(common_rows) if r["dataset"] == ds and r["metric"] == "sparsity_macro_f1_auc"]
        ps = [1.0 if np.isnan(common_rows[i]["wilcoxon_p"]) else common_rows[i]["wilcoxon_p"] for i in idxs]
        if not idxs:
            continue
        adj = holm(ps)
        for i, a in zip(idxs, adj):
            common_rows[i]["wilcoxon_p_holm"] = float(a)

    with (TAB / "common_seed_comparisons.csv").open("w", newline="") as f:
        fields2 = [
            "dataset",
            "metric",
            "method_a",
            "method_b",
            "seed_ids",
            "n",
            "mean_diff",
            "ci_low",
            "ci_high",
            "t_stat",
            "t_p",
            "wilcoxon_stat",
            "wilcoxon_p",
            "wilcoxon_p_holm",
            "dz",
            "wins",
            "losses",
            "ties",
        ]
        w = csv.DictWriter(f, fieldnames=fields2, extrasaction="ignore")
        w.writeheader()
        for r in common_rows:
            w.writerow(r)
    print(f"Wrote {len(common_rows)} paired comparisons")


if __name__ == "__main__":
    main()
