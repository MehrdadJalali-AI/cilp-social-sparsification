#!/usr/bin/env python3
"""Aggregate grid results: stats, RQ11 decision, curves, AUC, Pareto."""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.evaluation.statistics import friedman_test, holm_correct, paired_compare, summarize
from src.utils.io import ensure_dir, save_json, setup_logging


def load_grid_rows() -> List[dict]:
    rows = []
    grid = ROOT / "results" / "raw" / "grid"
    if not grid.exists():
        return rows
    for p in sorted(grid.glob("*.json")):
        if p.name.endswith(".error.json"):
            continue
        # Keep ablation JSONs out of the core multi-seed / RQ11 analysis
        if p.name.startswith("ablation_"):
            continue
        try:
            data = json.loads(p.read_text())
        except Exception:
            continue
        if isinstance(data, list):
            rows.extend(data)
        elif isinstance(data, dict) and "test_macro_f1" in data:
            rows.append(data)
    return rows


def sparsity_auc(budgets: List[float], f1s: List[float]) -> float:
    """Area under sparsity–performance curve (removal_rate vs Macro-F1), trapezoid."""
    pairs = sorted(zip(budgets, f1s))
    if len(pairs) < 2:
        return float("nan")
    xs = np.array([p[0] for p in pairs], dtype=float)
    ys = np.array([p[1] for p in pairs], dtype=float)
    return float(np.trapz(ys, xs))


def main() -> None:
    setup_logging()
    rows = load_grid_rows()
    out_dir = ensure_dir(ROOT / "results" / "processed")
    fig_dir = ensure_dir(ROOT / "results" / "figures")
    save_json(rows, out_dir / "grid_all_rows.json")
    print(f"Loaded {len(rows)} grid rows")

    # Surrogate quality table
    sur_rows = []
    seen = set()
    for r in rows:
        sur = r.get("surrogate")
        if not sur:
            continue
        key = (r.get("dataset"), r.get("seed"), r.get("method"), r.get("teacher"))
        if key in seen:
            continue
        seen.add(key)
        sur_rows.append(
            {
                "dataset": r.get("dataset"),
                "seed": r.get("seed"),
                "method": r.get("method"),
                "teacher": r.get("teacher") or sur.get("teacher"),
                "mae": sur.get("mae"),
                "spearman": sur.get("spearman"),
                "kendall": sur.get("kendall"),
                "top_k_precision": sur.get("top_k_precision"),
                "top_k_recall": sur.get("top_k_recall"),
                "calibration_ece": sur.get("calibration_ece"),
                "teacher_n": sur.get("teacher_n"),
            }
        )
    save_json(sur_rows, out_dir / "surrogate_quality.json")

    # Summaries per dataset/method/budget
    groups: Dict[Tuple, List[float]] = defaultdict(list)
    struct_groups: Dict[Tuple, Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))
    for r in rows:
        if r.get("test_macro_f1") is None:
            continue
        key = (r.get("dataset"), r.get("method"), float(r.get("removal_rate")))
        groups[key].append(float(r["test_macro_f1"]))
        for sk in (
            "giant_component_ratio",
            "bridge_preservation",
            "degree_ks",
            "minority_degree_retention",
            "train_seconds",
        ):
            if r.get(sk) is not None:
                struct_groups[key][sk].append(float(r[sk]))

    summary_table = []
    for key, vals in sorted(groups.items()):
        ds, method, rem = key
        s = summarize(vals)
        row = {"dataset": ds, "method": method, "removal_rate": rem, **s}
        for sk, arr in struct_groups[key].items():
            ss = summarize(arr)
            row[f"{sk}_mean"] = ss["mean"]
            row[f"{sk}_std"] = ss["std"]
        summary_table.append(row)
    save_json(summary_table, out_dir / "grid_summary.json")

    # AUC per dataset/method/seed then summarize
    auc_groups: Dict[Tuple, List[float]] = defaultdict(list)
    by_dms: Dict[Tuple, List[tuple]] = defaultdict(list)
    for r in rows:
        if r.get("test_macro_f1") is None:
            continue
        by_dms[(r.get("dataset"), r.get("method"), r.get("seed"))].append(
            (float(r["removal_rate"]), float(r["test_macro_f1"]))
        )
    for (ds, method, seed), pts in by_dms.items():
        auc = sparsity_auc([p[0] for p in pts], [p[1] for p in pts])
        auc_groups[(ds, method)].append(auc)
    auc_table = []
    for (ds, method), vals in sorted(auc_groups.items()):
        auc_table.append({"dataset": ds, "method": method, "metric": "sparsity_f1_auc", **summarize(vals)})
    save_json(auc_table, out_dir / "sparsity_auc.json")

    # Paired tests: CAILP multi vs each baseline at each budget, Holm across comparisons
    comparisons = []
    pvals = []
    datasets = sorted({r.get("dataset") for r in rows if r.get("dataset")})
    for ds in datasets:
        for rem in sorted({float(r["removal_rate"]) for r in rows if r.get("dataset") == ds}):
            multi = {}
            others = defaultdict(dict)
            for r in rows:
                if r.get("dataset") != ds or float(r.get("removal_rate")) != rem:
                    continue
                if r.get("method") == "cailp_multi":
                    multi[r["seed"]] = r["test_macro_f1"]
                else:
                    others[r["method"]][r["seed"]] = r["test_macro_f1"]
            for method, seed_map in others.items():
                common = sorted(set(multi) & set(seed_map))
                if len(common) < 2:
                    continue
                a = [multi[s] for s in common]
                b = [seed_map[s] for s in common]
                stats = paired_compare(a, b)
                comparisons.append(
                    {
                        "dataset": ds,
                        "removal_rate": rem,
                        "method_a": "cailp_multi",
                        "method_b": method,
                        "n_seeds": len(common),
                        "mean_a": float(np.mean(a)),
                        "mean_b": float(np.mean(b)),
                        **stats,
                    }
                )
                pvals.append(stats.get("wilcoxon_p", float("nan")))

    if pvals:
        # Replace nan with 1 for correction stability
        clean = [1.0 if (p is None or np.isnan(p)) else p for p in pvals]
        adj = holm_correct(clean)
        for c, p_adj in zip(comparisons, adj):
            c["wilcoxon_p_holm"] = p_adj
    save_json(comparisons, out_dir / "paired_tests.json")

    # RQ11: multi vs A31
    rq11 = []
    rq11_decision = {"retain_six_component_as_headline": False, "reason": "", "evidence": []}
    datasets_win = set()
    for ds in datasets:
        dims_win = 0
        # Macro-F1 across budgets: compare mean AUC and per-budget wins
        multi_auc = auc_groups.get((ds, "cailp_multi"), [])
        a31_auc = auc_groups.get((ds, "cailp_a31"), [])
        entry = {"dataset": ds}
        if len(multi_auc) >= 2 and len(a31_auc) >= 2:
            # align by taking available seeds summarized
            entry["auc_multi"] = summarize(multi_auc)
            entry["auc_a31"] = summarize(a31_auc)
            # paired if same length seeds - approximate via means per seed from by_dms
            multi_seed_auc = {}
            a31_seed_auc = {}
            for (d, m, s), pts in by_dms.items():
                if d != ds:
                    continue
                auc = sparsity_auc([p[0] for p in pts], [p[1] for p in pts])
                if m == "cailp_multi":
                    multi_seed_auc[s] = auc
                if m == "cailp_a31":
                    a31_seed_auc[s] = auc
            common = sorted(set(multi_seed_auc) & set(a31_seed_auc))
            if len(common) >= 2:
                pc = paired_compare([multi_seed_auc[s] for s in common], [a31_seed_auc[s] for s in common])
                entry["auc_paired"] = pc
                if pc.get("effect_size_cohens_d", 0) > 0 and pc.get("wilcoxon_p", 1) < 0.05:
                    dims_win += 1
        # Structural dimensions at 50% removal — seed-aligned pairing
        def collect_by_seed(method, metric, rem=0.5):
            out = {}
            for r in rows:
                if (
                    r.get("dataset") == ds
                    and r.get("method") == method
                    and float(r.get("removal_rate")) == rem
                    and r.get(metric) is not None
                    and r.get("seed") is not None
                ):
                    out[r["seed"]] = float(r[metric])
            return out

        for metric in ("giant_component_ratio", "bridge_preservation", "minority_degree_retention"):
            a_map = collect_by_seed("cailp_multi", metric)
            b_map = collect_by_seed("cailp_a31", metric)
            common_s = sorted(set(a_map) & set(b_map))
            if len(common_s) >= 2:
                pc = paired_compare([a_map[s] for s in common_s], [b_map[s] for s in common_s])
                entry[f"{metric}_paired"] = pc
                if pc.get("effect_size_cohens_d", 0) > 0 and pc.get("wilcoxon_p", 1) < 0.05:
                    dims_win += 1
        entry["dims_win"] = dims_win
        if dims_win >= 2:
            datasets_win.add(ds)
        rq11.append(entry)

    if len(datasets_win) >= 2:
        rq11_decision = {
            "retain_multi_criteria_as_headline": True,
            "reason": (
                f"CILP outperformed Task-only on >=2 dimensions for datasets {sorted(datasets_win)} "
                "(paired Wilcoxon p<0.05 and positive Cohen d; fusion fixed)."
            ),
            "evidence": rq11,
        }
    else:
        rq11_decision = {
            "retain_multi_criteria_as_headline": False,
            "reason": (
                "Decision rule not met: need consistent outperformance vs Task-only on >=2 datasets "
                "and multiple evaluation dimensions. Report mixed/narrowed claim."
            ),
            "datasets_with_multi_dim_win": sorted(datasets_win),
            "evidence": rq11,
        }
    save_json({"multi_dim_entries": rq11, "decision": rq11_decision}, out_dir / "multi_dim_decision.json")
    # Keep legacy filename for compatibility
    save_json({"rq11_entries": rq11, "decision": rq11_decision}, out_dir / "rq11_decision.json")

    # Adversarial rejection note
    adv_note = {
        "included_in_core": False,
        "pilot_lastfm_seed0_removal_0.5_macro_f1": 0.6658,
        "core_cailp_same_setting_macro_f1": 0.8290,
        "status": "rejected_hypothesis_unless_overturned_by_multi_seed",
    }
    save_json(adv_note, out_dir / "adversarial_rejection.json")

    # Figures
    try:
        import matplotlib.pyplot as plt

        for ds in sorted({r["dataset"] for r in summary_table if r.get("dataset")}):
            plt.figure(figsize=(8, 5))
            methods = sorted({r["method"] for r in summary_table if r["dataset"] == ds})
            for method in methods:
                pts = [
                    (r["removal_rate"], r["mean"], r["ci95_low"], r["ci95_high"])
                    for r in summary_table
                    if r["dataset"] == ds and r["method"] == method
                ]
                pts = sorted(pts)
                if not pts:
                    continue
                xs = [p[0] for p in pts]
                ys = [p[1] for p in pts]
                lo = [p[2] for p in pts]
                hi = [p[3] for p in pts]
                plt.plot(xs, ys, marker="o", label=method)
                plt.fill_between(xs, lo, hi, alpha=0.15)
            plt.xlabel("Edge removal rate")
            plt.ylabel("Test Macro-F1 (mean ± 95% CI)")
            plt.title(f"Sparsity–performance: {ds}")
            plt.legend(fontsize=7)
            plt.tight_layout()
            plt.savefig(fig_dir / f"sparsity_f1_{ds}.png", dpi=200)
            plt.close()

        # Pareto: Macro-F1 vs giant component at 0.5, mean over seeds
        plt.figure(figsize=(7, 5))
        for ds in sorted({r["dataset"] for r in summary_table}):
            for r in summary_table:
                if r["dataset"] != ds or float(r["removal_rate"]) != 0.5:
                    continue
                x = r.get("giant_component_ratio_mean", np.nan)
                y = r["mean"]
                if np.isnan(x):
                    continue
                plt.scatter(x, y, label=f"{ds}:{r['method']}", s=40)
        plt.xlabel("Giant-component ratio (mean)")
        plt.ylabel("Test Macro-F1 (mean)")
        plt.title("Pareto snapshot @ 50% removal")
        plt.legend(fontsize=6, loc="best")
        plt.tight_layout()
        plt.savefig(fig_dir / "pareto_f1_vs_giant_0.5.png", dpi=200)
        plt.close()
        print("Wrote figures to", fig_dir)
    except Exception as e:
        print("Figure generation issue:", e)

    # Markdown report (no abstract/novelty)
    lines = [
        "# Multi-seed Grid Analysis",
        "",
        f"Rows loaded: {len(rows)}",
        "",
        "## RQ11 decision",
        "",
        f"- retain_six_component_as_headline: **{rq11_decision['retain_six_component_as_headline']}**",
        f"- reason: {rq11_decision['reason']}",
        "",
        "## Adversarial module",
        "",
        "Excluded from core method (pilot negative; not overturned here).",
        "",
        "## Artifacts",
        "",
        "- `results/processed/grid_summary.json`",
        "- `results/processed/paired_tests.json`",
        "- `results/processed/sparsity_auc.json`",
        "- `results/processed/surrogate_quality.json`",
        "- `results/processed/rq11_decision.json`",
        "",
        "Abstract / novelty / contributions remain deferred.",
    ]
    (ROOT / "docs" / "progress" / "grid_analysis.md").write_text("\n".join(lines), encoding="utf-8")
    print("RQ11 retain headline:", rq11_decision["retain_six_component_as_headline"])
    print("Wrote docs/progress/grid_analysis.md")


if __name__ == "__main__":
    main()
