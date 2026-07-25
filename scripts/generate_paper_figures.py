#!/usr/bin/env python3
"""Generate paper figures from authoritative_results.csv only."""
from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
AUTH = ROOT / "results" / "processed" / "authoritative_results.csv"
FIG = ROOT / "paper" / "figures"
PROV = ROOT / "docs" / "figure_data_provenance.md"

ORDER = [
    "cailp_social_multi",
    "cailp_a31",
    "original_ilp_gcn",
    "ptdnet",
    "neuralsparse",
    "random",
    "resistance_style_proxy",
]
LABELS = {
    "cailp_social_multi": "CILP",
    "cailp_a31": "Task-only",
    "original_ilp_gcn": "ILP-GCN",
    "ptdnet": "PTDNet",
    "neuralsparse": "NeuralSparse",
    "random": "Random",
    "resistance_style_proxy": "Resistance proxy",
}
SHORT = {
    "cailp_social_multi": "CILP",
    "cailp_a31": "Task-only",
    "original_ilp_gcn": "ILP-GCN",
    "ptdnet": "PTDNet",
    "neuralsparse": "NeuralSparse",
    "random": "Random",
    "resistance_style_proxy": "Resistance proxy",
}
COLORS = {
    "cailp_social_multi": "#0072B2",
    "cailp_a31": "#56B4E9",
    "original_ilp_gcn": "#009E73",
    "ptdnet": "#D55E00",
    "neuralsparse": "#CC79A7",
    "random": "#999999",
    "resistance_style_proxy": "#E69F00",
}


def load():
    return list(csv.DictReader(AUTH.open()))


def mean_ci(vals):
    a = np.asarray(vals, float)
    n = len(a)
    if n == 0:
        return np.nan, np.nan, np.nan, 0
    m = float(a.mean())
    sd = float(a.std(ddof=1)) if n > 1 else 0.0
    if n >= 2:
        tcrit = float(stats.t.ppf(0.975, n - 1))
        half = tcrit * sd / np.sqrt(n)
    else:
        half = 0.0
    return m, m - half, m + half, n


def auc(pts):
    pts = sorted(pts)
    xs = np.array([p[0] for p in pts])
    ys = np.array([p[1] for p in pts])
    return float(np.trapz(ys, xs))


def style():
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 8,
            "axes.labelsize": 9,
            "pdf.fonttype": 42,
        }
    )


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    rows = load()
    style()
    provenance = ["# Figure data provenance\n", f"Source: `{AUTH.relative_to(ROOT)}`\n", "CI: mean ± t₀.₉₇₅,ₙ₋₁ × SEM over seeds.\n"]

    DATASETS = [
        ("lastfm", "LastFM Asia"),
        ("facebook", "Facebook Page-Page"),
        ("github", "GitHub Developers"),
    ]
    # keep only datasets with authoritative rows
    DATASETS = [(ds, title) for ds, title in DATASETS if any(r["dataset"] == ds for r in rows)]

    # Fig 2
    fig, axes = plt.subplots(1, len(DATASETS), figsize=(3.7 * len(DATASETS), 3.2), squeeze=False)
    axes = axes[0]
    for ax, (ds, title) in zip(axes, DATASETS):
        for m in ORDER:
            xs, ys, lo, hi = [], [], [], []
            for rem in [round(0.1 * i, 1) for i in range(1, 10)]:
                vals = [
                    float(r["test_macro_f1"])
                    for r in rows
                    if r["dataset"] == ds and r["method"] == m and abs(float(r["edge_removal_rate"]) - rem) < 1e-9
                ]
                mu, l, h, n = mean_ci(vals)
                if n == 0:
                    continue
                xs.append(rem)
                ys.append(mu)
                lo.append(l)
                hi.append(h)
            ax.plot(xs, ys, color=COLORS[m], lw=1.5, label=LABELS[m])
            ax.fill_between(xs, lo, hi, color=COLORS[m], alpha=0.15, lw=0)
        seeds = sorted({int(r["seed"]) for r in rows if r["dataset"] == ds and r["method"] == "cailp_social_multi"})
        ax.set_xlabel("Edge-removal rate")
        ax.set_ylabel("Macro-F1")
        ax.set_title(f"{title} (n={len(seeds)} seeds)")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="center left", bbox_to_anchor=(1.01, 0.5), frameon=False, fontsize=7)
    fig.tight_layout()
    fig.savefig(FIG / "fig2_sparsity_curves.pdf", bbox_inches="tight")
    fig.savefig(FIG / "fig2_sparsity_curves.png", dpi=300, bbox_inches="tight")
    plt.close()
    provenance.append(
        "\n## Fig. 2\n- Filter: all methods, budgets 0.1–0.9\n"
        f"- Datasets: {[d for d,_ in DATASETS]}\n- File: fig2_sparsity_curves.pdf\n"
    )

    # Fig 3
    fig, axes = plt.subplots(1, len(DATASETS), figsize=(3.7 * len(DATASETS), 3.3), squeeze=False)
    axes = axes[0]
    for ax, (ds, title) in zip(axes, DATASETS):
        means, errs, labs, cols = [], [], [], []
        for m in ORDER:
            by = defaultdict(list)
            for r in rows:
                if r["dataset"] == ds and r["method"] == m:
                    by[int(r["seed"])].append((float(r["edge_removal_rate"]), float(r["test_macro_f1"])))
            s = mean_ci([auc(v) for v in by.values()])
            if s[3] == 0:
                continue
            means.append(s[0])
            errs.append(s[2] - s[0])
            labs.append(LABELS[m])
            cols.append(COLORS[m])
        y = np.arange(len(means))
        ax.barh(y, means, xerr=errs, color=cols, height=0.72, error_kw={"lw": 0.8, "capsize": 2})
        ax.set_yticks(y)
        ax.set_yticklabels(labs)
        ax.set_xlabel("Sparsity–Macro-F1 AUC")
        ax.set_title(f"{title} (n=10)")
        ax.invert_yaxis()
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(FIG / "fig3_method_comparison.pdf", bbox_inches="tight")
    fig.savefig(FIG / "fig3_method_comparison.png", dpi=300, bbox_inches="tight")
    plt.close()
    provenance.append(
        "\n## Fig. 3\n- Metric: per-seed sparsity–Macro-F1 AUC then mean±t·SEM\n"
        f"- Datasets: {[d for d,_ in DATASETS]}\n- File: fig3_method_comparison.pdf\n"
    )

    # Fig 4 paired common seeds only
    fig, axes = plt.subplots(1, len(DATASETS), figsize=(3.7 * len(DATASETS), 3.0), squeeze=False)
    axes = axes[0]
    for ax, (ds, title) in zip(axes, DATASETS):
        seeds_m = {int(r["seed"]) for r in rows if r["dataset"] == ds and r["method"] == "cailp_social_multi"}
        seeds_a = {int(r["seed"]) for r in rows if r["dataset"] == ds and r["method"] == "cailp_a31"}
        common = sorted(seeds_m & seeds_a)
        for m, ls in [("cailp_social_multi", "-"), ("cailp_a31", "--")]:
            xs, ys, lo, hi = [], [], [], []
            for rem in [round(0.1 * i, 1) for i in range(1, 10)]:
                vals = [
                    float(r["test_macro_f1"])
                    for r in rows
                    if r["dataset"] == ds
                    and r["method"] == m
                    and int(r["seed"]) in common
                    and abs(float(r["edge_removal_rate"]) - rem) < 1e-9
                ]
                mu, l, h, n = mean_ci(vals)
                xs.append(rem)
                ys.append(mu)
                lo.append(l)
                hi.append(h)
            ax.plot(xs, ys, ls=ls, color=COLORS[m], lw=1.7, label=LABELS[m])
            ax.fill_between(xs, lo, hi, color=COLORS[m], alpha=0.15, lw=0)
        ax.set_xlabel("Edge-removal rate")
        ax.set_ylabel("Macro-F1")
        ax.set_title(f"{title} | paired n={len(common)}")
        ax.legend(frameon=False, fontsize=7)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(FIG / "fig4_teacher_compare.pdf", bbox_inches="tight")
    fig.savefig(FIG / "fig4_teacher_compare.png", dpi=300, bbox_inches="tight")
    plt.close()
    provenance.append(
        "\n## Fig. 4\n- Paired common seeds only for CILP vs Task-only\n"
        f"- Datasets: {[d for d,_ in DATASETS]}\n- File: fig4_teacher_compare.pdf\n"
    )

    # Fig 5
    metrics = [
        ("giant_component_ratio", "GC ratio"),
        ("bridge_preservation", "Bridge ret."),
        ("minority_degree_retention", "Min.-deg. ret."),
    ]
    methods = ["cailp_social_multi", "cailp_a31", "original_ilp_gcn", "ptdnet", "random"]
    n_ds = len(DATASETS)
    fig = plt.figure(figsize=(3.2 * n_ds + 0.6, 3.1))
    gs = GridSpec(1, n_ds + 1, width_ratios=[1] * n_ds + [0.06], wspace=0.45)
    axes = [fig.add_subplot(gs[0, i]) for i in range(n_ds)]
    cax = fig.add_subplot(gs[0, n_ds])
    im = None
    for ax, (ds, title) in zip(axes, DATASETS):
        data = np.full((len(methods), len(metrics)), np.nan)
        for i, m in enumerate(methods):
            for j, (k, _) in enumerate(metrics):
                vals = [
                    float(r[k])
                    for r in rows
                    if r["dataset"] == ds
                    and r["method"] == m
                    and abs(float(r["edge_removal_rate"]) - 0.5) < 1e-9
                    and r.get(k) not in (None, "")
                ]
                if vals:
                    data[i, j] = np.mean(vals)
        im = ax.imshow(data, cmap="viridis", aspect="auto", vmin=0.35, vmax=1.0)
        ax.set_xticks(range(3))
        ax.set_xticklabels([t for _, t in metrics], rotation=18, ha="right", fontsize=7)
        ax.set_yticks(range(len(methods)))
        ax.set_yticklabels([SHORT[m] for m in methods], fontsize=7)
        ax.set_title(f"{title} @50%")
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                if np.isfinite(data[i, j]):
                    ax.text(
                        j,
                        i,
                        f"{data[i, j]:.2f}",
                        ha="center",
                        va="center",
                        color="white" if data[i, j] < 0.7 else "black",
                        fontsize=7,
                    )
    fig.colorbar(im, cax=cax, label="Metric")
    fig.savefig(FIG / "fig5_structure.pdf", bbox_inches="tight")
    fig.savefig(FIG / "fig5_structure.png", dpi=300, bbox_inches="tight")
    plt.close()
    provenance.append(
        "\n## Fig. 5\n- Budget 0.5 only; mean over seeds 0–9\n"
        f"- Datasets: {[d for d,_ in DATASETS]}\n- File: fig5_structure.pdf\n"
    )

    # Fig 6 ablations from ablation json (separate)
    GRID = ROOT / "results" / "raw" / "grid"
    abl = []
    for p in GRID.glob("ablation_lastfm_*.json"):
        import json

        d = json.loads(p.read_text())
        abl.extend(d if isinstance(d, list) else [d])
    groups = [
        ("Fusion", ["A8_concat", "A9_gated", "A10_cross_attn"]),
        ("Priors", ["A19_no_mass", "A24_pagerank_prior", "A32_black_hole_mass"]),
        ("Teacher", ["A8_concat", "A31_single_obj_concat"]),
        ("Context", ["A8_concat", "A2_original_ilp"]),
    ]
    panel_labels = {
        "Fusion": {"A8_concat": "Concatenation", "A9_gated": "Gated", "A10_cross_attn": "Cross-attn"},
        "Priors": {"A19_no_mass": "No prior", "A24_pagerank_prior": "PageRank", "A32_black_hole_mass": "Black Hole"},
        "Teacher": {"A8_concat": "Multi-criteria", "A31_single_obj_concat": "Task-only"},
        "Context": {"A8_concat": "CILP", "A2_original_ilp": "ILP-GCN"},
    }
    fig, axes = plt.subplots(1, 4, figsize=(8.2, 3.0), sharey=True)
    rem_colors = [(0.3, "#0072B2"), (0.5, "#D55E00"), (0.7, "#009E73")]
    legend_handles = []
    for ax, (gname, names) in zip(axes, groups):
        x = np.arange(len(names))
        width = 0.25
        for k, (rem, color) in enumerate(rem_colors):
            means, errs = [], []
            for name in names:
                vals = [
                    r["test_macro_f1"]
                    for r in abl
                    if r.get("ablation") == name and abs(float(r.get("removal_rate", -1)) - rem) < 1e-9
                ]
                mu, l, h, n = mean_ci(vals)
                means.append(mu)
                errs.append((h - mu) if n else 0)
            bars = ax.bar(
                x + k * width,
                means,
                width,
                yerr=errs,
                color=color,
                error_kw={"lw": 0.6, "capsize": 1.5},
            )
            if gname == "Fusion":
                legend_handles.append((bars, f"{int(rem*100)}%"))
        ax.set_xticks(x + width)
        ax.set_xticklabels([panel_labels[gname][n] for n in names], rotation=20, ha="right", fontsize=7.5)
        ax.set_title(gname, fontsize=9)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(axis="x", pad=2)
    axes[0].set_ylabel("Macro-F1")
    fig.legend(
        [h[0] for h, _ in legend_handles],
        [lab for _, lab in legend_handles],
        loc="upper center",
        ncol=3,
        frameon=False,
        fontsize=8,
        bbox_to_anchor=(0.5, 1.08),
    )
    fig.suptitle("LastFM ablations (n=5 seeds; Fusion/Priors use multi-criteria teacher)", fontsize=9, y=1.14)
    fig.tight_layout()
    fig.savefig(FIG / "fig6_ablations.pdf", bbox_inches="tight")
    fig.savefig(FIG / "fig6_ablations.png", dpi=300, bbox_inches="tight")
    plt.close()
    provenance.append(
        "\n## Fig. 6\n- Source: ablation_lastfm_*.json seeds 0–4 (n=5)\n"
        "- Fusion: Concatenation / Gated / Cross-attn (multi-criteria teacher)\n"
        "- Teacher: Multi-criteria vs Task-only (concatenation)\n"
        "- Context: CILP vs ILP-GCN (not a matched ablation)\n"
        "- File: fig6_ablations.pdf\n"
    )

    # Teacher-size sensitivity figure (validation only)
    import json as _json

    diag = _json.loads((ROOT / "results" / "raw" / "grid" / "facebook_diagnostic_seed0.json").read_text())
    by_n = defaultdict(dict)
    for r in diag:
        n = int(str(r["method"]).split("teacher")[-1])
        by_n[n][float(r["removal_rate"])] = float(r["val_macro_f1"])
    ns = [20, 40, 80, 120]
    fig, ax = plt.subplots(figsize=(4.2, 3.0))
    for rem, color, ls in [(0.3, "#0072B2", "-"), (0.5, "#D55E00", "--"), (0.7, "#009E73", "-.")]:
        ax.plot(ns, [by_n[n][rem] for n in ns], color=color, ls=ls, marker="o", label=f"Val @{int(rem*100)}%")
    means = [np.mean([by_n[n][r] for r in (0.3, 0.5, 0.7)]) for n in ns]
    ax.plot(ns, means, color="black", lw=1.8, marker="s", label="Mean val")
    ax.axvline(120, color="gray", ls=":", lw=1)
    ax.set_xlabel("Teacher size n")
    ax.set_ylabel("Validation Macro-F1")
    ax.set_xticks(ns)
    ax.legend(frameon=False, fontsize=7)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_title("Facebook seed 0 teacher-size sensitivity")
    fig.tight_layout()
    fig.savefig(FIG / "fig_s8_teacher_size.pdf", bbox_inches="tight")
    fig.savefig(FIG / "fig_s8_teacher_size.png", dpi=300, bbox_inches="tight")
    plt.close()
    provenance.append(
        "\n## SI teacher-size figure\n- Source: facebook_diagnostic_seed0.json\n"
        "- Selection rule in code: max validation Macro-F1 at 50% removal\n"
        "- File: fig_s8_teacher_size.pdf\n"
    )

    # Fig 7
    fig, axes = plt.subplots(1, len(DATASETS), figsize=(3.7 * len(DATASETS), 3.1), squeeze=False)
    axes = axes[0]
    rt_m = ["random", "original_ilp_gcn", "ptdnet", "neuralsparse", "cailp_a31", "cailp_social_multi"]
    for ax, (ds, title) in zip(axes, DATASETS):
        means, errs, labs, cols = [], [], [], []
        for m in rt_m:
            by = {}
            for r in rows:
                if r["dataset"] == ds and r["method"] == m and r.get("train_seconds") not in (None, ""):
                    by[int(r["seed"])] = float(r["train_seconds"])
            mu, l, h, n = mean_ci(list(by.values()))
            if n == 0:
                continue
            means.append(mu)
            errs.append(h - mu)
            labs.append(LABELS[m])
            cols.append(COLORS[m])
        y = np.arange(len(means))
        ax.barh(y, means, xerr=errs, color=cols, height=0.7, error_kw={"lw": 0.8, "capsize": 2})
        ax.set_yticks(y)
        ax.set_yticklabels(labs, fontsize=7)
        ax.set_xlabel("train_seconds (includes teacher for CILP)")
        ax.set_title(f"{title} (CPU)")
        ax.set_xscale("log")
        ax.invert_yaxis()
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(FIG / "fig7_runtime.pdf", bbox_inches="tight")
    fig.savefig(FIG / "fig7_runtime.png", dpi=300, bbox_inches="tight")
    plt.close()
    provenance.append(
        "\n## Fig. 7\n- Field: train_seconds from authoritative CSV "
        "(teacher+scorer for CILP/Task-only; not separable)\n"
        f"- Datasets: {[d for d,_ in DATASETS]}\n- File: fig7_runtime.pdf\n"
    )

    PROV.write_text("\n".join(provenance))
    print("figures + provenance written")


if __name__ == "__main__":
    main()
