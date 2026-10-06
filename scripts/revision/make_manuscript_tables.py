#!/usr/bin/env python3
"""Generate REVISED-protocol LaTeX tables and figures for the manuscript from results/revised.

Every number written here is computed from completed REVISED result files (or, where labelled ORIGINAL,
copied verbatim from the submitted manuscript tables). Cells whose runs are not complete are written as
"pending" and listed in results/revised/tables/PENDING.txt.
"""
from __future__ import annotations

import glob
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import stats  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.revision.analyze import (BASELINES, DISPLAY, auc_table, load_final,  # noqa: E402
                                      primary_rows)
from src.revision.common import BUDGETS, REVISED_DIR, teacher_variants  # noqa: E402
from src.revision.stats import holm, paired  # noqa: E402

MS = ROOT / "manuscript" / "minimal"
TAB, FIG = MS / "tables", MS / "figures"
DS = ["lastfm", "facebook", "github", "pubmed"]
DSN = {"lastfm": "LastFM", "facebook": "Facebook", "github": "GitHub", "pubmed": "PubMed"}
PENDING: list[str] = []

# ORIGINAL (submitted) CILP - Task-only statistics, verbatim from manuscript/source/tables/s21_paired_multidim.tex
ORIG = {
    ("lastfm", "auc"): ("+0.0265", "0.007812"), ("lastfm", "giant_component_ratio"): ("+0.1564", "0.007812"),
    ("lastfm", "bridge_retention"): ("+0.2230", "0.007812"), ("lastfm", "minority_degree_retention"): ("+0.2182", "0.01151"),
    ("facebook", "auc"): ("+0.0142", "0.01562"), ("facebook", "giant_component_ratio"): ("+0.2971", "0.02306"),
    ("facebook", "bridge_retention"): ("+0.4053", "0.02306"), ("facebook", "minority_degree_retention"): ("+0.1569", "0.02306"),
    ("github", "auc"): ("+0.0033", "0.007812"), ("github", "giant_component_ratio"): ("+0.2284", "0.007812"),
    ("github", "bridge_retention"): ("+0.3195", "0.007812"), ("github", "minority_degree_retention"): ("+0.1136", "0.007812"),
}
MET = [("auc", "AUC"), ("giant_component_ratio", "GC@50\\%"), ("bridge_retention", "Bridge@50\\%"),
       ("minority_degree_retention", "Min-deg@50\\%")]
COLORS = {"cilp_full": "#0072B2", "task_only": "#56B4E9", "original_ilp": "#009E73", "ptdnet": "#D55E00",
          "random": "#999999", "nk_local_degree": "#000000", "nk_local_similarity": "#CC79A7", "full_graph": "#444444"}


def f(x, d=3, sign=False):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "--"
    return (f"{x:+.{d}f}" if sign else f"{x:.{d}f}")


def fp(p):
    return "--" if p is None or np.isnan(p) else (f"{p:.4f}" if p >= 0.0001 else "$<$0.0001")


def seedvec(df, ds, m, col):
    return df[(df.dataset == ds) & (df.method == m)].set_index("seed")[col].sort_index()


def complete(ds):
    exp = {"pubmed": 540}.get(ds, 560)
    return len(glob.glob(str(REVISED_DIR / "final" / ds / "seed*" / "*.json"))) >= exp


def main():
    TAB.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)
    lock = json.loads((ROOT / "configs/revised/locked.json").read_text())
    df_all = load_final()
    df = primary_rows(df_all, lock)
    auc = auc_table(df)
    fam = pd.read_csv(REVISED_DIR / "tables" / "family_P.csv")
    r05 = df[np.isclose(df.removal_rate, 0.5)]
    full = df[df.removal_rate == 0]

    # ---------------- moderator and learnability
    mech = []
    for ds in DS:
        fg = seedvec(full, ds, "full_graph", "test_macro_f1")
        rd = seedvec(r05, ds, "random", "test_macro_f1")
        drop = (fg - rd[fg.index])
        p = paired(fg.values, rd[fg.index].values)
        row = {"ds": ds, "full": fg.mean(), "rand50": rd.mean(), "drop": drop.mean(), "drop_lo": p["ci_low"], "drop_hi": p["ci_high"]}
        for v in ("cilp_full", "task_only"):
            n = lock["teacher_n"][ds][v]
            vals = {"surrogate": [], "scorer": []}
            for s in range(10):
                r = json.loads((REVISED_DIR / "final" / ds / f"seed{s}" / f"{v}_n{n}_REVISED.json").read_text())
                for k in vals:
                    vals[k].append(r["agreement"][f"holdout_{k}_spearman"])
            for k in vals:
                a = np.array(vals[k], float)
                row[f"{v}_{k}"] = np.nanmean(a)
                row[f"{v}_{k}_sd"] = np.nanstd(a, ddof=1)
        a, b = seedvec(auc, ds, "cilp_full", "auc"), seedvec(auc, ds, "task_only", "auc")
        row["dauc"] = (a - b[a.index]).mean()
        mech.append(row)
    mech = pd.DataFrame(mech)
    mech.to_csv(REVISED_DIR / "tables" / "mechanism_moderator.csv", index=False)
    rho = stats.spearmanr(mech["drop"], mech["dauc"]).correlation
    lines = ["\\begin{tabular}{@{}lccccc@{}}", "\\toprule",
             "Dataset & Full-graph F1 & $\\Delta_{\\mathrm{edge}}$ [95\\% CI] & $\\rho_{\\mathrm{ho}}$ CILP & $\\rho_{\\mathrm{ho}}$ Task-only & $\\Delta$AUC (CILP$-$Task-only) \\\\", "\\midrule"]
    for _, r in mech.iterrows():
        lines.append(f"{DSN[r.ds]} & {f(r.full)} & {f(r['drop'])} [{f(r.drop_lo)}, {f(r.drop_hi)}] & "
                     f"{f(r.cilp_full_surrogate,2)} $\\pm$ {f(r.cilp_full_surrogate_sd,2)} & "
                     f"{f(r.task_only_surrogate,2)} $\\pm$ {f(r.task_only_surrogate_sd,2)} & {f(r.dauc,4,True)} \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    (TAB / "rev_mechanism.tex").write_text("\n".join(lines) + "\n")
    (REVISED_DIR / "tables" / "moderator_rank_corr.txt").write_text(f"Spearman(drop, dAUC) over 4 datasets = {rho:.3f}\n")

    # ---------------- CILP vs Task-only (Family P part) with ORIGINAL column
    lines = ["\\begin{longtable}{@{}llrrrcrrrcc@{}}",
             "\\caption{\\textbf{CILP versus Task-only (REVISED protocol)}, paired seeds $0$--$9$. $p$: two-sided exact Wilcoxon; Holm: within each dataset's $14$-hypothesis family. ORIGINAL: submitted values (development-set protocol), shown for transparency.}\\label{tab:teacher}\\\\",
             "\\toprule",
             "Dataset & Metric & CILP & Task-only & $\\Delta$ & 95\\% CI & $p$ & Holm $p$ & $d_z$ & W/T/L & ORIGINAL $\\Delta$ (Holm $p$) \\\\", "\\midrule", "\\endfirsthead",
             "\\toprule", "Dataset & Metric & CILP & Task-only & $\\Delta$ & 95\\% CI & $p$ & Holm $p$ & $d_z$ & W/T/L & ORIGINAL \\\\", "\\midrule", "\\endhead"]
    for ds in DS:
        for key, lab in MET:
            comp = "CILP - Task-only: AUC" if key == "auc" else f"CILP - Task-only: {key}@0.5"
            r = fam[(fam.dataset == ds) & (fam.comparison == comp)].iloc[0]
            if key == "auc":
                a, b = seedvec(auc, ds, "cilp_full", "auc").mean(), seedvec(auc, ds, "task_only", "auc").mean()
            else:
                a, b = seedvec(r05, ds, "cilp_full", key).mean(), seedvec(r05, ds, "task_only", key).mean()
            o = ORIG.get((ds, key))
            orig = f"{o[0]} ({o[1]})" if o else "n/a"
            lines.append(f"{DSN[ds]} & {lab} & {f(a)} & {f(b)} & {f(r.mean_diff,4,True)} & [{f(r.ci_low,4,True)}, {f(r.ci_high,4,True)}] & "
                         f"{fp(r.p_raw)} & {fp(r.p_holm)} & {f(r.d_z,2)} & {int(r.wins)}/{int(r.ties)}/{int(r.losses)} & {orig} \\\\")
        lines.append("\\midrule" if ds != DS[-1] else "\\bottomrule")
    lines.append("\\end{longtable}")
    (TAB / "rev_teacher.tex").write_text("\n".join(lines) + "\n")

    # ---------------- CILP vs every baseline (AUC), Delta and Holm p
    lines = ["\\begin{tabular}{@{}l" + "cc" * len(DS) + "@{}}", "\\toprule",
             "Comparator & " + " & ".join(f"\\multicolumn{{2}}{{c}}{{{DSN[d]}}}" for d in DS) + " \\\\",
             " & " + " & ".join("$\\Delta$AUC & Holm $p$" for _ in DS) + " \\\\", "\\midrule"]
    for b in BASELINES:
        cells = []
        for ds in DS:
            r = fam[(fam.dataset == ds) & (fam.comparison == f"CILP - {DISPLAY[b]}: AUC")].iloc[0]
            cells.append(f"{f(r.mean_diff,4,True)} & {fp(r.p_holm)}")
        lines.append(f"{DISPLAY[b]} & " + " & ".join(cells) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    (TAB / "rev_baselines_auc.tex").write_text("\n".join(lines) + "\n")

    # ---------------- main predictive table
    order = ["cilp_full", "task_only", "original_ilp", "ptdnet", "neuralsparse", "resistance_proxy", "gassip_adapted",
             "mog_adapted", "psgnn_reimpl", "nk_local_degree", "nk_local_similarity", "random"]
    lines = ["\\begin{longtable}{@{}llcccc@{}}",
             "\\caption{\\textbf{Predictive results (REVISED protocol)}: test Macro-F1 (mean$\\pm$s.d., $n{=}10$ seeds) at $30/50/70\\%$ removal and sparsity--Macro-F1 AUC. Full graph: $r{=}0$ reference. Bold: best AUC per dataset.}\\label{tab:main}\\\\",
             "\\toprule", "Dataset & Method & 30\\% & 50\\% & 70\\% & AUC \\\\", "\\midrule", "\\endfirsthead",
             "\\toprule", "Dataset & Method & 30\\% & 50\\% & 70\\% & AUC \\\\", "\\midrule", "\\endhead"]
    for ds in DS:
        fg = seedvec(full, ds, "full_graph", "test_macro_f1")
        lines.append(f"{DSN[ds]} & Full graph ($r{{=}}0$) & \\multicolumn{{3}}{{c}}{{{f(fg.mean())}$\\pm${f(fg.std(ddof=1))}}} & -- \\\\")
        best = max(order, key=lambda m: seedvec(auc, ds, m, "auc").mean())
        for m in order:
            cells = []
            for r in (0.3, 0.5, 0.7):
                v = seedvec(df[np.isclose(df.removal_rate, r)], ds, m, "test_macro_f1")
                cells.append(f"{f(v.mean())}$\\pm${f(v.std(ddof=1))}")
            a = seedvec(auc, ds, m, "auc")
            av = f"{f(a.mean())}$\\pm${f(a.std(ddof=1))}"
            av = f"\\textbf{{{av}}}" if m == best else av
            lines.append(f" & {DISPLAY[m]} & " + " & ".join(cells) + f" & {av} \\\\")
        lines.append("\\midrule" if ds != DS[-1] else "\\bottomrule")
    lines.append("\\end{longtable}")
    (TAB / "rev_main.tex").write_text("\n".join(lines) + "\n")

    # ---------------- structure + community at 50%
    sm = ["cilp_full", "task_only", "random", "original_ilp", "ptdnet", "nk_local_degree", "nk_local_similarity"]
    cols = [("giant_component_ratio", "GC"), ("bridge_retention", "Bridge"), ("minority_degree_retention", "Min-deg"),
            ("nmi_leiden", "NMI"), ("ari_leiden", "ARI"), ("conductance_ratio", "Cond.\\ ratio"), ("modularity_retention", "Mod.\\ ret.")]
    lines = ["\\begin{longtable}{@{}ll" + "c" * len(cols) + "@{}}",
             "\\caption{\\textbf{Structural and community metrics at $50\\%$ removal (REVISED protocol)}, means over $10$ seeds. Community metrics use an independent Leiden reference partition (seed~$0$).}\\label{tab:structure}\\\\",
             "\\toprule", "Dataset & Method & " + " & ".join(c for _, c in cols) + " \\\\", "\\midrule", "\\endfirsthead",
             "\\toprule", "Dataset & Method & " + " & ".join(c for _, c in cols) + " \\\\", "\\midrule", "\\endhead"]
    for ds in DS:
        for i, m in enumerate(sm):
            vals = [f(seedvec(r05, ds, m, k).mean()) for k, _ in cols]
            lines.append(f"{DSN[ds] if i == 0 else ''} & {DISPLAY[m]} & " + " & ".join(vals) + " \\\\")
        lines.append("\\midrule" if ds != DS[-1] else "\\bottomrule")
    lines.append("\\end{longtable}")
    (TAB / "rev_structure.tex").write_text("\n".join(lines) + "\n")

    # ---------------- GC and bridge at 30/50/70
    lines = ["\\begin{tabular}{@{}llcccccc@{}}", "\\toprule",
             "Dataset & Method & GC@30 & GC@50 & GC@70 & Br@30 & Br@50 & Br@70 \\\\", "\\midrule"]
    for ds in DS:
        for i, m in enumerate(sm):
            vals = [f(seedvec(df[np.isclose(df.removal_rate, r)], ds, m, k).mean())
                    for k in ("giant_component_ratio", "bridge_retention") for r in (0.3, 0.5, 0.7)]
            lines.append(f"{DSN[ds] if i == 0 else ''} & {DISPLAY[m]} & " + " & ".join(vals) + " \\\\")
        lines.append("\\midrule" if ds != DS[-1] else "\\bottomrule")
    lines.append("\\end{tabular}")
    (TAB / "rev_structure_multi.tex").write_text("\n".join(lines) + "\n")

    # ---------------- ablation summary (per dataset; pending where incomplete)
    abl = pd.read_csv(REVISED_DIR / "tables" / "ablation_weights.csv")
    named = ["task_only", "loo_task", "loo_comm", "loo_conn", "loo_spec", "loo_repr", "loo_group",
             "grp_deletion_only", "grp_task_plus_proxies", "grp_proxies_only",
             "w_balanced", "w_task_focused", "w_structure_focused", "w_reduced_proxy"]
    nice = {"task_only": "Task-only", "loo_task": "$-$task", "loo_comm": "$-$community", "loo_conn": "$-$connectivity",
            "loo_spec": "$-$spectral", "loo_repr": "$-$representation", "loo_group": "$-$group",
            "grp_deletion_only": "deletion-induced only", "grp_task_plus_proxies": "task + proxies",
            "grp_proxies_only": "proxies only", "w_balanced": "balanced", "w_task_focused": "task-focused",
            "w_structure_focused": "structure-focused", "w_reduced_proxy": "reduced proxy"}
    lines = ["\\begin{tabular}{@{}l" + "c" * len(DS) + "@{}}", "\\toprule",
             "Teacher variant & " + " & ".join(DSN[d] for d in DS) + " \\\\", "\\midrule"]
    fullauc = {ds: seedvec(auc, ds, "cilp_full", "auc").mean() for ds in DS}
    lines.append("Full CILP (AUC) & " + " & ".join(f(fullauc[d], 4) for d in DS) + " \\\\")
    lines.append("\\midrule \\multicolumn{" + str(len(DS) + 1) + "}{@{}l}{$\\Delta$AUC versus full CILP [95\\% CI]} \\\\")
    for v in named:
        cells = []
        for ds in DS:
            r = abl[(abl.dataset == ds) & (abl.variant == v)]
            n_ok = len(seedvec(auc, ds, v, "auc")) == 10
            if len(r) and n_ok:
                r = r.iloc[0]
                cells.append(f"{f(r.vs_full_mean_diff,4,True)} [{f(r.vs_full_ci_low,4,True)}, {f(r.vs_full_ci_high,4,True)}]")
            else:
                cells.append("pending")
                PENDING.append(f"ablation {v} {ds}")
        lines.append(f"{nice[v]} & " + " & ".join(cells) + " \\\\")
    dl = []
    for ds in DS:
        nd = abl[(abl.dataset == ds) & abl.variant.str.startswith("dir_")]
        if len(nd) == 20 and all(len(seedvec(auc, ds, v, "auc")) == 10 for v in nd.variant):
            to = seedvec(auc, ds, "task_only", "auc").mean()
            dl.append(f"{f(nd.auc_mean.min(),3)}--{f(nd.auc_mean.max(),3)} ({(nd.auc_mean > to).sum()}/20 $>$ Task-only)")
        else:
            dl.append("pending")
            PENDING.append(f"dirichlet {ds}")
    lines.append("\\midrule 20 Dirichlet $\\beta$ (AUC range) & " + " & ".join(dl) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    (TAB / "rev_ablation.tex").write_text("\n".join(lines) + "\n")

    # ---------------- figures
    plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
                         "font.size": 8, "axes.labelsize": 9, "pdf.fonttype": 42})
    cm = ["cilp_full", "task_only", "original_ilp", "ptdnet", "nk_local_degree", "nk_local_similarity", "random"]
    fig, axes = plt.subplots(1, 4, figsize=(14.8, 3.2), squeeze=False)
    for ax, ds in zip(axes[0], DS):
        fg = seedvec(full, ds, "full_graph", "test_macro_f1")
        ax.axhline(fg.mean(), color=COLORS["full_graph"], ls=":", lw=1.2, label="Full graph ($r$=0)")
        for m in cm:
            ys, lo, hi = [], [], []
            for r in BUDGETS:
                v = seedvec(df[np.isclose(df.removal_rate, r)], ds, m, "test_macro_f1").values
                h = stats.t.ppf(0.975, len(v) - 1) * v.std(ddof=1) / np.sqrt(len(v))
                ys.append(v.mean()); lo.append(v.mean() - h); hi.append(v.mean() + h)
            ax.plot(BUDGETS, ys, color=COLORS[m], lw=1.6 if m == "cilp_full" else 1.1, label=DISPLAY[m])
            ax.fill_between(BUDGETS, lo, hi, color=COLORS[m], alpha=0.12, lw=0)
        ax.set_title(DSN[ds]); ax.set_xlabel("Edge-removal rate $r$"); ax.grid(alpha=0.25)
    axes[0][0].set_ylabel("Test Macro-F1")
    axes[0][-1].legend(fontsize=6.5, frameon=False, loc="lower left")
    fig.savefig(FIG / "rev_fig_sparsity_curves.pdf", bbox_inches="tight"); plt.close(fig)

    fig, axes = plt.subplots(1, 4, figsize=(14.8, 3.4), squeeze=False)
    comps = ["Task-only"] + [DISPLAY[b] for b in BASELINES]
    for ax, ds in zip(axes[0], DS):
        rows = [fam[(fam.dataset == ds) & (fam.comparison == f"CILP - {c}: AUC")].iloc[0] for c in comps]
        y = np.arange(len(rows))[::-1]
        m = [r.mean_diff for r in rows]
        err = [[r.mean_diff - r.ci_low for r in rows], [r.ci_high - r.mean_diff for r in rows]]
        col = ["#0072B2" if r.p_holm < 0.05 else "#999999" for r in rows]
        ax.errorbar(m, y, xerr=err, fmt="none", ecolor="#555555", lw=0.8, capsize=2)
        ax.scatter(m, y, c=col, s=18, zorder=3)
        ax.axvline(0, color="black", lw=0.8)
        for s in (-0.008, 0.008):
            ax.axvline(s, color="#D55E00", lw=0.6, ls="--")
        ax.set_yticks(y); ax.set_yticklabels(comps if ds == DS[0] else [""] * len(comps))
        ax.set_title(DSN[ds]); ax.set_xlabel("CILP $-$ comparator AUC"); ax.grid(alpha=0.25, axis="x")
    fig.savefig(FIG / "rev_fig_forest.pdf", bbox_inches="tight"); plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.2))
    ax = axes[0]
    xs = np.arange(len(DS))
    for k, (v, lab) in enumerate((("cilp_full", "CILP"), ("task_only", "Task-only"))):
        ax.bar(xs + (k - 0.5) * 0.36, mech[f"{v}_surrogate"], 0.36, yerr=mech[f"{v}_surrogate_sd"],
               color=COLORS[v], label=lab, error_kw={"lw": 0.8, "capsize": 2})
    ax.set_xticks(xs); ax.set_xticklabels([DSN[d] for d in DS]); ax.axhline(0, color="black", lw=0.6)
    ax.set_ylabel("Held-out rank agreement $\\rho_{\\mathrm{ho}}$"); ax.legend(frameon=False); ax.set_title("(a) Learnability of the teacher target")
    ax = axes[1]
    for _, r in mech.iterrows():
        ax.scatter(r["drop"], r.dauc, color="#0072B2", s=28)
        ax.annotate(DSN[r.ds], (r["drop"], r.dauc), textcoords="offset points", xytext=(4, 3), fontsize=7)
    ax.axhline(0, color="black", lw=0.6); ax.axhline(0.008, color="#D55E00", lw=0.6, ls="--")
    ax.set_xlabel("$\\Delta_{\\mathrm{edge}}$: full-graph F1 $-$ Random F1 at $r$=0.5"); ax.set_ylabel("CILP $-$ Task-only AUC")
    ax.set_title("(b) Gain versus edge dependence"); ax.grid(alpha=0.25)
    fig.tight_layout(); fig.savefig(FIG / "rev_fig_mechanism.pdf", bbox_inches="tight"); plt.close(fig)

    from scripts.revision.make_supplement_tables import supplement
    supplement(df, auc, fam, abl, lock, PENDING)
    (REVISED_DIR / "tables" / "PENDING.txt").write_text("\n".join(PENDING) + ("\n" if PENDING else ""))
    print("tables/figures written; pending:", len(PENDING))


if __name__ == "__main__":
    main()
