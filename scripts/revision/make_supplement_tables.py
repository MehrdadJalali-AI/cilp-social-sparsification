"""Supplementary REVISED tables (called from make_manuscript_tables.py)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from scripts.revision.make_manuscript_tables import DS, DSN, TAB, f, fp, seedvec
from src.revision.common import REVISED_DIR, teacher_variants

BS = "\\"
NL = BS + BS  # LaTeX row terminator


def longtab(cols: str, head: str, rows: list, cap: str, label: str) -> str:
    out = [BS + "begin{longtable}{@{}" + cols + "@{}}",
           BS + "caption{" + cap + "}" + BS + "label{" + label + "}" + NL,
           BS + "toprule", head + " " + NL, BS + "midrule", BS + "endfirsthead",
           BS + "toprule", head + " " + NL, BS + "midrule", BS + "endhead"]
    return "\n".join(out + rows + [BS + "bottomrule", BS + "end{longtable}"]) + "\n"


def comp_label(c: str) -> str:
    return c.replace("CILP - ", "").replace("_", " ").replace("@0.5", "@50" + BS + "%")


def supplement(df, auc, fam, abl, lock, pending: list) -> None:
    # confirmatory family P (all 14 per dataset) + cross-dataset Holm
    rows = []
    for _, r in fam.iterrows():
        g = r["p_holm_global_3datasets"] if "p_holm_global_3datasets" in r else np.nan
        rows.append(" & ".join([DSN[r.dataset], comp_label(r.comparison), f(r.mean_diff, 4, True),
                                f"[{f(r.ci_low, 4, True)}, {f(r.ci_high, 4, True)}]", f(r.d_z, 2),
                                f"{int(r.wins)}/{int(r.ties)}/{int(r.losses)}", fp(r.p_raw), fp(r.p_holm), fp(g)]) + " " + NL)
    (TAB / "rev_s_familyP.tex").write_text(longtab(
        "llrcrcrrr", "Dataset & CILP versus & $" + BS + "Delta$ & 95" + BS + "% CI & $d_z$ & W/T/L & $p$ & Holm $p$ & Holm (3 datasets)",
        rows, BS + "textbf{Confirmatory family P (REVISED).} $14$ planned CILP comparisons per dataset; Holm within dataset. "
        "Last column: one Holm family over the $42$ hypotheses of LastFM, Facebook and GitHub (sensitivity).", "tab:s_familyP"))

    cf = pd.read_csv(REVISED_DIR / "tables" / "family_C_community.csv")
    rows = [" & ".join([DSN[r.dataset], comp_label(r.comparison), f(r.mean_diff, 4, True),
                        f"[{f(r.ci_low, 4, True)}, {f(r.ci_high, 4, True)}]", f"{int(r.wins)}/{int(r.ties)}/{int(r.losses)}",
                        fp(r.p_raw), fp(r.p_holm)]) + " " + NL for _, r in cf.iterrows()]
    (TAB / "rev_s_familyC.tex").write_text(longtab(
        "llrcccc", "Dataset & CILP versus & $" + BS + "Delta$ & 95" + BS + "% CI & W/T/L & $p$ & Holm $p$", rows,
        BS + "textbf{Secondary community family (REVISED)} at $50" + BS + "%$ removal; conductance ratio below $1$ means "
        "better-separated reference communities.", "tab:s_familyC"))

    sel = pd.read_csv(REVISED_DIR / "tables" / "selection_n_sensitivity.csv")
    rows = []
    for _, r in sel.iterrows():
        rows.append(" & ".join([DSN[r.dataset], "CILP" if r.teacher == "cilp_full" else "Task-only", str(int(r.n)),
                                f"{f(r['val_macro_f1_r0.5_mean'])}$" + BS + "pm$" + f(r["val_macro_f1_r0.5_sd"]),
                                f(r.holdout_surrogate_spearman_mean, 2), f(r.holdout_surrogate_kendall_mean, 2),
                                f(r.holdout_scorer_spearman_mean, 2), f(r.holdout_scorer_kendall_mean, 2),
                                f(r.insample_spearman_mean, 2)]) + " " + NL)
    locked = "; ".join(f"{DSN[d]} CILP {lock['teacher_n'][d]['cilp_full']}, Task-only {lock['teacher_n'][d]['task_only']}" for d in DS)
    (TAB / "rev_s_selection.tex").write_text(longtab(
        "llrcccccc", "Dataset & Teacher & $n$ & Val. Macro-F1 @50" + BS + "% & $" + BS + "rho_{ho}$ sur. & $" + BS + "tau_{ho}$ sur. & $"
        + BS + "rho_{ho}$ scorer & $" + BS + "tau_{ho}$ scorer & $" + BS + "rho$ in-sample", rows,
        BS + "textbf{Teacher-size selection (validation only; REVISED).} Means over seeds $0$--$9$; ho: held-out agreement on "
        "$200$ probed edges not used for fitting. Locked $n$: " + locked + ".", "tab:s_selection"))

    mn = pd.read_csv(REVISED_DIR / "tables" / "matched_n_sensitivity.csv")
    rows = []
    for ds in DS:
        for n in (20, 40, 80, 120, 240):
            c = mn[(mn.dataset == ds) & (mn.method == "cilp_full") & (mn.teacher_n == n)]
            t = mn[(mn.dataset == ds) & (mn.method == "task_only") & (mn.teacher_n == n)]
            if len(c) and len(t) and int(c.seeds.iloc[0]) == 10 and int(t.seeds.iloc[0]) == 10:
                a, b = c.auc_mean.iloc[0], t.auc_mean.iloc[0]
                rows.append(f"{DSN[ds]} & {n} & {f(a, 4)} & {f(b, 4)} & {f(a - b, 4, True)} " + NL)
            else:
                rows.append(f"{DSN[ds]} & {n} & pending & pending & pending " + NL)
                pending.append(f"matched-n {ds} n={n}")
    (TAB / "rev_s_matched_n.tex").write_text(longtab(
        "lrccc", "Dataset & $n$ & CILP AUC & Task-only AUC & Difference", rows,
        BS + "textbf{Matched teacher size (REVISED, sensitivity).} Test AUC of both teachers at every $n$, evaluated after locking.",
        "tab:s_matched"))

    rp_path = REVISED_DIR / "tables" / "auc_original_replay.csv"
    rp = pd.read_csv(rp_path) if rp_path.exists() else pd.DataFrame(columns=["dataset", "method", "mean", "count"])
    arch = {"lastfm": ("0.593", "0.566"), "facebook": ("0.739", "0.725"), "github": ("0.648", "0.645")}
    rows = []
    for ds in ("lastfm", "facebook", "github"):
        c = rp[(rp.dataset == ds) & (rp.method == "cilp_full")]
        t = rp[(rp.dataset == ds) & (rp.method == "task_only")]
        if len(c) and len(t) and int(c["count"].iloc[0]) == 10 and int(t["count"].iloc[0]) == 10:
            rep = (f(c["mean"].iloc[0]), f(t["mean"].iloc[0]))
        else:
            rep = ("pending", "pending")
            pending.append(f"replay {ds}")
        rv = (f(seedvec(auc, ds, "cilp_full", "auc").mean()), f(seedvec(auc, ds, "task_only", "auc").mean()))
        rows.append(" & ".join([DSN[ds], *arch[ds], *rep, *rv]) + " " + NL)
    (TAB / "rev_s_original_vs_revised.tex").write_text(longtab(
        "lcccccc", "Dataset & " + BS + "multicolumn{2}{c}{ORIGINAL (submitted)} & " + BS + "multicolumn{2}{c}{ORIGINAL (re-executed)} & "
        + BS + "multicolumn{2}{c}{REVISED} " + NL + " & CILP & Task-only & CILP & Task-only & CILP & Task-only", rows,
        BS + "textbf{Mean AUC under the ORIGINAL and REVISED protocols.} Submitted: Table~3 of the original submission. "
        "Re-executed: original label access, original surrogate behaviour and original teacher sizes, run in the revision pipeline. "
        "REVISED: training-label teacher, locked $n$, evaluation-mode surrogate.", "tab:s_orig_rev"))

    nd = abl[abl.variant.str.startswith("dir_")]
    tv = teacher_variants()
    rows = []
    for v in sorted(nd.variant.unique()):
        w = tv[v]
        cells = []
        for d in DS:
            x = nd[(nd.dataset == d) & (nd.variant == v)]
            cells.append(f(x.auc_mean.iloc[0], 4) if len(x) else "--")
        rows.append(" & ".join([v.replace("_", BS + "_")] + [f"{w[k]:.2f}" for k in ("task", "comm", "conn", "spec", "repr", "group")] + cells) + " " + NL)
    head = "Variant & " + " & ".join("$" + BS + "beta_{" + k + "}$" for k in ("task", "comm", "conn", "spec", "repr", "group")) + " & " + " & ".join(DSN[d] for d in DS)
    (TAB / "rev_s_dirichlet.tex").write_text(longtab(
        "l" + "c" * 6 + "c" * len(DS), head, rows,
        BS + "textbf{Dirichlet-sampled weight vectors (REVISED).} $20$ vectors from $" + BS + "mathrm{Dir}(" + BS + "mathbf{1})$ "
        "(seed 20261004); mean test AUC over $10$ seeds.", "tab:s_dirichlet"))
