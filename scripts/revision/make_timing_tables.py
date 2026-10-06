#!/usr/bin/env python3
"""Step 3 tables: timing-grade cost decomposition, message counts, amortized cost, and break-even.

Break-even (per seed): K* = mean one-time sparsification cost / mean saving per downstream run (full graph minus G_r).
Against another sparsifier at the same exact budget the downstream graphs have the same |E_r| and hence the
same message count; the downstream training time was measured on CILP's sparsified graphs, so an equal
downstream cost is assumed for the comparator and no finite break-even exists if CILP's one-time cost is higher.
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

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.revision.common import REVISED_DIR  # noqa: E402

TAB, FIG = ROOT / "manuscript" / "minimal" / "tables", ROOT / "manuscript" / "minimal" / "figures"
DS = ["lastfm", "facebook", "github", "pubmed"]
DSN = {"lastfm": "LastFM", "facebook": "Facebook", "github": "GitHub", "pubmed": "PubMed"}
NL = "\\\\"


def ms(v, d=1):
    v = np.asarray(v, float)
    return f"{v.mean():.{d}f}$\\pm${v.std(ddof=1):.{d}f}"


def load(ds):
    return [json.loads(Path(f).read_text()) for f in sorted(glob.glob(str(REVISED_DIR / "timing" / ds / "seed*.json")))]


def main():
    TAB.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)
    summary = {}
    # ---- main table: one-time cost decomposition + memory
    rows = []
    for ds in DS:
        R = load(ds)
        assert len(R) == 10, ds
        for v, lab in (("cilp_full", "CILP"), ("task_only", "Task-only")):
            T = lambda k: [r[v]["timings"][k] for r in R]  # noqa: E731
            rows.append(" & ".join([DSN[ds] if v == "cilp_full" else "", lab, str(R[0][v]["teacher_n"]),
                                    ms(T("teacher_total")), ms(T("teacher_probes"), 2), ms(T("scorer_total")),
                                    ms(T("scorer_rank_inference"), 2), ms(T("one_time_total")),
                                    f"{np.mean(T('teacher_peak_rss_mb'))/1024:.1f}", f"{np.mean(T('scorer_fit_peak_rss_mb'))/1024:.1f}"]) + " " + NL)
        b = R[0]["baselines"]
        rows.append(" & ".join(["", "Local Degree / Local Sim.", "--", "--", "--", "--", "--",
                                f"{np.mean([r['baselines']['nk_local_degree']['scorer_fit_seconds'] for r in R]):.1f} / "
                                f"{np.mean([r['baselines']['nk_local_similarity']['scorer_fit_seconds'] for r in R]):.1f}", "--", "--"]) + " " + NL)
        rows.append(" & ".join(["", "ILP-GCN / PTDNet", "--", "--", "--", "--", "--",
                                f"{np.mean([r['baselines']['original_ilp']['scorer_fit_seconds'] for r in R]):.1f} / "
                                f"{np.mean([r['baselines']['ptdnet']['scorer_fit_seconds'] for r in R]):.1f}", "--", "--"]) + " " + NL)
        rows.append("\\midrule" if ds != DS[-1] else "\\bottomrule")
        summary[ds] = {"cilp_one_time": float(np.mean(T("one_time_total"))) if False else float(np.mean([r["cilp_full"]["timings"]["one_time_total"] for r in R]))}
    head = ("Dataset & Teacher & $n$ & Teacher (s) & of which probes (s) & Scorer (s) & Ranking (s) & One-time total (s) & "
            "Peak RSS teacher (GB) & Peak RSS scorer (GB) " + NL)
    (TAB / "rev_runtime.tex").write_text("\n".join(["\\begin{tabular}{@{}llrcccccc c@{}}".replace(" c@", "c@"), "\\toprule", head, "\\midrule"] + rows[:-1] + ["\\bottomrule", "\\end{tabular}"]) + "\n")

    # ---- break-even and message counts per budget
    rows, fig_data = [], {}
    for ds in DS:
        R = load(ds)
        C = np.array([r["cilp_full"]["timings"]["one_time_total"] for r in R])
        sp = np.array([sum(b["sparsify_seconds"] for b in r["cilp_full"]["per_budget"]) for r in R])
        Ftr = np.array([r["cilp_full"]["full_graph"]["downstream_train_seconds"] for r in R])
        Finf = np.array([r["cilp_full"]["full_graph"]["downstream_inference_seconds"] for r in R])
        M0 = R[0]["cilp_full"]["full_graph"]["messages_per_layer"]
        fig_data[ds] = []
        for i, b0 in enumerate(R[0]["cilp_full"]["per_budget"]):
            r_ = b0["removal_rate"]
            tr = np.array([r["cilp_full"]["per_budget"][i]["downstream_train_seconds"] for r in R])
            inf = np.array([r["cilp_full"]["per_budget"][i]["downstream_inference_seconds"] for r in R])
            save = Ftr - tr
            k = np.where(save > 0, C / np.where(save > 0, save, 1), np.inf)
            sv_inf = Finf - inf
            kinf = np.where(sv_inf > 0, C / np.where(sv_inf > 0, sv_inf, 1), np.inf)
            fig_data[ds].append((r_, b0["messages_per_layer"] / M0, tr.mean() / Ftr.mean()))
            if r_ in (0.1, 0.3, 0.5, 0.7, 0.9):
                nz = int((save <= 0).sum())
                kk = (f"{C.mean()/save.mean():.0f}" if save.mean() > 0 else "$\\infty$") + (f" ({nz}/10 seeds no saving)" if nz else "")
                ki = f"{C.mean()/sv_inf.mean():.0f}" if sv_inf.mean() > 0 else "$\\infty$"
                rows.append(" & ".join([DSN[ds] if r_ == 0.1 else "", f"{r_:.1f}", f"{b0['messages_per_layer']:,}".replace(",", "\\,"),
                                        f"{100*b0['messages_per_layer']/M0:.0f}\\%", ms(tr, 2), f"{100*(1-tr.mean()/Ftr.mean()):.0f}\\%",
                                        f"{1000*inf.mean():.1f}", kk, ki]) + " " + NL)
        rows.append(" & ".join(["", "0 (full)", f"{M0:,}".replace(",", "\\,"), "100\\%", ms(Ftr, 2), "--", f"{1000*Finf.mean():.1f}",
                                f"amortized one-time cost per budget: {np.mean((C + sp)/9):.1f}\\,s", ""]) + " " + NL)
        rows.append("\\midrule" if ds != DS[-1] else "\\bottomrule")
        summary[ds].update({"amortized_per_budget": float(np.mean((C + sp) / 9)), "full_train": float(Ftr.mean())})
    head = ("Dataset & $r$ & Messages/layer & of full & GCN train (s) & Saving & Inference (ms) & "
            "Break-even $K^\\ast$ (train runs) & $K^\\ast$ (inferences) " + NL)
    (TAB / "rev_breakeven.tex").write_text("\n".join(["\\begin{tabular}{@{}lrrrcrrcr@{}}", "\\toprule", head, "\\midrule"] + rows[:-1] + ["\\bottomrule", "\\end{tabular}"]) + "\n")

    # ---- figure: relative messages vs relative downstream train time
    plt.rcParams.update({"font.family": "sans-serif", "font.size": 8, "pdf.fonttype": 42})
    fig, ax = plt.subplots(figsize=(4.4, 3.1))
    for ds, c in zip(DS, ["#0072B2", "#D55E00", "#009E73", "#CC79A7"]):
        r_, m_, t_ = zip(*fig_data[ds])
        ax.plot(r_, m_, color=c, ls="--", lw=1)
        ax.plot(r_, t_, color=c, lw=1.6, marker="o", ms=2.5, label=DSN[ds])
    ax.set_xlabel("Edge-removal rate $r$"); ax.set_ylabel("Fraction of full graph")
    ax.set_title("Messages (dashed) vs. GCN training time (solid)", fontsize=8)
    ax.legend(frameon=False, fontsize=7); ax.grid(alpha=0.25)
    fig.savefig(FIG / "rev_fig_runtime.pdf", bbox_inches="tight"); plt.close(fig)
    (REVISED_DIR / "tables" / "timing_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
