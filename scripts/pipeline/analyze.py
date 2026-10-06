#!/usr/bin/env python3
"""Analysis: builds every table from results/main/final (+ timing), exactly per
docs/analysis/PREREGISTRATION.md. Outputs CSV to results/main/tables/ (all rows carry a protocol tag).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.pipeline.common import BUDGETS, RESULTS_DIR, teacher_variants  # noqa: E402
from src.pipeline.stats import SESOI_AUC, holm, paired, sparsity_auc  # noqa: E402

TAB = RESULTS_DIR / "tables"
DATASETS = ["lastfm", "facebook", "github", "pubmed"]
BASELINES = ["random", "original_ilp", "neuralsparse", "ptdnet", "resistance_proxy", "gassip_adapted",
             "mog_adapted", "psgnn_reimpl", "nk_local_degree", "nk_local_similarity"]
STRUCT = ["giant_component_ratio", "bridge_retention", "minority_degree_retention"]
COMM = ["conductance_ratio", "nmi_leiden", "ari_leiden", "modularity_retention"]
DISPLAY = {"cilp_full": "CILP", "task_only": "Task-only", "random": "Random", "original_ilp": "ILP-GCN",
           "neuralsparse": "NeuralSparse", "ptdnet": "PTDNet", "resistance_proxy": "Resistance proxy",
           "gassip_adapted": "GASSIP-adapted", "mog_adapted": "MoG-adapted", "psgnn_reimpl": "PSGNN-reimpl",
           "nk_local_degree": "Local Degree", "nk_local_similarity": "Local Similarity", "full_graph": "Full graph"}


def load_final() -> pd.DataFrame:
    recs = []
    for p in (RESULTS_DIR / "final").glob("*/seed*/*.json"):
        r = json.loads(p.read_text())
        method = r["variant"].replace("baseline_", "")
        for row in r["rows"]:
            recs.append({"dataset": r["dataset"], "seed": r["seed"], "method": method,
                         "protocol": r["protocol"], "teacher_n": r.get("teacher_n"), "file": p.name, **row})
    return pd.DataFrame(recs)


def primary_rows(df: pd.DataFrame, lock: dict) -> pd.DataFrame:
    """Rows of the train setting used in the confirmatory analysis: each teacher at its locked n."""
    keep = []
    for ds in df.dataset.unique():
        d = df[(df.dataset == ds) & (df.protocol == "train")]
        for m in d.method.unique():
            dm = d[d.method == m]
            if m in teacher_variants():
                n = lock["teacher_n"][ds]["task_only" if m == "task_only" else "cilp_full"]
                dm = dm[dm.teacher_n == n]
            keep.append(dm)
    return pd.concat(keep)


def auc_table(df: pd.DataFrame) -> pd.DataFrame:
    out = []
    for (ds, m, s), g in df[df.removal_rate > 0].groupby(["dataset", "method", "seed"]):
        if len(g) != len(BUDGETS):
            continue
        out.append({"dataset": ds, "method": m, "seed": s, "auc": sparsity_auc(g.removal_rate, g.test_macro_f1)})
    return pd.DataFrame(out)


def at(df: pd.DataFrame, r: float, metric: str) -> pd.DataFrame:
    return df[np.isclose(df.removal_rate, r)][["dataset", "method", "seed", metric]]


def seed_vec(tab: pd.DataFrame, ds: str, m: str, col: str) -> pd.Series:
    return tab[(tab.dataset == ds) & (tab.method == m)].set_index("seed")[col].sort_index()


def family(ds: str, contrasts: list) -> list:
    rows = []
    for name, a, b in contrasts:
        common = a.index.intersection(b.index)
        if len(common) < 2:
            continue
        rows.append({"dataset": ds, "comparison": name, **paired(a[common].values, b[common].values)})
    if rows:
        for r, h in zip(rows, holm([r["p_raw"] for r in rows])):
            r["p_holm"] = h
    return rows


def main() -> None:
    TAB.mkdir(parents=True, exist_ok=True)
    lock = json.loads((ROOT / "configs" / "selected" / "locked.json").read_text())
    df_all = load_final()
    df = primary_rows(df_all, lock)
    auc = auc_table(df)
    auc["protocol"] = "train"
    auc.to_csv(TAB / "auc_per_seed.csv", index=False)

    # descriptive per method / budget (supplement), incl. r = 0 reference
    desc = (df.groupby(["dataset", "method", "removal_rate"])
              [["test_macro_f1"] + STRUCT + COMM + ["downstream_train_seconds", "messages_per_layer"]]
              .agg(["mean", "std"]).reset_index())
    desc.columns = ["_".join(c).strip("_") for c in desc.columns]
    desc.insert(0, "protocol", "train")
    desc.to_csv(TAB / "per_budget_descriptive.csv", index=False)
    (auc.groupby(["dataset", "method"]).auc.agg(["mean", "std", "count"]).reset_index()
        .assign(protocol="train").to_csv(TAB / "auc_summary.csv", index=False))

    # Family P (confirmatory) and global sensitivity
    fam_rows, comm_rows = [], []
    for ds in DATASETS:
        if ds not in set(auc.dataset):
            continue
        c = [("CILP - Task-only: AUC", seed_vec(auc, ds, "cilp_full", "auc"), seed_vec(auc, ds, "task_only", "auc"))]
        for met in STRUCT:
            t = at(df, 0.5, met)
            c.append((f"CILP - Task-only: {met}@0.5", seed_vec(t, ds, "cilp_full", met), seed_vec(t, ds, "task_only", met)))
        for b in BASELINES:
            c.append((f"CILP - {DISPLAY[b]}: AUC", seed_vec(auc, ds, "cilp_full", "auc"), seed_vec(auc, ds, b, "auc")))
        fam_rows += family(ds, c)
        cc = []
        for comp in ("task_only", "original_ilp"):
            for met in COMM:
                t = at(df, 0.5, met)
                cc.append((f"CILP - {DISPLAY[comp]}: {met}@0.5", seed_vec(t, ds, "cilp_full", met), seed_vec(t, ds, comp, met)))
        comm_rows += family(ds, cc)
    fam = pd.DataFrame(fam_rows)
    if len(fam):
        main_sets = fam[fam.dataset != "pubmed"]
        g = holm(main_sets.p_raw.tolist())
        fam.loc[main_sets.index, "p_holm_global_3datasets"] = g
        fam["practically_relevant"] = fam.apply(
            lambda r: (r.ci_low > 0 and r.mean_diff > SESOI_AUC) if r.comparison.endswith("AUC") else np.nan, axis=1)
        fam.insert(0, "protocol", "train")
        fam.to_csv(TAB / "family_P.csv", index=False)
    cf = pd.DataFrame(comm_rows)
    if len(cf):
        cf.insert(0, "protocol", "train")
        cf.to_csv(TAB / "family_C_community.csv", index=False)

    # Ablation / weights (exploratory): variant vs full CILP, at CILP's locked n
    abl = []
    variants = [v for v in teacher_variants() if v not in ("cilp_full",)]
    for ds in DATASETS:
        full_auc = seed_vec(auc, ds, "cilp_full", "auc")
        if full_auc.empty:
            continue
        rows = []
        for v in variants:
            va = seed_vec(auc, ds, v, "auc")
            if va.empty:
                continue
            rec = {"dataset": ds, "variant": v, "auc_mean": va.mean(), "auc_sd": va.std(ddof=1),
                   **{f"vs_full_{k}": x for k, x in paired(va.values, full_auc[va.index].values).items()}}
            for met in STRUCT + COMM:
                t = at(df, 0.5, met)
                vv, ff = seed_vec(t, ds, v, met), seed_vec(t, ds, "cilp_full", met)
                rec[f"{met}@0.5_mean"] = vv.mean()
                rec[f"{met}@0.5_diff_vs_full"] = (vv - ff[vv.index]).mean()
            rows.append(rec)
        for r, h in zip(rows, holm([r["vs_full_p_raw"] for r in rows])):
            r["vs_full_p_holm_exploratory"] = h
        abl += rows
    if abl:
        a = pd.DataFrame(abl)
        a.insert(0, "protocol", "train")
        a.to_csv(TAB / "ablation_weights.csv", index=False)

    # trainval sensitivity setting and matched-n sensitivity
    rep = df_all[df_all.protocol == "trainval"]
    if len(rep):
        ra = auc_table(rep)
        ra.groupby(["dataset", "method"]).auc.agg(["mean", "std", "count"]).reset_index().assign(
            protocol="trainval").to_csv(TAB / "auc_stage_a_setting.csv", index=False)
    sens = df_all[(df_all.protocol == "train") & df_all.method.isin(["cilp_full", "task_only"])]
    sa = []
    for (ds, m, n), g in sens.groupby(["dataset", "method", "teacher_n"]):
        t = auc_table(g)
        if len(t):
            sa.append({"dataset": ds, "method": m, "teacher_n": n, "auc_mean": t.auc.mean(), "auc_sd": t.auc.std(ddof=1), "seeds": len(t)})
    if sa:
        pd.DataFrame(sa).assign(protocol="train").to_csv(TAB / "matched_n_sensitivity.csv", index=False)
    print("tables written to", TAB)


if __name__ == "__main__":
    main()
