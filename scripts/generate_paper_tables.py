#!/usr/bin/env python3
"""Regenerate paper tables/CSVs from authoritative results (no invented values)."""
from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
AUTH = ROOT / "results" / "processed" / "authoritative_results.csv"
TAB = ROOT / "paper" / "tables"
DIAG = ROOT / "results" / "raw" / "grid" / "facebook_diagnostic_seed0.json"

LABELS = {
    "cailp_social_multi": "CILP",
    "cailp_a31": "Task-only",
    "original_ilp_gcn": "ILP-GCN",
    "ptdnet": "PTDNet",
    "neuralsparse": "NeuralSparse",
    "random": "Random",
    "resistance_style_proxy": "Resistance proxy",
}
STRUCT_METHODS = [
    "cailp_social_multi",
    "cailp_a31",
    "original_ilp_gcn",
    "ptdnet",
    "random",
]
STRUCT_KEYS = [
    ("giant_component_ratio", "Giant-component ratio"),
    ("bridge_preservation", "Bridge retention"),
    ("minority_degree_retention", "Minority-degree retention"),
]
METRIC_NAMES = [
    "sparsity_macro_f1_auc",
    "giant_component_ratio",
    "bridge_preservation",
    "minority_degree_retention",
]


def load_rows():
    return list(csv.DictReader(AUTH.open()))


def sparsity_auc(pts):
    pts = sorted(pts)
    xs = np.array([p[0] for p in pts])
    ys = np.array([p[1] for p in pts])
    return float(np.trapz(ys, xs))


def paired(a, b):
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    d = a - b
    n = len(d)
    mean_d = float(d.mean())
    sd = float(d.std(ddof=1))
    sem = sd / np.sqrt(n)
    tcrit = float(stats.t.ppf(0.975, n - 1))
    t_stat, t_p = stats.ttest_rel(a, b)
    w = stats.wilcoxon(d)
    dz = mean_d / sd if sd > 1e-12 else float("nan")
    return {
        "n": n,
        "mean_a": float(a.mean()),
        "mean_b": float(b.mean()),
        "mean_diff": mean_d,
        "ci_low": mean_d - tcrit * sem,
        "ci_high": mean_d + tcrit * sem,
        "t_stat": float(t_stat),
        "t_p": float(t_p),
        "wilcoxon_stat": float(w.statistic),
        "wilcoxon_p": float(w.pvalue),
        "dz": float(dz),
        "wins": int((d > 0).sum()),
        "ties": int((d == 0).sum()),
        "losses": int((d < 0).sum()),
    }


def holm(ps):
    m = len(ps)
    order = np.argsort(ps)
    out = np.empty(m)
    running = 0.0
    for i, idx in enumerate(order):
        running = max(running, min(1.0, ps[idx] * (m - i)))
        out[idx] = running
    return out


def write_paired_structural(rows):
    multi, task = "cailp_social_multi", "cailp_a31"
    all_tests = []
    decision = {"family": "per-dataset 4-metric Holm on Wilcoxon p (CILP − Task-only)", "datasets": {}}
    for ds in ("lastfm", "facebook", "github"):
        if not any(r["dataset"] == ds for r in rows):
            continue
        auc = {multi: {}, task: {}}
        struct = {multi: {k: {} for k in METRIC_NAMES[1:]}, task: {k: {} for k in METRIC_NAMES[1:]}}
        by_m = defaultdict(lambda: defaultdict(list))
        for r in rows:
            if r["dataset"] != ds or r["method"] not in (multi, task):
                continue
            seed = int(r["seed"])
            rem = float(r["edge_removal_rate"])
            by_m[r["method"]][seed].append((rem, float(r["test_macro_f1"])))
            if abs(rem - 0.5) < 1e-9:
                for k in METRIC_NAMES[1:]:
                    if r.get(k) not in (None, ""):
                        struct[r["method"]][k][seed] = float(r[k])
        for m in (multi, task):
            for s, pts in by_m[m].items():
                auc[m][s] = sparsity_auc(pts)
        family = []
        for name in METRIC_NAMES:
            if name == "sparsity_macro_f1_auc":
                common = sorted(set(auc[multi]) & set(auc[task]))
                if len(common) < 2:
                    continue
                st = paired([auc[multi][s] for s in common], [auc[task][s] for s in common])
            else:
                common = sorted(set(struct[multi][name]) & set(struct[task][name]))
                if len(common) < 2:
                    continue
                st = paired(
                    [struct[multi][name][s] for s in common],
                    [struct[task][name][s] for s in common],
                )
            st.update(dataset=ds, metric=name, seed_ids=" ".join(map(str, common)))
            family.append(st)
        if len(family) != 4:
            decision["datasets"][ds] = {
                "dims_supported": 0,
                "n_metrics": len(family),
                "status": "incomplete_family",
            }
            all_tests.extend(family)
            continue
        adj = holm([t["wilcoxon_p"] for t in family])
        dims = 0
        for t, h in zip(family, adj):
            t["wilcoxon_p_holm"] = float(h)
            t["supported"] = bool(t["mean_diff"] > 0 and t["wilcoxon_p_holm"] < 0.05)
            if t["supported"]:
                dims += 1
            all_tests.append(t)
        decision["datasets"][ds] = {
            "dims_supported": dims,
            "n_metrics": 4,
            "supported_on_dataset": dims >= 2,
        }
    n_ds = sum(1 for v in decision["datasets"].values() if v.get("supported_on_dataset"))
    decision["multi_dimensional_support"] = n_ds >= 2
    decision["rule"] = (
        "Per dataset: multi-criteria supported if ≥2 of 4 metrics have positive Δ and Holm Wilcoxon p<0.05. "
        "Across datasets: headline support if ≥2 datasets meet that criterion."
    )
    gh = decision["datasets"].get("github", {})
    decision["github_decision"] = (
        "Supported"
        if gh.get("supported_on_dataset")
        else ("Not supported" if gh.get("n_metrics") == 4 else "Inconclusive due to incomplete paired seeds")
    )
    decision["conclusion"] = (
        f"Across-dataset support={decision['multi_dimensional_support']}; "
        f"GitHub={decision['github_decision']}; details per dataset in datasets map."
    )

    fields = [
        "dataset",
        "metric",
        "seed_ids",
        "n",
        "mean_a",
        "mean_b",
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
        "ties",
        "losses",
        "supported",
    ]
    with (TAB / "paired_structural_tests.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for t in all_tests:
            w.writerow({k: t[k] for k in fields})
    (ROOT / "results" / "processed" / "multi_dim_decision.json").write_text(json.dumps(decision, indent=2))

    # SI embedded table
    lines = [
        r"\begin{center}\scriptsize",
        r"\begin{longtable}{@{}llrrrrrrrcc@{}}",
        r"\caption{Paired CILP versus Task-only tests (common seeds). "
        r"Family: four metrics per dataset; Holm adjustment on Wilcoxon $p$.}\\",
        r"\toprule",
        r"Dataset & Metric & CILP & Task-only & $\Delta$ & 95\% CI & Wilcoxon $p$ & Holm $p$ & $d_z$ & W/T/L \\",
        r"\midrule",
        r"\endfirsthead",
        r"\toprule",
        r"Dataset & Metric & CILP & Task-only & $\Delta$ & 95\% CI & Wilcoxon $p$ & Holm $p$ & $d_z$ & W/T/L \\",
        r"\midrule",
        r"\endhead",
    ]
    nice = {
        "sparsity_macro_f1_auc": "AUC",
        "giant_component_ratio": "GC @50\\%",
        "bridge_preservation": "Bridge @50\\%",
        "minority_degree_retention": "Min-deg @50\\%",
    }
    for t in all_tests:
        lines.append(
            f"{t['dataset']} & {nice[t['metric']]} & {t['mean_a']:.3f} & {t['mean_b']:.3f} & "
            f"{t['mean_diff']:+.4f} & [{t['ci_low']:+.4f},{t['ci_high']:+.4f}] & "
            f"{t['wilcoxon_p']:.4g} & {t['wilcoxon_p_holm']:.4g} & {t['dz']:.3f} & "
            f"{t['wins']}/{t['ties']}/{t['losses']} \\\\"
        )
    lines += [r"\bottomrule", r"\end{longtable}", r"\end{center}"]
    (TAB / "s21_paired_multidim.tex").write_text("\n".join(lines) + "\n")
    return decision


def write_structure_table(rows):
    """Main-text Table 4 + SI fragment; must match Fig. 5 method set."""
    lines = [
        r"\scriptsize",
        r"\begin{tabular}{@{}llcccc@{}}",
        r"\toprule",
        r"Dataset & Method & $n$ & Giant-component ratio & Bridge retention & Minority-degree retention \\",
        r"\midrule",
    ]
    csv_rows = []
    for ds in ("lastfm", "facebook", "github"):
        if not any(r["dataset"] == ds for r in rows):
            continue
        for m in STRUCT_METHODS:
            vals = {k: [] for k, _ in STRUCT_KEYS}
            seeds = set()
            for r in rows:
                if r["dataset"] != ds or r["method"] != m:
                    continue
                if abs(float(r["edge_removal_rate"]) - 0.5) > 1e-9:
                    continue
                seeds.add(int(r["seed"]))
                for k, _ in STRUCT_KEYS:
                    if r.get(k) not in (None, ""):
                        vals[k].append(float(r[k]))
            n = len(seeds)
            means = {k: float(np.mean(v)) for k, v in vals.items() if v}
            lines.append(
                f"{ds} & {LABELS[m]} & {n} & {means['giant_component_ratio']:.3f} & "
                f"{means['bridge_preservation']:.3f} & {means['minority_degree_retention']:.3f} \\\\"
            )
            csv_rows.append(
                {
                    "dataset": ds,
                    "method": m,
                    "method_display": LABELS[m],
                    "n": n,
                    "giant_component_ratio": means["giant_component_ratio"],
                    "bridge_preservation": means["bridge_preservation"],
                    "minority_degree_retention": means["minority_degree_retention"],
                }
            )
    lines += [r"\bottomrule", r"\end{tabular}"]
    (TAB / "table4_structure.tex").write_text("\n".join(lines) + "\n")
    (TAB / "s22_structure.tex").write_text("\n".join(lines) + "\n")
    with (TAB / "table4_structure.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(csv_rows[0].keys()))
        w.writeheader()
        w.writerows(csv_rows)
    return csv_rows


def write_teacher_size():
    rows = json.loads(DIAG.read_text())
    by_n = defaultdict(dict)
    for r in rows:
        n = int(str(r["method"]).split("teacher")[-1])
        rem = float(r["removal_rate"])
        by_n[n][rem] = (float(r["val_macro_f1"]), float(r["test_macro_f1"]))
    csv_rows = []
    for n in [20, 40, 80, 120]:
        vals = [by_n[n][r][0] for r in (0.3, 0.5, 0.7)]
        csv_rows.append(
            {
                "teacher_n": n,
                "val_0.3": by_n[n][0.3][0],
                "test_0.3": by_n[n][0.3][1],
                "val_0.5": by_n[n][0.5][0],
                "test_0.5": by_n[n][0.5][1],
                "val_0.7": by_n[n][0.7][0],
                "test_0.7": by_n[n][0.7][1],
                "mean_val_macro_f1": float(np.mean(vals)),
                "selected_by_val_at_0.5": n == 120,
                "negative_pilot": n == 40,
            }
        )
    with (TAB / "teacher_size_sensitivity.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(csv_rows[0].keys()))
        w.writeheader()
        w.writerows(csv_rows)
    # selection: code ranks by val@0.5
    lines = [
        r"\begin{tabular}{@{}lccccccc@{}}",
        r"\toprule",
        r"Teacher $n$ & Val @0.3 & Test @0.3 & Val @0.5 & Test @0.5 & Val @0.7 & Test @0.7 & Mean val \\",
        r"\midrule",
    ]
    for r in csv_rows:
        tag = ""
        if r["teacher_n"] == 120:
            tag = r" (selected @50\% val)"
        elif r["teacher_n"] == 40:
            tag = r" (neg.\ pilot)"
        lines.append(
            f"{r['teacher_n']}{tag} & {r['val_0.3']:.3f} & {r['test_0.3']:.3f} & "
            f"{r['val_0.5']:.3f} & {r['test_0.5']:.3f} & {r['val_0.7']:.3f} & {r['test_0.7']:.3f} & "
            f"{r['mean_val_macro_f1']:.3f} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}"]
    (TAB / "s8_teacher_size.tex").write_text("\n".join(lines) + "\n")
    return csv_rows


def write_teacher_components():
    rows = [
        {
            "Criterion": "Task loss",
            "Category": "Deletion-induced",
            "Exact formula": "CE_TV(G_{-e}) - CE_TV(G) on train∪val",
            "Direction": "Higher = more harmful",
            "Normalization": "Min-max over teacher sample",
            "Weight β": "1.0",
            "Code location": "exact_teacher.py::task_and_repr_and_group_effects",
        },
        {
            "Criterion": "Connectivity change",
            "Category": "Deletion-induced",
            "Exact formula": "1[bridge]+(n_comp(G_{-e})-n_comp(G))+max(0,giant(G)-giant(G_{-e}))/|V|",
            "Direction": "Higher = more harmful",
            "Normalization": "Min-max over teacher sample",
            "Weight β": "1.0",
            "Code location": "exact_teacher.py::connectivity_effect",
        },
        {
            "Criterion": "Representation shift",
            "Category": "Deletion-induced",
            "Exact formula": "mean_i ||h_i(G)-h_i(G_{-e})||_2",
            "Direction": "Higher = more harmful",
            "Normalization": "Min-max over teacher sample",
            "Weight β": "0.5",
            "Code location": "exact_teacher.py::task_and_repr_and_group_effects",
        },
        {
            "Criterion": "Group impact",
            "Category": "Deletion-induced",
            "Exact formula": "max(0, F1_worst_val(G)-F1_worst_val(G_{-e}))",
            "Direction": "Higher = more harmful",
            "Normalization": "Min-max over teacher sample",
            "Weight β": "0.5",
            "Code location": "exact_teacher.py::task_and_repr_and_group_effects",
        },
        {
            "Criterion": "Community proxy",
            "Category": "Structural proxy",
            "Exact formula": "max(0,local)+inter; local=(1-du*dv/(2m)) if same community else |1-du*dv/(2m)|; inter=0 if same else 1; Louvain once on G",
            "Direction": "Higher = more harmful",
            "Normalization": "Min-max over teacher sample",
            "Weight β": "0.5",
            "Code location": "exact_teacher.py::community_effect",
        },
        {
            "Criterion": "Degree-based spectral proxy",
            "Category": "Structural proxy",
            "Exact formula": "For |V|>400 (all reported datasets): 1/du+1/dv with du,dv>=1; else ||λ_norm[1:4](G)-λ_norm[1:4](G_{-e})||_2",
            "Direction": "Higher = more harmful",
            "Normalization": "Min-max over teacher sample",
            "Weight β": "0.3",
            "Code location": "exact_teacher.py::spectral_effect",
        },
    ]
    with (TAB / "teacher_component_definitions.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def write_hyperparameters():
    rows = [
        {"group": "Model", "parameter": "encoder_type", "value": "gcn", "source": "run_full_grid.py CAILPConfig"},
        {"group": "Model", "parameter": "num_layers", "value": "2", "source": "run_full_grid.py CAILPConfig"},
        {"group": "Model", "parameter": "hidden_dim", "value": "64", "source": "run_full_grid.py CAILPConfig"},
        {"group": "Model", "parameter": "dropout (CAILP)", "value": "0.4", "source": "run_full_grid.py CAILPConfig"},
        {"group": "Model", "parameter": "fusion (core)", "value": "concat", "source": "run_full_grid.py"},
        {"group": "Model", "parameter": "decoder", "value": "ImportanceDecoder hidden 64, dropout 0.2, ReLU", "source": "importance_decoder.py"},
        {"group": "Model", "parameter": "mu activation", "value": "sigmoid", "source": "importance_decoder.py"},
        {"group": "Model", "parameter": "logvar clamp", "value": "[-8, 4]", "source": "importance_decoder.py"},
        {"group": "Model", "parameter": "LayerNorm in fusion", "value": "yes (ConcatFusion)", "source": "cross_attention.py"},
        {"group": "Model", "parameter": "NodeEncoder residual/norm defaults", "value": "residual=True, norm=True", "source": "node_encoders.py"},
        {"group": "Optimization", "parameter": "optimizer (CAILP)", "value": "Adam", "source": "run_full_grid.py"},
        {"group": "Optimization", "parameter": "learning_rate (CAILP)", "value": "1e-3", "source": "run_full_grid.py"},
        {"group": "Optimization", "parameter": "weight_decay (CAILP)", "value": "Not explicitly configured; library default used (0.0)", "source": "torch.optim.Adam"},
        {"group": "Optimization", "parameter": "scheduler", "value": "Not explicitly configured; none used", "source": "run_full_grid.py"},
        {"group": "Optimization", "parameter": "teacher_encoder_lr", "value": "1e-2", "source": "run_full_grid.py pretrain_encoder"},
        {"group": "Optimization", "parameter": "surrogate_epochs", "value": "60", "source": "run_full_grid.py"},
        {"group": "Optimization", "parameter": "surrogate_hidden", "value": "64", "source": "run_full_grid.py"},
        {"group": "Optimization", "parameter": "early_stopping_patience", "value": "Not explicitly configured; fixed epoch loops", "source": "run_full_grid.py"},
        {"group": "Optimization", "parameter": "batch_size", "value": "Not explicitly configured; full-batch graphs", "source": "run_full_grid.py"},
        {"group": "Optimization", "parameter": "gradient_clipping", "value": "Not explicitly configured; none used", "source": "run_full_grid.py"},
        {"group": "Optimization", "parameter": "downstream_classifier", "value": "2-layer GCN, hidden 64, dropout 0.5", "source": "node_classification.py"},
        {"group": "Optimization", "parameter": "downstream_lr", "value": "1e-2", "source": "node_classification.py"},
        {"group": "Optimization", "parameter": "downstream_weight_decay", "value": "5e-4", "source": "node_classification.py"},
        {"group": "Loss", "parameter": "lambda_CE", "value": "0.5", "source": "run_full_grid.py"},
        {"group": "Loss", "parameter": "lambda_NLL", "value": "1.0", "source": "run_full_grid.py"},
        {"group": "Loss", "parameter": "lambda_rank", "value": "0.2", "source": "run_full_grid.py"},
        {"group": "Loss", "parameter": "ranking_margin", "value": "0.1", "source": "importance_decoder.py default"},
        {"group": "Loss", "parameter": "ranking_num_pairs", "value": "256", "source": "importance_decoder.py default"},
        {"group": "Loss", "parameter": "NLL variance floor", "value": "1e-6", "source": "importance_decoder.py"},
        {"group": "Teacher", "parameter": "teacher_n LastFM", "value": "50", "source": "GRID_CFG"},
        {"group": "Teacher", "parameter": "teacher_n Facebook", "value": "120", "source": "GRID_CFG; selected by val Macro-F1 @50%"},
        {"group": "Teacher", "parameter": "teacher_n GitHub", "value": "40", "source": "GRID_CFG"},
        {"group": "Teacher", "parameter": "teacher_epochs LastFM/Facebook/GitHub", "value": "25/25/15", "source": "GRID_CFG"},
        {"group": "Teacher", "parameter": "scorer_epochs LastFM/Facebook/GitHub", "value": "25/25/15", "source": "GRID_CFG train_epochs"},
        {"group": "Teacher", "parameter": "downstream_epochs LastFM/Facebook/GitHub", "value": "50/40/35", "source": "GRID_CFG"},
        {"group": "Teacher", "parameter": "ilp_epochs LastFM/Facebook/GitHub", "value": "25/20/15", "source": "GRID_CFG"},
        {"group": "Teacher", "parameter": "sampling", "value": "stratified_edge_sample without replacement", "source": "sampling.py / run_full_grid.py"},
        {"group": "Teacher", "parameter": "beta", "value": "(1.0,0.5,1.0,0.3,0.5,0.5)", "source": "CFCoefficients defaults"},
        {"group": "Teacher", "parameter": "beta tuned?", "value": "No; fixed across datasets", "source": "exact_teacher.py"},
        {"group": "Teacher", "parameter": "zero-range handling", "value": "hi-lo<1e-12 → zeros", "source": "sampling.py::normalize_scores"},
        {"group": "Pruning", "parameter": "rule", "value": "keep top (1-r)|E| by descending mu", "source": "budget_prune_unconstrained"},
        {"group": "Pruning", "parameter": "rounding", "value": "n_keep = m - int(round(r*m))", "source": "constrained_pruning.py"},
        {"group": "Pruning", "parameter": "ties", "value": "torch.argsort (stable by implementation)", "source": "constrained_pruning.py"},
        {"group": "Pruning", "parameter": "constraints", "value": "Disabled in reported multi-seed comparison", "source": "run_full_grid.py"},
        {"group": "Statistics", "parameter": "CI", "value": "mean ± t_0.975,n-1 × SEM", "source": "build_authoritative_results.py"},
        {"group": "Statistics", "parameter": "paired tests", "value": "paired t + Wilcoxon on common seeds", "source": "generate_paper_tables.py"},
        {"group": "Statistics", "parameter": "Holm scope", "value": "Within dataset over 4 CILP vs Task-only metrics", "source": "generate_paper_tables.py"},
        {"group": "Statistics", "parameter": "effect size", "value": "dz = mean(diff)/sd(diff)", "source": "generate_paper_tables.py"},
    ]
    with (TAB / "hyperparameters_complete.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["group", "parameter", "value", "source"])
        w.writeheader()
        w.writerows(rows)
    # SI S13 compact tabular
    lines = [
        r"\begin{center}\scriptsize",
        r"\begin{longtable}{@{}p{2.2cm}p{4.2cm}p{6.8cm}@{}}",
        r"\caption{Hyperparameters used in the reported multi-seed comparison.}\\",
        r"\toprule Group & Parameter & Value \\ \midrule \endfirsthead",
        r"\toprule Group & Parameter & Value \\ \midrule \endhead",
    ]
    for r in rows:
        param = r["parameter"].replace("_", r"\_")
        val = r["value"].replace("_", r"\_")
        lines.append(f"{r['group']} & {param} & {val} \\\\")
    lines += [r"\bottomrule", r"\end{longtable}", r"\end{center}"]
    (TAB / "s13_hyperparameters.tex").write_text("\n".join(lines) + "\n")


def write_dataset_audit():
    out = []
    for ds in ("lastfm", "facebook", "github"):
        p = ROOT / "data" / "processed" / f"{ds}.pt"
        data = torch.load(p, weights_only=False)
        n = int(data.num_nodes)
        ei = data.edge_index.cpu().numpy()
        edges = set()
        self_loops = 0
        for i in range(ei.shape[1]):
            u, v = int(ei[0, i]), int(ei[1, i])
            if u == v:
                self_loops += 1
                continue
            edges.add((min(u, v), max(u, v)))
        import networkx as nx

        G = nx.Graph()
        G.add_nodes_from(range(n))
        G.add_edges_from(edges)
        m = len(edges)
        degs = [d for _, d in G.degree()]
        comps = list(nx.connected_components(G))
        y = data.y.cpu().numpy()
        classes = Counter(y.tolist())
        x = data.x
        dens = float((x != 0).float().mean()) if not x.is_sparse else x._nnz() / (n * x.size(1))
        same = sum(1 for u, v in edges if y[u] == y[v])
        row = {
            "dataset": ds,
            "nodes": n,
            "raw_edge_records": int(ei.shape[1]),
            "unique_undirected_edges": m,
            "self_loops_in_tensor": self_loops,
            "duplicates_implied_by_directed_encoding": int(ei.shape[1]) - self_loops - 2 * m,
            "avg_degree": float(np.mean(degs)),
            "median_degree": float(np.median(degs)),
            "max_degree": int(np.max(degs)),
            "n_components": len(comps),
            "giant_component_ratio": max(len(c) for c in comps) / n,
            "avg_clustering": float(nx.average_clustering(G)),
            "n_classes": len(classes),
            "minority_class_proportion": min(classes.values()) / n,
            "feature_dim": int(data.x.size(1)),
            "feature_density": dens,
            "label_homophily": same / m if m else float("nan"),
            "graph_density": 2 * m / (n * (n - 1)),
            "class_counts": " ".join(f"{k}:{v}" for k, v in sorted(classes.items())),
        }
        out.append(row)
    with (TAB / "dataset_audit_complete.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(out[0].keys()))
        w.writeheader()
        w.writerows(out)
    lines = [
        r"\begin{center}\scriptsize",
        r"\begin{tabular}{@{}lrrr@{}}",
        r"\toprule",
        r"Statistic & LastFM & Facebook & GitHub \\",
        r"\midrule",
    ]
    keys = [
        ("nodes", "Nodes", "{:.0f}"),
        ("raw_edge_records", "Raw edge records", "{:.0f}"),
        ("unique_undirected_edges", "Unique undirected edges", "{:.0f}"),
        ("self_loops_in_tensor", "Self-loops in tensor", "{:.0f}"),
        ("avg_degree", "Average degree", "{:.3f}"),
        ("median_degree", "Median degree", "{:.1f}"),
        ("max_degree", "Maximum degree", "{:.0f}"),
        ("n_components", "Connected components", "{:.0f}"),
        ("giant_component_ratio", "Giant-component ratio", "{:.3f}"),
        ("avg_clustering", "Avg.\ clustering", "{:.3f}"),
        ("n_classes", "Classes", "{:.0f}"),
        ("minority_class_proportion", "Minority-class proportion", "{:.4f}"),
        ("feature_dim", "Feature dimension", "{:.0f}"),
        ("feature_density", "Feature density", "{:.4f}"),
        ("label_homophily", "Label homophily", "{:.3f}"),
        ("graph_density", "Graph density", "{:.2e}"),
    ]
    by = {r["dataset"]: r for r in out}
    for k, lab, fmt in keys:
        lines.append(
            f"{lab} & {fmt.format(by['lastfm'][k])} & {fmt.format(by['facebook'][k])} & {fmt.format(by['github'][k])} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{center}"]
    (TAB / "s2_dataset_audit.tex").write_text("\n".join(lines) + "\n")


def main():
    TAB.mkdir(parents=True, exist_ok=True)
    rows = load_rows()
    decision = write_paired_structural(rows)
    write_structure_table(rows)
    write_teacher_size()
    write_teacher_components()
    write_hyperparameters()
    write_dataset_audit()
    print("Tables written. Multi-dim support:", decision["multi_dimensional_support"])
    print(decision["conclusion"])


if __name__ == "__main__":
    main()
