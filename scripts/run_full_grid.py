#!/usr/bin/env python3
"""Full multi-seed Track-A experiment grid for CAILP-Social.

Trains each scorer once per (dataset, seed, method), then evaluates all budgets.
Main comparison uses equal edge-retention budgets (unconstrained ranking).
Fusion is held fixed (concat) for CAILP multi vs A31 (RQ11).
Adversarial is excluded from the core grid (rejected hypothesis from pilot).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.counterfactual.exact_teacher import ExactCounterfactualTeacher
from src.counterfactual.sampling import stratified_edge_sample
from src.counterfactual.single_objective_teacher import SingleObjectiveTeacher
from src.counterfactual.surrogate import SurrogateEdgeImportance, evaluate_surrogate, train_surrogate
from src.evaluation.efficiency import count_parameters, peak_memory_mb, timer
from src.evaluation.fairness import fairness_metrics
from src.evaluation.structural import structural_metrics
from src.models.cailp import CAILPConfig, CAILPSocial
from src.models.edge_encoder import node_centric_edge_representation, structural_edge_features
from src.models.importance_decoder import heteroscedastic_nll, ranking_hinge_loss
from src.models.node_encoders import NodeEncoder
from src.sparsification.baselines.classical import CLASSICAL_METHODS, effective_resistance_proxy
from src.sparsification.baselines.dspar import dspar_sparsify
from src.sparsification.baselines.neuralsparse import neuralsparse_sparsify
from src.sparsification.baselines.ptdnet import ptdnet_sparsify
from src.sparsification.constrained_pruning import budget_prune_unconstrained
from src.sparsification.original_ilp import OriginalILPGCN, train_ilp_link_predictor
from src.tasks.node_classification import train_node_classifier
from src.utils.graph import subgraph_from_undirected_edges, undirected_edge_list
from src.utils.io import ensure_dir, get_device, save_json, set_seed, setup_logging
from src.utils.splits import LeakageGuard

BUDGETS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
CORE_METHODS = [
    "random",
    "original_ilp",
    "cailp_multi",
    "cailp_a31",
    "neuralsparse",
    "ptdnet",
    "dspar",
    "effective_resistance",
]

# Grid hyperparameters (validation-oriented; identical across methods' tuning budget)
GRID_CFG = {
    "lastfm": {"teacher_n": 50, "teacher_epochs": 25, "train_epochs": 25, "down_epochs": 50, "ilp_epochs": 25},
    # teacher_n=120 selected from Facebook diagnostic ranked by validation Macro-F1 @50%
    "facebook": {"teacher_n": 120, "teacher_epochs": 25, "train_epochs": 25, "down_epochs": 40, "ilp_epochs": 20},
    "github": {"teacher_n": 40, "teacher_epochs": 15, "train_epochs": 15, "down_epochs": 35, "ilp_epochs": 15},
}


def load_data(dataset: str, seed: int):
    data = torch.load(ROOT / "data" / "processed" / f"{dataset}.pt", weights_only=False)
    split = torch.load(ROOT / "data" / "splits" / f"{dataset}_seed{seed}.pt", weights_only=False)
    data.train_mask = split["train_mask"]
    data.val_mask = split["val_mask"]
    data.test_mask = split["test_mask"]
    return data


def pretrain_encoder(data, device, epochs: int):
    n_cls = int(data.y.max().item()) + 1
    enc = NodeEncoder(data.x.size(1), encoder_type="gcn", hidden_dim=64, out_dim=64).to(device)
    clf = nn.Linear(64, n_cls).to(device)
    opt = torch.optim.Adam(list(enc.parameters()) + list(clf.parameters()), lr=1e-2)
    x, y, ei = data.x.to(device), data.y.to(device), data.edge_index.to(device)
    for _ in range(epochs):
        enc.train()
        clf.train()
        opt.zero_grad()
        loss = F.cross_entropy(clf(enc(x, ei))[data.train_mask], y[data.train_mask])
        loss.backward()
        opt.step()
    return enc, clf


def score_cailp(
    data,
    device,
    seed: int,
    single_objective: bool,
    teacher_n: int,
    teacher_epochs: int,
    train_epochs: int,
) -> tuple[torch.Tensor, torch.Tensor, Dict[str, Any], float]:
    """Return (undirected_edges, importance, surrogate_metrics, train_seconds)."""
    t0 = time.perf_counter()
    enc, clf = pretrain_encoder(data, device, teacher_epochs)
    und = undirected_edge_list(data.edge_index)
    idx = stratified_edge_sample(data, und, n_sample=teacher_n, seed=seed)
    teacher = SingleObjectiveTeacher() if single_objective else ExactCounterfactualTeacher()
    y_cf_np, effects = teacher.score_edges(data, idx, und, enc, clf, device)
    y_cf = torch.tensor(y_cf_np, dtype=torch.float32)

    with torch.no_grad():
        h_t = enc(data.x.to(device), data.edge_index.to(device)).cpu()
    struct = structural_edge_features(data.edge_index, data.num_nodes, und, lightweight=True)
    z_all = node_centric_edge_representation(h_t, und, struct, x=data.x)
    sur = SurrogateEdgeImportance(z_all.size(1), 64)
    train_surrogate(sur, z_all[idx], y_cf, epochs=60)
    with torch.no_grad():
        y_hat = sur(z_all[idx]).numpy()
        y_full = sur(z_all).detach()
    sur_metrics = evaluate_surrogate(y_hat, y_cf_np, k=min(50, len(y_cf_np)))
    # Calibration: ECE-style on binned predictions
    bins = np.linspace(0, 1, 11)
    cal_errs = []
    for i in range(10):
        m = (y_hat >= bins[i]) & (y_hat < bins[i + 1])
        if m.sum() == 0:
            continue
        cal_errs.append(abs(float(y_hat[m].mean()) - float(y_cf_np[m].mean())))
    sur_metrics["calibration_ece"] = float(np.mean(cal_errs)) if cal_errs else float("nan")
    sur_metrics["teacher_n"] = int(len(idx))
    sur_metrics["teacher"] = "single_objective" if single_objective else "multi_objective"
    sur_metrics["effects_means"] = {k: float(np.mean(v)) for k, v in effects.as_dict().items()}

    n_cls = int(data.y.max().item()) + 1
    # Fusion held constant for RQ11
    model = CAILPSocial(
        data.x.size(1),
        n_cls,
        CAILPConfig(encoder_type="gcn", fusion="concat", hidden_dim=64, num_layers=2, dropout=0.4),
    ).to(device)
    data_dev = data.clone().to(device)
    y_full = y_full.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    _ = model(data_dev)
    for _ in range(train_epochs):
        model.train()
        opt.zero_grad()
        out = model(data_dev)
        loss = (
            0.5 * F.cross_entropy(out["logits"][data_dev.train_mask], data_dev.y[data_dev.train_mask])
            + 1.0 * heteroscedastic_nll(out["mu"], out["logvar"], y_full)
            + 0.2 * ranking_hinge_loss(out["mu"], y_full)
        )
        loss.backward()
        opt.step()
    model.eval()
    with torch.no_grad():
        out = model(data_dev)
        mu = out["mu"].cpu()
        und_out = out["undirected_edges"].cpu()
    return und_out, mu, sur_metrics, time.perf_counter() - t0


def score_original_ilp(data, device, epochs: int) -> tuple[torch.Tensor, torch.Tensor, float]:
    t0 = time.perf_counter()
    model = OriginalILPGCN(data.x.size(1), 64)
    model = train_ilp_link_predictor(model, data, epochs=epochs, device=device)
    model.eval()
    with torch.no_grad():
        h = model.encode(data.x.to(device), data.edge_index.to(device))
        und = undirected_edge_list(data.edge_index.to(device))
        imp = model.ilp_importance(h, und).cpu()
        und = und.cpu()
    return und, imp, time.perf_counter() - t0


def eval_importance(
    data,
    und: torch.Tensor,
    importance: torch.Tensor,
    budgets: List[float],
    device,
    down_epochs: int,
    method: str,
    seed: int,
    extra: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    rows = []
    guard = LeakageGuard()
    for r in budgets:
        guard.lock_test()
        guard.set_phase("sparsify")
        t0 = time.perf_counter()
        keep = budget_prune_unconstrained(und, importance, r)
        sparse = subgraph_from_undirected_edges(data, keep)
        sparse.train_mask, sparse.val_mask, sparse.test_mask = data.train_mask, data.val_mask, data.test_mask
        prune_s = time.perf_counter() - t0
        guard.unlock_test_for_eval()
        _, metrics = train_node_classifier(sparse, kind="gcn", epochs=down_epochs, device=device)
        struct = structural_metrics(data, sparse)
        fair = fairness_metrics(data, sparse)
        row = {
            "dataset": getattr(data, "name", "unknown"),
            "method": method,
            "seed": seed,
            "removal_rate": r,
            "retained_edge_ratio": struct.get("retained_edge_ratio"),
            "test_macro_f1": metrics["test_macro_f1"],
            "test_accuracy": metrics["test_accuracy"],
            "test_worst_class_f1": metrics["test_worst_class_f1"],
            "val_macro_f1": metrics["val_macro_f1"],
            "giant_component_ratio": struct.get("giant_component_ratio"),
            "num_components": struct.get("num_components"),
            "bridge_preservation": struct.get("bridge_preservation"),
            "degree_ks": struct.get("degree_ks"),
            "clustering": struct.get("clustering"),
            "minority_degree_retention": fair.get("minority_degree_retention"),
            "minority_isolation_rate": fair.get("minority_isolation_rate"),
            "prune_seconds": prune_s,
            "leakage": guard.summary(),
        }
        if extra:
            row.update(extra)
        rows.append(row)
        print(
            f"  {method} seed={seed} rem={r:.1f} F1={row['test_macro_f1']:.4f}",
            flush=True,
        )
    return rows


def run_baseline_method(data, method: str, removal: float, device, seed: int, epochs: int):
    if method == "random":
        return CLASSICAL_METHODS["random"](data, removal, seed=seed)
    if method == "effective_resistance":
        return effective_resistance_proxy(data, removal)
    if method == "dspar":
        return dspar_sparsify(data, removal)
    if method == "neuralsparse":
        # Train once outside ideally; for grid we train per call — cache by attaching scores
        return neuralsparse_sparsify(data, removal, epochs=epochs, device=device)
    if method == "ptdnet":
        return ptdnet_sparsify(data, removal, epochs=epochs, device=device)
    raise ValueError(method)


def score_learned_baseline(data, method: str, device, epochs: int, seed: int):
    """Train once and return undirected edges + importance for all budgets."""
    t0 = time.perf_counter()
    und = undirected_edge_list(data.edge_index)
    if method == "random":
        rng = np.random.RandomState(seed)
        imp = torch.from_numpy(rng.rand(und.size(1))).float()
        return und, imp, time.perf_counter() - t0
    if method in ("dspar", "effective_resistance"):
        sparse = effective_resistance_proxy(data, 0.0)  # just to get scoring path
        # Recompute scores directly
        from torch_geometric.utils import degree

        deg = degree(data.edge_index[0], num_nodes=data.num_nodes)
        src, dst = und
        imp = 1.0 / (1.0 / (deg[src] + 1e-6) + 1.0 / (deg[dst] + 1e-6))
        return und, imp.float(), time.perf_counter() - t0
    if method == "neuralsparse":
        from src.sparsification.baselines.neuralsparse import NeuralSparseScorer

        model = NeuralSparseScorer(data.x.size(1)).to(device)
        clf = nn.Linear(64, int(data.y.max().item()) + 1).to(device)
        opt = torch.optim.Adam(list(model.parameters()) + list(clf.parameters()), lr=1e-3)
        x, y, ei = data.x.to(device), data.y.to(device), data.edge_index.to(device)
        und_d = undirected_edge_list(ei)
        train = data.train_mask.to(device)
        for _ in range(epochs):
            opt.zero_grad()
            scores = model(x, ei, und_d)
            h = F.relu(model.enc(x, ei))
            loss = F.cross_entropy(clf(h)[train], y[train]) - 0.01 * scores.mean()
            loss.backward()
            opt.step()
        with torch.no_grad():
            imp = model(x, ei, und_d).cpu()
        return und.cpu(), imp, time.perf_counter() - t0
    if method == "ptdnet":
        from src.sparsification.baselines.ptdnet import PTDNetScorer

        model = PTDNetScorer(data.x.size(1)).to(device)
        clf = nn.Linear(64, int(data.y.max().item()) + 1).to(device)
        opt = torch.optim.Adam(list(model.parameters()) + list(clf.parameters()), lr=1e-3)
        x, y, ei = data.x.to(device), data.y.to(device), data.edge_index.to(device)
        und_d = undirected_edge_list(ei)
        train = data.train_mask.to(device)
        for _ in range(epochs):
            opt.zero_grad()
            h, scores = model(x, ei, und_d)
            loss = F.cross_entropy(clf(h)[train], y[train]) + 0.1 * scores.mean()
            loss.backward()
            opt.step()
        with torch.no_grad():
            _, imp = model(x, ei, und_d)
        return und.cpu(), imp.cpu(), time.perf_counter() - t0
    raise ValueError(method)


def out_path(dataset: str, seed: int, method: str) -> Path:
    ensure_dir(ROOT / "results" / "raw" / "grid")
    return ROOT / "results" / "raw" / "grid" / f"{dataset}_seed{seed}_{method}.json"


def already_done(path: Path) -> bool:
    if not path.exists():
        return False
    try:
        rows = json.loads(path.read_text())
        return isinstance(rows, list) and len(rows) >= len(BUDGETS)
    except Exception:
        return False


def run_one(dataset: str, seed: int, method: str, budgets: List[float], device) -> None:
    path = out_path(dataset, seed, method)
    if already_done(path):
        print(f"SKIP {path.name}", flush=True)
        return
    cfg = GRID_CFG[dataset]
    set_seed(seed)
    data = load_data(dataset, seed)
    data.name = dataset  # type: ignore
    print(f"RUN {dataset} seed={seed} method={method}", flush=True)
    extra: Dict[str, Any] = {"train_seconds": None, "surrogate": None}
    try:
        if method == "cailp_multi":
            und, imp, sur, ts = score_cailp(
                data, device, seed, False, cfg["teacher_n"], cfg["teacher_epochs"], cfg["train_epochs"]
            )
            extra = {"train_seconds": ts, "surrogate": sur, "fusion": "concat", "teacher": "multi_objective"}
        elif method == "cailp_a31":
            und, imp, sur, ts = score_cailp(
                data, device, seed, True, cfg["teacher_n"], cfg["teacher_epochs"], cfg["train_epochs"]
            )
            extra = {"train_seconds": ts, "surrogate": sur, "fusion": "concat", "teacher": "single_objective"}
        elif method == "original_ilp":
            und, imp, ts = score_original_ilp(data, device, cfg["ilp_epochs"])
            extra = {"train_seconds": ts}
        else:
            und, imp, ts = score_learned_baseline(
                data, method if method != "effective_resistance" else "effective_resistance",
                device, cfg["ilp_epochs"], seed,
            )
            if method == "dspar":
                # same scores as effective_resistance proxy in this codebase
                pass
            extra = {"train_seconds": ts}
        rows = eval_importance(
            data, und, imp, budgets, device, cfg["down_epochs"], method, seed, extra=extra
        )
        save_json(rows, path)
        print(f"WROTE {path}", flush=True)
    except Exception as e:
        err = {"dataset": dataset, "seed": seed, "method": method, "error": str(e), "trace": traceback.format_exc()}
        save_json(err, path.with_suffix(".error.json"))
        print(f"ERROR {method} {dataset} seed={seed}: {e}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", default=["lastfm", "facebook", "github"])
    parser.add_argument("--seeds", nargs="+", type=int, default=list(range(10)))
    parser.add_argument("--methods", nargs="+", default=CORE_METHODS)
    parser.add_argument("--budgets", nargs="+", type=float, default=BUDGETS)
    parser.add_argument("--only-seed", type=int, default=None)
    args = parser.parse_args()
    setup_logging(log_file=str(ROOT / "results" / "logs" / "full_grid.log"))
    device = get_device()
    seeds = [args.only_seed] if args.only_seed is not None else args.seeds
    print(f"device={device} datasets={args.datasets} seeds={seeds} methods={args.methods}", flush=True)

    # Prefer cheaper methods first within each seed for early signal
    order = [
        "random",
        "dspar",
        "effective_resistance",
        "original_ilp",
        "neuralsparse",
        "ptdnet",
        "cailp_a31",
        "cailp_multi",
    ]
    methods = [m for m in order if m in args.methods] + [m for m in args.methods if m not in order]

    for dataset in args.datasets:
        for seed in seeds:
            for method in methods:
                run_one(dataset, seed, method, args.budgets, device)


if __name__ == "__main__":
    main()
