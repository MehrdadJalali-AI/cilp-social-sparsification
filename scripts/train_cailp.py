#!/usr/bin/env python3
"""Train core CAILP-Social and evaluate under Track A budgets."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.counterfactual.exact_teacher import ExactCounterfactualTeacher
from src.counterfactual.sampling import stratified_edge_sample
from src.counterfactual.single_objective_teacher import SingleObjectiveTeacher
from src.counterfactual.surrogate import SurrogateEdgeImportance, evaluate_surrogate, train_surrogate
from src.evaluation.centrality import centrality_preservation
from src.evaluation.community import community_metrics
from src.evaluation.fairness import fairness_metrics
from src.evaluation.structural import structural_metrics
from src.models.cailp import CAILPConfig, CAILPSocial
from src.models.node_encoders import NodeEncoder
from src.sparsification.constrained_pruning import PruningConstraints, budget_prune_unconstrained, constrained_prune
from src.tasks.node_classification import train_node_classifier
from src.utils.graph import subgraph_from_undirected_edges, undirected_edge_list
from src.utils.io import get_device, load_yaml, save_json, set_seed, setup_logging
from src.utils.splits import LeakageGuard


def pretrain_teacher_encoder(data, device, epochs: int = 40):
    n_cls = int(data.y.max().item()) + 1
    enc = NodeEncoder(data.x.size(1), encoder_type="gcn", hidden_dim=64, out_dim=64).to(device)
    clf = nn.Linear(64, n_cls).to(device)
    opt = torch.optim.Adam(list(enc.parameters()) + list(clf.parameters()), lr=1e-2)
    x, y, ei = data.x.to(device), data.y.to(device), data.edge_index.to(device)
    for _ in range(epochs):
        enc.train()
        clf.train()
        opt.zero_grad()
        h = enc(x, ei)
        loss = F.cross_entropy(clf(h)[data.train_mask], y[data.train_mask])
        loss.backward()
        opt.step()
    return enc, clf


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="lastfm")
    parser.add_argument("--config", default="configs/experiments/pilot.yaml")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--single-objective", action="store_true", help="A31 fidelity-only teacher")
    args = parser.parse_args()
    cfg = load_yaml(ROOT / args.config)
    seed = args.seed if args.seed is not None else int(cfg.get("seed", 0))
    setup_logging(log_file=str(ROOT / "results" / "logs" / f"train_cailp_{args.dataset}_seed{seed}.log"))
    set_seed(seed)
    device = get_device()

    data = torch.load(ROOT / "data" / "processed" / f"{args.dataset}.pt", weights_only=False)
    split = torch.load(ROOT / "data" / "splits" / f"{args.dataset}_seed{seed}.pt", weights_only=False)
    data.train_mask = split["train_mask"]
    data.val_mask = split["val_mask"]
    data.test_mask = split["test_mask"]

    guard = LeakageGuard()
    guard.set_phase("sparsification")
    guard.lock_test()

    print("Pretraining teacher encoder...", flush=True)
    enc, clf = pretrain_teacher_encoder(data, device, epochs=int(cfg["teacher"]["epochs_downstream"]))
    und = undirected_edge_list(data.edge_index)
    idx = stratified_edge_sample(data, und, n_sample=int(cfg["teacher"]["n_sample"]), seed=seed)
    print(f"Teacher edges: {len(idx)}", flush=True)

    single = args.single_objective or bool(cfg["teacher"].get("single_objective", False))
    teacher = SingleObjectiveTeacher() if single else ExactCounterfactualTeacher()
    print("Scoring counterfactual teacher...", flush=True)
    y_cf_np, effects = teacher.score_edges(data, idx, und, enc, clf, device)
    y_cf = torch.tensor(y_cf_np, dtype=torch.float32)
    print("Teacher done.", flush=True)

    # Surrogate on node-centric structural features for remaining edges
    from src.models.edge_encoder import structural_edge_features, node_centric_edge_representation

    print("Building surrogate features...", flush=True)
    with torch.no_grad():
        h_t = enc(data.x.to(device), data.edge_index.to(device)).cpu()
    struct = structural_edge_features(data.edge_index, data.num_nodes, und, lightweight=True)
    z_all = node_centric_edge_representation(h_t, und, struct, x=data.x)
    sur = SurrogateEdgeImportance(z_all.size(1), 64)
    train_surrogate(sur, z_all[idx], y_cf, epochs=80)
    with torch.no_grad():
        y_hat = sur(z_all[idx]).numpy()
    sur_metrics = evaluate_surrogate(y_hat, y_cf_np)
    with torch.no_grad():
        y_full = sur(z_all).detach()

    # Train CAILP to fit surrogate targets + task
    n_cls = int(data.y.max().item()) + 1
    model_cfg = CAILPConfig(
        encoder_type=cfg["encoder"]["type"],
        hidden_dim=cfg["encoder"]["hidden_dim"],
        num_layers=cfg["encoder"]["num_layers"],
        dropout=cfg["encoder"]["dropout"],
        heads=cfg["encoder"].get("heads", 4),
        fusion=cfg.get("fusion", "cross_attention"),
        lambda_cf=cfg["train"]["lambda_cf"],
        lambda_rank=cfg["train"]["lambda_rank"],
        lambda_task=cfg["train"]["lambda_task"],
    )
    model = CAILPSocial(data.x.size(1), n_cls, model_cfg).to(device)
    # Move data tensors
    data_dev = data.clone()
    data_dev = data_dev.to(device)
    # masks stay bool on device
    opt = torch.optim.Adam(model.parameters(), lr=float(cfg["train"]["lr"]))
    y_full = y_full.to(device)
    # Warm-up forward to build lazy modules
    model.train()
    _ = model(data_dev)
    for epoch in range(int(cfg["train"]["epochs"])):
        model.train()
        opt.zero_grad()
        out = model(data_dev)
        # Use all-edge surrogate targets
        from src.models.importance_decoder import heteroscedastic_nll, ranking_hinge_loss

        loss_task = F.cross_entropy(out["logits"][data_dev.train_mask], data_dev.y[data_dev.train_mask])
        loss_cf = heteroscedastic_nll(out["mu"], out["logvar"], y_full)
        loss_rank = ranking_hinge_loss(out["mu"], y_full, margin=model_cfg.ranking_margin)
        loss = (
            model_cfg.lambda_task * loss_task
            + model_cfg.lambda_cf * loss_cf
            + model_cfg.lambda_rank * loss_rank
        )
        loss.backward()
        opt.step()
        if epoch % 10 == 0:
            print(f"epoch={epoch} loss={loss.item():.4f} task={loss_task.item():.4f} cf={loss_cf.item():.4f}")

    model.eval()
    with torch.no_grad():
        out = model(data_dev)
        mu = out["mu"].cpu()
        und_cpu = out["undirected_edges"].cpu()

    results = []
    skip_constrained = data.num_nodes > 15000 or und_cpu.size(1) > 80000
    if skip_constrained:
        print("Skipping constrained prune on large graph (budget-only).", flush=True)
    for removal in cfg.get("budgets", [0.3, 0.5, 0.7]):
        # Equal-budget unconstrained ranking for main comparison + constrained variant
        keep = budget_prune_unconstrained(und_cpu, mu, removal)
        sparse = subgraph_from_undirected_edges(data, keep)
        sparse.train_mask, sparse.val_mask, sparse.test_mask = data.train_mask, data.val_mask, data.test_mask

        variants = [("budget", sparse)]
        if not skip_constrained:
            cons = constrained_prune(
                data,
                und_cpu,
                mu,
                removal_rate=removal,
                constraints=PruningConstraints(),
            )
            sparse_c = subgraph_from_undirected_edges(data, cons.keep_undirected)
            sparse_c.train_mask, sparse_c.val_mask, sparse_c.test_mask = data.train_mask, data.val_mask, data.test_mask
            variants.append(("constrained", sparse_c))
        else:
            cons = type("C", (), {"budget_shortfall": False})()

        guard.unlock_test_for_eval()
        row_base = {
            "method": "cailp_single_obj" if single else "cailp_social",
            "removal_rate": removal,
            "surrogate": sur_metrics,
            "teacher": "single_objective" if single else "multi_objective",
            "effects_sample_means": {k: float(sum(v) / max(len(v), 1)) for k, v in effects.as_dict().items()},
        }
        for tag, g in variants:
            metrics_all = {}
            for kind in cfg["downstream"]["kinds"]:
                _, m = train_node_classifier(
                    g, kind=kind, epochs=int(cfg["downstream"]["epochs"]), device=device
                )
                metrics_all[kind] = m
            struct = structural_metrics(data, g)
            # Skip expensive community/centrality on every prune mode in pilot; compute once for budget mode
            if tag == "budget" and data.num_nodes <= 15000:
                comm = community_metrics(data, g, seed=seed)
                cent = centrality_preservation(data, g)
                fair = fairness_metrics(data, g)
            else:
                fair = fairness_metrics(data, g) if tag == "budget" else {}
                comm, cent = {}, {}
            results.append(
                {
                    **row_base,
                    "prune_mode": tag,
                    "budget_shortfall": cons.budget_shortfall if tag == "constrained" else False,
                    "metrics": metrics_all,
                    "structural": struct,
                    "community": comm,
                    "centrality": cent,
                    "fairness": fair,
                    "leakage": guard.summary(),
                }
            )
            print(tag, removal, metrics_all["gcn"]["test_macro_f1"])
        guard.lock_test()
        guard.set_phase("sparsification")

    # checkpoint
    ckpt = ROOT / "results" / "raw" / f"cailp_{args.dataset}_seed{seed}{'_a31' if single else ''}.pt"
    torch.save({"model": model.state_dict(), "cfg": cfg, "seed": seed}, ckpt)
    out_json = ROOT / "results" / "raw" / f"cailp_{args.dataset}_seed{seed}{'_a31' if single else ''}.json"
    save_json(results, out_json)
    print("Wrote", out_json)


if __name__ == "__main__":
    main()
