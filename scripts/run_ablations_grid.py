#!/usr/bin/env python3
"""Ablations A1–A32 (fusion-fixed for RQ11 / A31).

Runs on LastFM and optionally Facebook. Skips adversarial (A26–A28) as rejected
unless --include-adversarial is set.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.run_full_grid import BUDGETS, eval_importance, load_data, out_path, score_cailp, score_original_ilp
from src.models.cailp import CAILPConfig, CAILPSocial
from src.models.importance_decoder import heteroscedastic_nll, ranking_hinge_loss
from src.utils.graph import undirected_edge_list
from src.utils.io import get_device, save_json, set_seed, setup_logging
import torch
import torch.nn.functional as F


def score_cailp_fusion(data, device, seed, fusion: str, single: bool, teacher_n=50, te=25, tr=25):
    """Like score_cailp but with selectable fusion; A31 comparisons use concat only in grid."""
    from scripts.run_full_grid import pretrain_encoder
    from src.counterfactual.exact_teacher import ExactCounterfactualTeacher
    from src.counterfactual.sampling import stratified_edge_sample
    from src.counterfactual.single_objective_teacher import SingleObjectiveTeacher
    from src.counterfactual.surrogate import SurrogateEdgeImportance, evaluate_surrogate, train_surrogate
    from src.models.edge_encoder import node_centric_edge_representation, structural_edge_features
    import time

    t0 = time.perf_counter()
    enc, clf = pretrain_encoder(data, device, te)
    und = undirected_edge_list(data.edge_index)
    idx = stratified_edge_sample(data, und, n_sample=teacher_n, seed=seed)
    teacher = SingleObjectiveTeacher() if single else ExactCounterfactualTeacher()
    y_cf_np, _ = teacher.score_edges(data, idx, und, enc, clf, device)
    y_cf = torch.tensor(y_cf_np, dtype=torch.float32)
    with torch.no_grad():
        h_t = enc(data.x.to(device), data.edge_index.to(device)).cpu()
    struct = structural_edge_features(data.edge_index, data.num_nodes, und, lightweight=True)
    z_all = node_centric_edge_representation(h_t, und, struct, x=data.x)
    sur = SurrogateEdgeImportance(z_all.size(1), 64)
    train_surrogate(sur, z_all[idx], y_cf, epochs=60)
    with torch.no_grad():
        y_full = sur(z_all).detach().to(device)
        y_hat = sur(z_all[idx]).numpy()
    sur_m = evaluate_surrogate(y_hat, y_cf_np)
    n_cls = int(data.y.max().item()) + 1
    model = CAILPSocial(
        data.x.size(1), n_cls, CAILPConfig(encoder_type="gcn", fusion=fusion, hidden_dim=64)
    ).to(device)
    data_dev = data.clone().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    _ = model(data_dev)
    for _ in range(tr):
        model.train()
        opt.zero_grad()
        out = model(data_dev)
        loss = (
            0.5 * F.cross_entropy(out["logits"][data_dev.train_mask], data_dev.y[data_dev.train_mask])
            + heteroscedastic_nll(out["mu"], out["logvar"], y_full)
            + 0.2 * ranking_hinge_loss(out["mu"], y_full)
        )
        loss.backward()
        opt.step()
    model.eval()
    with torch.no_grad():
        out = model(data_dev)
    return out["undirected_edges"].cpu(), out["mu"].cpu(), sur_m, time.perf_counter() - t0


ABLATIONS = {
    # Core / RQ11 (fusion held to concat for A1 vs A31)
    "A1_full_core": {"kind": "cailp", "fusion": "concat", "single": False},
    "A2_original_ilp": {"kind": "ilp"},
    "A8_concat": {"kind": "cailp", "fusion": "concat", "single": False},
    "A9_gated": {"kind": "cailp", "fusion": "gated", "single": False},
    "A10_cross_attn": {"kind": "cailp", "fusion": "cross_attention", "single": False},
    "A31_single_obj_concat": {"kind": "cailp", "fusion": "concat", "single": True},
    # Architecture sensitivity for single-obj (not used for RQ11 headline decision)
    "A31_single_obj_gated": {"kind": "cailp", "fusion": "gated", "single": True},
    # Optional mass priors (A19–A24 / A32) — edge-score reweight experiments
    "A19_no_mass": {"kind": "cailp", "fusion": "concat", "single": False, "mass": "none"},
    "A23_degree_prior": {"kind": "cailp", "fusion": "concat", "single": False, "mass": "degree"},
    "A24_pagerank_prior": {"kind": "cailp", "fusion": "concat", "single": False, "mass": "pagerank"},
    "A20_analytical_mass": {"kind": "cailp", "fusion": "concat", "single": False, "mass": "analytical"},
    "A32_black_hole_mass": {"kind": "cailp", "fusion": "concat", "single": False, "mass": "black_hole"},
    # Rejected adversarial (record as skipped unless forced)
    "A25_no_adversarial": {"kind": "cailp", "fusion": "concat", "single": False},
    "A26_repr_disc": {"kind": "skip", "reason": "adversarial_rejected_in_pilot"},
    "A27_mask_gan": {"kind": "skip", "reason": "adversarial_rejected_in_pilot"},
    "A28_gan_without_ilp": {"kind": "skip", "reason": "adversarial_rejected_in_pilot"},
}


def apply_mass_prior(data, und, imp, mass_kind: str):
    """Optional edge prior from node mass (Module A). Does not replace ILP scores."""
    if mass_kind in (None, "none"):
        return imp
    import networkx as nx
    from torch_geometric.utils import degree, to_networkx
    from src.models.node_mass import analytical_mass, black_hole_mass_unmodified, gravity_edge_prior, normalize_columns

    deg = degree(data.edge_index[0], num_nodes=data.num_nodes)
    G = to_networkx(data, to_undirected=True)
    if mass_kind == "degree":
        mass = normalize_columns(deg.unsqueeze(-1)).squeeze(-1)
    elif mass_kind == "pagerank":
        pr = nx.pagerank(G)
        mass = torch.tensor([pr[i] for i in range(data.num_nodes)], dtype=torch.float)
        mass = normalize_columns(mass.unsqueeze(-1)).squeeze(-1)
    elif mass_kind == "analytical":
        pr = nx.pagerank(G)
        cl = nx.clustering(G)
        r = torch.stack(
            [
                torch.log1p(deg),
                torch.tensor([pr[i] for i in range(data.num_nodes)], dtype=torch.float),
                torch.tensor([cl[i] for i in range(data.num_nodes)], dtype=torch.float),
            ],
            dim=-1,
        )
        mass = analytical_mass(r)
    elif mass_kind == "black_hole":
        pr = nx.pagerank(G)
        cl = nx.clustering(G)
        mass = black_hole_mass_unmodified(
            deg,
            torch.tensor([pr[i] for i in range(data.num_nodes)], dtype=torch.float),
            torch.tensor([cl[i] for i in range(data.num_nodes)], dtype=torch.float),
        )
    else:
        return imp
    src, dst = und
    # Cosine feature distance as gravity distance
    x = data.x.float()
    xi, xj = x[src], x[dst]
    dist = 1 - (xi * xj).sum(-1) / (xi.norm(dim=-1) * xj.norm(dim=-1) + 1e-8)
    prior = gravity_edge_prior(mass, und, dist.clamp(min=0), p=2.0)
    prior = (prior - prior.min()) / (prior.max() - prior.min() + 1e-8)
    return 0.7 * imp + 0.3 * prior


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", default=["lastfm", "facebook"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    parser.add_argument("--budgets", nargs="+", type=float, default=[0.3, 0.5, 0.7])
    args = parser.parse_args()
    setup_logging(log_file=str(ROOT / "results" / "logs" / "ablation.log"))
    device = get_device()

    for ds in args.datasets:
        for seed in args.seeds:
            set_seed(seed)
            data = load_data(ds, seed)
            data.name = ds
            for name, spec in ABLATIONS.items():
                out = ROOT / "results" / "raw" / "grid" / f"ablation_{ds}_seed{seed}_{name}.json"
                if out.exists():
                    print("SKIP", out.name, flush=True)
                    continue
                print(f"ABLATION {name} {ds} seed={seed}", flush=True)
                if spec["kind"] == "skip":
                    save_json(
                        [{"method": name, "status": "skipped", "reason": spec.get("reason", "")}],
                        out,
                    )
                    continue
                if spec["kind"] == "ilp":
                    und, imp, ts = score_original_ilp(data, device, epochs=20)
                    extra = {"train_seconds": ts, "ablation": name}
                else:
                    und, imp, sur, ts = score_cailp_fusion(
                        data, device, seed, spec["fusion"], spec["single"]
                    )
                    imp = apply_mass_prior(data, und, imp, spec.get("mass", "none"))
                    extra = {
                        "train_seconds": ts,
                        "surrogate": sur,
                        "ablation": name,
                        "fusion": spec["fusion"],
                        "teacher": "single_objective" if spec["single"] else "multi_objective",
                        "mass": spec.get("mass", "none"),
                    }
                rows = eval_importance(
                    data, und, imp, args.budgets, device, 40, name, seed, extra=extra
                )
                save_json(rows, out)


if __name__ == "__main__":
    main()
