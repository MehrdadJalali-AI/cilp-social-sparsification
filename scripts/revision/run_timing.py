#!/usr/bin/env python3
"""Step 3 — timing-grade cost decomposition (run SOLO, nothing else on the machine).

Per (dataset, seed), from scratch (no cache), for the locked CILP and Task-only configurations:
  teacher construction = encoder pre-training + edge sampling + graph context (bridges, Louvain)
                         + n counterfactual probes + criterion aggregation
  scorer fitting       = teacher features + surrogate fit + scorer view build + scorer epochs
  ranking/sparsify     = scorer inference + exact-budget argsort + subgraph construction (per budget)
  downstream training  = 2-layer GCN, fixed epochs, incl. per-epoch validation pass (per budget and r = 0)
  downstream inference = one full forward pass, median of 5 (per budget and r = 0)
Peak memory = maximum process RSS during each stage (CPU only). Messages = 2|E_r| + |V| per layer.
Baseline scorer fit times are measured in the same process for comparison.
The probed Δ values are checked against the cache (same seed ⇒ identical teacher).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.revision.run_baselines import score as baseline_score  # noqa: E402
from scripts.revision.run_experiment import LOCK  # noqa: E402
from scripts.run_full_grid import load_data, pretrain_encoder  # noqa: E402
from src.counterfactual.sampling import stratified_edge_sample  # noqa: E402
from src.revision.common import (BUDGETS, EPOCHS, REVISED_DIR, PeakRSS, append_ledger, gcn_messages,  # noqa: E402
                                 git_commit, hardware, set_seed_all, teacher_variants, timed)
from src.revision.evaluate import downstream  # noqa: E402
from src.revision.scorer import fit_scorer, fit_surrogate, teacher_features  # noqa: E402
from src.revision.teacher import composite_target, compute_deltas, graph_context  # noqa: E402
from src.sparsification.constrained_pruning import budget_prune_unconstrained  # noqa: E402
from src.utils.graph import subgraph_from_undirected_edges, undirected_edge_list  # noqa: E402
from src.utils.io import set_seed  # noqa: E402

BASELINES = ["random", "original_ilp", "neuralsparse", "ptdnet", "resistance_proxy",
             "gassip_adapted", "mog_adapted", "psgnn_reimpl", "nk_local_degree", "nk_local_similarity"]


def time_teacher_pipeline(ds: str, seed: int, variant: str, n: int) -> dict:
    T: dict = {}
    t_all = time.perf_counter()
    set_seed(seed)
    data = load_data(ds, seed)
    with PeakRSS() as mem:
        with timed(T, "teacher_pretrain"):
            enc, clf = pretrain_encoder(data, torch.device("cpu"), EPOCHS[ds]["teacher_epochs"])
        und = undirected_edge_list(data.edge_index)
        with timed(T, "teacher_sampling"):
            ids = stratified_edge_sample(data, und, n_sample=n, seed=seed).numpy()
        with timed(T, "teacher_graph_context"):
            ctx = graph_context(data, louvain_seed=seed)
        with timed(T, "teacher_probes"):
            raw, per = compute_deltas(data, enc, clf, und, ids.tolist(), ctx)
        with timed(T, "teacher_aggregate"):
            y = composite_target(raw, teacher_variants()[variant], "REVISED")
    T["teacher_peak_rss_mb"] = mem.peak_mb
    T["teacher_probe_components"] = per
    T["teacher_total"] = sum(T[k] for k in ("teacher_pretrain", "teacher_sampling", "teacher_graph_context",
                                            "teacher_probes", "teacher_aggregate"))
    # reproducibility check against the cache
    c = np.load(ROOT / "results" / "cache" / "teacher_deltas" / f"{ds}_seed{seed}.npz")
    pos = {int(e): i for i, e in enumerate(c["edge_ids"])}
    j = [pos[int(e)] for e in ids]
    T["cache_max_abs_diff_task_rev"] = float(np.max(np.abs(c["raw_task_rev"][j] - raw["task_rev"])))
    set_seed_all(seed)
    with timed(T, "scorer_teacher_features"):
        z_all = teacher_features(data, enc, und)
    with timed(T, "scorer_surrogate_fit"):
        _, y_full, _ = fit_surrogate(z_all, torch.as_tensor(ids, dtype=torch.long), y, eval_mode=True)
    mu = fit_scorer(data, y_full, EPOCHS[ds]["train_epochs"], T)  # adds scorer_view_build/fit/rank_inference
    T["scorer_total"] = (T["scorer_teacher_features"] + T["scorer_surrogate_fit"] + T["scorer_view_build"]
                         + T["scorer_fit"])
    T["one_time_total"] = T["teacher_total"] + T["scorer_total"] + T["scorer_rank_inference"]
    per_budget = []
    for r in BUDGETS:
        t = time.perf_counter()
        keep = budget_prune_unconstrained(und, mu, r)
        sparse = subgraph_from_undirected_edges(data, keep)
        prune = time.perf_counter() - t
        d = downstream(sparse, EPOCHS[ds]["down_epochs"], seed * 1000 + int(round(r * 100)), with_test=False)
        per_budget.append({"removal_rate": r, "sparsify_seconds": prune, "undirected_edges": int(keep.size(1)),
                           "messages_per_layer": gcn_messages(keep.size(1), data.num_nodes),
                           **{k: v for k, v in d.items() if "seconds" in k or "rss" in k}})
    full = downstream(data, EPOCHS[ds]["down_epochs"], seed * 1000, with_test=False)
    T["wall_clock"] = time.perf_counter() - t_all
    return {"timings": T, "per_budget": per_budget,
            "full_graph": {"undirected_edges": int(und.size(1)), "messages_per_layer": gcn_messages(und.size(1), data.num_nodes),
                           **{k: v for k, v in full.items() if "seconds" in k or "rss" in k}}}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=["lastfm", "facebook", "github", "pubmed"])
    ap.add_argument("--seeds", nargs="+", type=int, default=list(range(10)))
    ap.add_argument("--threads", type=int, default=6)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    lock = json.loads(LOCK.read_text())
    commit, hw = git_commit(), hardware()
    for ds in a.datasets:
        for seed in a.seeds:
            out = REVISED_DIR / "timing" / ds / f"seed{seed}.json"
            if out.exists():
                continue
            res = {"dataset": ds, "seed": seed, "commit": commit, "hardware": hw, "threads": a.threads,
                   "timing_grade": True, "lock_sha": lock["sha"]}
            for v in ("cilp_full", "task_only"):
                n = lock["teacher_n"][ds][v]
                res[v] = {"teacher_n": n, **time_teacher_pipeline(ds, seed, v, n)}
            data = load_data(ds, seed)
            data.name = ds
            res["baselines"] = {}
            for m in BASELINES:
                set_seed_all(seed)
                with PeakRSS() as mem:
                    t = time.perf_counter()
                    baseline_score(m, data, seed)
                    s = time.perf_counter() - t
                res["baselines"][m] = {"scorer_fit_seconds": s, "peak_rss_mb": mem.peak_mb}
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(json.dumps(res, indent=1, default=float))
            append_ledger({"experiment_id": f"TIMING-{ds}-s{seed}", "purpose": "timing-grade cost decomposition (solo)",
                           "dataset": ds, "protocol": "REVISED", "config": str(LOCK.relative_to(ROOT)), "seeds": [seed],
                           "budgets": BUDGETS, "teacher_n": {v: res[v]["teacher_n"] for v in ("cilp_full", "task_only")},
                           "beta": "cilp_full / task_only", "cache_id": None, "commit": commit, "hardware": hw,
                           "threads": a.threads, "wall_clock_s": res["cilp_full"]["timings"]["wall_clock"] + res["task_only"]["timings"]["wall_clock"],
                           "peak_memory_mb": res["cilp_full"]["timings"].get("scorer_fit_peak_rss_mb"),
                           "status": "completed", "output": str(out.relative_to(ROOT))})
            print(f"timing {ds} s{seed} CILP one-time {res['cilp_full']['timings']['one_time_total']:.1f}s "
                  f"teacher {res['cilp_full']['timings']['teacher_total']:.1f}s", flush=True)


if __name__ == "__main__":
    main()
