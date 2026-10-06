#!/usr/bin/env python3
"""Step 0.3 — cache raw per-edge criteria Δ_k(e) for every dataset and seed.

For each (dataset, seed) the cache holds raw Δ for the union of the stratified teacher samples
for n ∈ N_GRID ∪ {ORIGINAL n} plus 200 uniformly drawn held-out probe edges (disjoint from the
teacher pool) used only to measure out-of-sample surrogate/scorer–teacher rank agreement.
No test label is read. Ablations and weight variants re-use these values and only change y_e.
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

from scripts.run_full_grid import load_data, pretrain_encoder  # noqa: E402
from src.counterfactual.sampling import stratified_edge_sample  # noqa: E402
from src.revision.common import (CACHE_DIR, EPOCHS, HOLDOUT_PROBES, N_GRID, ORIGINAL_TEACHER_N,  # noqa: E402
                                 append_ledger, git_commit, hardware, sha_of)
from src.revision.teacher import compute_deltas, graph_context  # noqa: E402
from src.utils.graph import undirected_edge_list  # noqa: E402
from src.utils.io import set_seed  # noqa: E402


def cache_paths(dataset: str, seed: int) -> tuple[Path, Path, Path]:
    base = CACHE_DIR / f"{dataset}_seed{seed}"
    return base.with_suffix(".npz"), base.with_suffix(".pt"), base.with_suffix(".json")


def build(dataset: str, seed: int) -> None:
    npz, pt, meta_p = cache_paths(dataset, seed)
    if meta_p.exists():
        print(f"SKIP {meta_p.name}", flush=True)
        return
    t_all = time.perf_counter()
    timings = {}
    set_seed(seed)  # same RNG entry point as the ORIGINAL run_one()
    data = load_data(dataset, seed)
    t = time.perf_counter()
    enc, clf = pretrain_encoder(data, torch.device("cpu"), EPOCHS[dataset]["teacher_epochs"])
    timings["teacher_pretrain"] = time.perf_counter() - t
    und = undirected_edge_list(data.edge_index)
    sizes = sorted(set(N_GRID) | ({ORIGINAL_TEACHER_N[dataset]} if dataset in ORIGINAL_TEACHER_N else set()))
    samples = {}
    t = time.perf_counter()
    for n in sizes:
        samples[n] = stratified_edge_sample(data, und, n_sample=n, seed=seed).numpy()
    timings["teacher_sampling_all_n"] = time.perf_counter() - t
    pool = np.unique(np.concatenate(list(samples.values())))
    rng = np.random.RandomState(10_000 + seed)
    holdout = rng.choice(np.setdiff1d(np.arange(und.size(1)), pool), size=HOLDOUT_PROBES, replace=False)
    edge_ids = np.concatenate([pool, np.sort(holdout)])

    ctx = graph_context(data, louvain_seed=seed)
    timings.update({f"ctx_{k}": v for k, v in ctx.seconds.items()})
    raw, sec = compute_deltas(data, enc, clf, und, edge_ids.tolist(), ctx)
    timings.update({f"delta_{k}_total": v for k, v in sec.items()})
    timings["n_probed_edges"] = int(len(edge_ids))
    timings["per_edge_seconds"] = {k: v / len(edge_ids) for k, v in sec.items()}
    timings["cache_build_total"] = time.perf_counter() - t_all

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(npz, edge_ids=edge_ids, holdout_ids=np.sort(holdout),
                        **{f"sample_n{n}": s for n, s in samples.items()},
                        **{f"raw_{k}": v for k, v in raw.items()})
    torch.save({"encoder": enc.state_dict(), "classifier": clf.state_dict()}, pt)
    meta = {
        "dataset": dataset, "seed": seed, "sizes": sizes, "holdout": HOLDOUT_PROBES,
        "n_pool": int(len(pool)), "label_access": "train (task_rev, group_rev); train∪val (task_orig) and val (group_orig) cached for provenance only; test never read",
        "louvain_seed": seed, "louvain_q0": ctx.q0, "n_bridges": len(ctx.bridges),
        "frac_nonzero": {k: float((v != 0).mean()) for k, v in raw.items()},
        "timings": timings, "commit": git_commit(), "hardware": hardware(),
    }
    meta["cache_id"] = sha_of({k: meta[k] for k in ("dataset", "seed", "sizes", "holdout", "commit")})
    meta_p.write_text(json.dumps(meta, indent=2, default=str))
    append_ledger({"experiment_id": f"CACHE-{dataset}-s{seed}", "purpose": "Δ_k cache (Step 0.3)",
                   "dataset": dataset, "protocol": "REVISED (train-only targets); ORIGINAL Δ cached for provenance",
                   "seeds": [seed], "budgets": None, "teacher_n": sizes, "beta": None,
                   "cache_id": meta["cache_id"], "commit": meta["commit"], "hardware": meta["hardware"],
                   "wall_clock_s": timings["cache_build_total"], "peak_memory_mb": None,
                   "status": "completed", "output": str(npz.relative_to(ROOT))})
    print(f"WROTE {npz.name} pool={len(pool)} total={timings['cache_build_total']:.1f}s "
          f"per-edge={timings['per_edge_seconds']}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=["lastfm", "facebook", "github"])
    ap.add_argument("--seeds", nargs="+", type=int, default=list(range(10)))
    ap.add_argument("--threads", type=int, default=0)
    a = ap.parse_args()
    if a.threads:
        torch.set_num_threads(a.threads)
    for d in a.datasets:
        for s in a.seeds:
            build(d, s)


if __name__ == "__main__":
    main()
