#!/usr/bin/env python3
"""Baselines under the same evaluation as run_experiment.py (final phase only).

Methods
-------
full_graph            r = 0 reference (no edge removed)
random                seeded uniform scores
original_ilp          ILP-GCN (src/sparsification/original_ilp.py)
neuralsparse, ptdnet  simplified matched-budget reimplementations (NOT faithful reproductions)
resistance_proxy      degree-based effective-resistance-inspired score (NOT DSpar, NOT exact ER)
gassip_adapted, mog_adapted, psgnn_reimpl   disclosed adaptations
nk_local_degree       NetworKit LocalDegreeScore  (Hamann et al. 2016)          — verified library
nk_local_similarity   NetworKit LocalSimilarityScore (Satuluri et al. 2011)     — verified library
                      NetworKit scores are the largest threshold at which an edge survives
                      GlobalThresholdFilter(above=True); exact budgets are imposed by ranking the
                      scores (descending) with seeded random tie-breaking.
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

from scripts.pipeline.run_experiment import LOCK, reference  # noqa: E402
from scripts.run_full_grid import load_data, score_learned_baseline, score_original_ilp  # noqa: E402
from src.pipeline.common import (BUDGETS, EPOCHS, RESULTS_DIR, PeakRSS, append_ledger, git_commit,  # noqa: E402
                                 hardware, set_seed_all)
from src.pipeline.evaluate import evaluate_scores  # noqa: E402
from src.sparsification.baselines.gassip_adapted import gassip_adapted_score_edges  # noqa: E402
from src.sparsification.baselines.mog_adapted import mog_adapted_score_edges  # noqa: E402
from src.sparsification.baselines.psgnn_reimpl import psgnn_reimpl_score_edges  # noqa: E402
from src.utils.graph import undirected_edge_list  # noqa: E402

METHODS = ["full_graph", "random", "original_ilp", "neuralsparse", "ptdnet", "resistance_proxy",
           "gassip_adapted", "mog_adapted", "psgnn_reimpl", "nk_local_degree", "nk_local_similarity"]


def networkit_scores(und: torch.Tensor, n: int, kind: str, seed: int) -> torch.Tensor:
    import subprocess
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        np.save(f"{tmp}/und.npy", und.numpy())
        subprocess.run([sys.executable, "-m", "src.pipeline.nk_scores", f"{tmp}/und.npy", str(n), kind,
                        f"{tmp}/score.npy"], cwd=ROOT, check=True)
        score = np.load(f"{tmp}/score.npy")
    tie = np.random.RandomState(seed).rand(len(score))
    order = np.lexsort((tie, score))  # ascending by score, ties broken by seeded uniform
    rank = np.empty(len(score))
    rank[order] = np.arange(len(score))
    return torch.from_numpy(rank).float()  # higher = keep


def score(method: str, data, seed: int):
    ds = data.name
    if method == "original_ilp":
        und, imp, _ = score_original_ilp(data, torch.device("cpu"), EPOCHS[ds]["ilp_epochs"])
        return und, imp
    if method in ("random", "neuralsparse", "ptdnet"):
        und, imp, _ = score_learned_baseline(data, method, torch.device("cpu"), EPOCHS[ds]["ilp_epochs"], seed)
        return und, imp
    if method == "resistance_proxy":
        und, imp, _ = score_learned_baseline(data, "dspar", torch.device("cpu"), EPOCHS[ds]["ilp_epochs"], seed)
        return und, imp
    if method in ("gassip_adapted", "mog_adapted", "psgnn_reimpl"):
        fn = {"gassip_adapted": gassip_adapted_score_edges, "mog_adapted": mog_adapted_score_edges,
              "psgnn_reimpl": psgnn_reimpl_score_edges}[method]
        und, imp, _ = fn(data, epochs=EPOCHS[ds]["train_epochs"], device=torch.device("cpu"))
        return und, imp
    if method.startswith("nk_"):
        und = undirected_edge_list(data.edge_index)
        return und, networkit_scores(und, data.num_nodes, method, seed)
    raise ValueError(method)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=["lastfm", "facebook", "github"])
    ap.add_argument("--seeds", nargs="+", type=int, default=list(range(10)))
    ap.add_argument("--methods", nargs="+", default=METHODS)
    ap.add_argument("--threads", type=int, default=0)
    a = ap.parse_args()
    if not LOCK.exists():
        sys.exit("baselines run only after configs/selected/locked.json is committed")
    if a.threads:
        torch.set_num_threads(a.threads)
    lock = json.loads(LOCK.read_text())
    commit, hw = git_commit(), hardware()
    for ds in a.datasets:
        for seed in a.seeds:
            data = None
            for m in a.methods:
                path = RESULTS_DIR / "final" / ds / f"seed{seed}" / f"baseline_{m}.json"
                if path.exists():
                    continue
                if data is None:
                    data = load_data(ds, seed)
                    data.name = ds
                ref = reference(ds, data)
                t0 = time.perf_counter()
                timings = {}
                set_seed_all(seed)
                if m == "full_graph":
                    imp, budgets = None, [0.0]
                else:
                    with PeakRSS() as mem:
                        t = time.perf_counter()
                        und, imp = score(m, data, seed)
                        timings["scorer_fit"] = time.perf_counter() - t
                    timings["scorer_fit_peak_rss_mb"] = mem.peak_mb
                    assert torch.equal(und.cpu(), ref.und), f"{m}: edge order differs from evaluator"
                    imp, budgets = imp.detach().cpu().float(), BUDGETS
                rows = evaluate_scores(data, ref, imp, budgets, EPOCHS[ds]["down_epochs"], seed,
                                       with_test=True, community=True)
                res = {"dataset": ds, "seed": seed, "variant": f"baseline_{m}", "protocol": "train",
                       "phase": "final", "timings": timings, "rows": rows, "commit": commit, "hardware": hw,
                       "wall_clock_s": time.perf_counter() - t0, "timing_grade": False, "lock_sha": lock.get("sha")}
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(json.dumps(res, indent=1, default=float))
                append_ledger({"experiment_id": f"FINAL-{ds}-s{seed}-baseline_{m}", "purpose": "baseline evaluation",
                               "dataset": ds, "protocol": "train", "config": "scripts/pipeline/run_baselines.py",
                               "seeds": [seed], "budgets": budgets, "teacher_n": None, "beta": None, "cache_id": None,
                               "commit": commit, "hardware": hw, "threads": torch.get_num_threads(),
                               "wall_clock_s": res["wall_clock_s"], "peak_memory_mb": timings.get("scorer_fit_peak_rss_mb"),
                               "status": "completed", "output": str(path.relative_to(ROOT))})
                print(f"baseline {ds} s{seed} {m} {res['wall_clock_s']:.0f}s "
                      f"valF1@.5={[r['val_macro_f1'] for r in rows if abs(r['removal_rate']-0.5)<1e-9 or len(rows)==1][0]:.4f}",
                      flush=True)


if __name__ == "__main__":
    main()
