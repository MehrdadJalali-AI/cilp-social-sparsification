#!/usr/bin/env python3
"""R1 experiment runner for CILP / Task-only / ablation / weight variants (REVISED protocol).

Phases
------
select : teacher-size grid n ∈ N_GRID for {cilp_full, task_only}; removal 0.5 only; the test mask
         is emptied (no test prediction is computed); reports validation Macro-F1 and held-out
         surrogate/scorer–teacher rank agreement.
final  : requires configs/revised/locked.json (committed before any final run); evaluates all
         budgets 0.1–0.9 with test metrics, structural and community metrics.

Every variant changes only the target y_e built from cached Δ_k(e); splits, seeds, scorer
architecture, concatenation fusion, budgets and downstream evaluation are held constant, and the
surrogate/scorer RNG is reset to the same state for every variant (common random numbers).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from scipy.stats import kendalltau, spearmanr

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.run_full_grid import load_data, pretrain_encoder  # noqa: E402
from src.revision.common import (BUDGETS, EPOCHS, N_GRID, ORIGINAL_TEACHER_N, REVISED_DIR,  # noqa: E402
                                 append_ledger, blind_test, git_commit, hardware, set_seed_all,
                                 teacher_variants, timed)
from src.revision.evaluate import GraphReference, evaluate_scores  # noqa: E402
from src.revision.scorer import fit_scorer, fit_surrogate, teacher_features  # noqa: E402
from src.revision.teacher import composite_target  # noqa: E402
from src.utils.graph import undirected_edge_list  # noqa: E402

CACHE = ROOT / "results" / "cache" / "teacher_deltas"
LOCK = ROOT / "configs" / "revised" / "locked.json"
_REF: Dict[str, GraphReference] = {}


def reference(dataset: str, data) -> GraphReference:
    if dataset not in _REF:
        _REF[dataset] = GraphReference(data, undirected_edge_list(data.edge_index), leiden_seed=0)
        out = REVISED_DIR / "community_reference" / f"{dataset}.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        r = _REF[dataset]
        out.write_text(json.dumps({"algorithm": "Leiden (leidenalg ModularityVertexPartition)", "seed": 0,
                                   "n_communities": r.n_communities, "modularity": r.q0, "conductance": r.phi0,
                                   "n_bridges": r.n_bridges, "minority_class": r.minority_class,
                                   "minority_size": int(r.minority_mask.sum()), "leiden_seconds": r.leiden_seconds},
                                  indent=2))
    return _REF[dataset]


class SeedContext:
    def __init__(self, dataset: str, seed: int):
        self.dataset, self.seed = dataset, seed
        c = np.load(CACHE / f"{dataset}_seed{seed}.npz")
        self.meta = json.loads((CACHE / f"{dataset}_seed{seed}.json").read_text())
        self.edge_ids = c["edge_ids"]
        self.pos = {int(e): i for i, e in enumerate(self.edge_ids)}
        self.samples = {int(k.split("n")[-1]): c[k] for k in c.files if k.startswith("sample_n")}
        self.holdout = c["holdout_ids"]
        self.raw = {k[4:]: c[k] for k in c.files if k.startswith("raw_")}
        self.data = load_data(dataset, seed)
        st = torch.load(CACHE / f"{dataset}_seed{seed}.pt", weights_only=False)
        enc, clf = pretrain_encoder(self.data, torch.device("cpu"), 0)  # architecture only; weights loaded next
        enc.load_state_dict(st["encoder"])
        self.und = undirected_edge_list(self.data.edge_index)
        self.t_feat = {}
        with timed(self.t_feat, "surrogate_features"):
            self.z_all = teacher_features(self.data, enc, self.und)

    def raw_for(self, ids: np.ndarray) -> Dict[str, np.ndarray]:
        j = [self.pos[int(e)] for e in ids]
        return {k: v[j] for k, v in self.raw.items()}


def run_variant(ctx: SeedContext, variant: str, beta: Dict[str, float], n: int, protocol: str,
                budgets: List[float], with_test: bool, community: bool, surrogate_eval: bool) -> Dict:
    ds, seed = ctx.dataset, ctx.seed
    timings: Dict[str, float] = dict(ctx.t_feat)
    ids = ctx.samples[n]
    y = composite_target(ctx.raw_for(ids), beta, protocol)
    set_seed_all(seed)  # common random numbers across variants
    with timed(timings, "surrogate_fit"):
        _, y_full, sur_m = fit_surrogate(ctx.z_all, torch.as_tensor(ids, dtype=torch.long), y, eval_mode=surrogate_eval)
    data = ctx.data if with_test else blind_test(ctx.data)
    mu = fit_scorer(data, y_full, EPOCHS[ds]["train_epochs"], timings)
    # out-of-sample agreement on held-out probed edges (teacher composite normalised over the held-out set)
    y_ho = composite_target(ctx.raw_for(ctx.holdout), beta, protocol)
    ho = torch.as_tensor(ctx.holdout, dtype=torch.long)
    agree = {}
    for name, pred in (("surrogate", y_full[ho].numpy()), ("scorer", mu[ho].numpy())):
        ok = np.std(y_ho) > 0 and np.std(pred) > 0
        agree[f"holdout_{name}_spearman"] = float(spearmanr(pred, y_ho).correlation) if ok else float("nan")
        agree[f"holdout_{name}_kendall"] = float(kendalltau(pred, y_ho).correlation) if ok else float("nan")
    agree["target_degenerate"] = bool(np.std(y) == 0)
    rows = evaluate_scores(data, reference(ds, ctx.data), mu, budgets, EPOCHS[ds]["down_epochs"], seed,
                           with_test=with_test, community=community)
    return {"dataset": ds, "seed": seed, "variant": variant, "beta": beta, "teacher_n": n, "protocol": protocol,
            "surrogate_eval_mode": surrogate_eval, "timings": timings, "surrogate": sur_m, "agreement": agree,
            "y_hash": hashlib.sha1(np.round(y, 12).tobytes()).hexdigest()[:12], "rows": rows}


def out_path(phase: str, r: Dict) -> Path:
    return REVISED_DIR / phase / r["dataset"] / f"seed{r['seed']}" / f"{r['variant']}_n{r['teacher_n']}_{r['protocol']}.json"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", choices=["select", "final"], required=True)
    ap.add_argument("--datasets", nargs="+", default=["lastfm", "facebook", "github"])
    ap.add_argument("--seeds", nargs="+", type=int, default=list(range(10)))
    ap.add_argument("--variants", nargs="+", default=None, help="default: select→cilp_full task_only; final→all")
    ap.add_argument("--n", nargs="+", type=int, default=None, help="final phase: override locked n (sensitivity)")
    ap.add_argument("--original-replay", action="store_true",
                    help="ORIGINAL label access + train-mode surrogate at the ORIGINAL n (attribution runs)")
    ap.add_argument("--threads", type=int, default=0)
    a = ap.parse_args()
    if a.threads:
        torch.set_num_threads(a.threads)
    allv = teacher_variants()
    commit, hw = git_commit(), hardware()
    lock = json.loads(LOCK.read_text()) if LOCK.exists() else None
    if a.phase == "final" and lock is None:
        sys.exit("final phase requires configs/revised/locked.json")
    for ds in a.datasets:
        for seed in a.seeds:
            ctx = None
            names = a.variants or (["cilp_full", "task_only"] if a.phase == "select" else list(allv))
            for v in names:
                if a.phase == "select":
                    n_list, budgets, with_test, community = N_GRID, [0.5], False, False
                elif a.original_replay:
                    n_list, budgets, with_test, community = [ORIGINAL_TEACHER_N[ds]], BUDGETS, True, True
                else:
                    n_list = a.n or [lock["teacher_n"][ds]["task_only" if v == "task_only" else "cilp_full"]]
                    budgets, with_test, community = BUDGETS, True, True
                protocol = "ORIGINAL" if a.original_replay else "REVISED"
                for n in n_list:
                    probe = {"dataset": ds, "seed": seed, "variant": v, "teacher_n": n, "protocol": protocol}
                    path = out_path(a.phase, probe)
                    if path.exists():
                        continue
                    if ctx is None:
                        ctx = SeedContext(ds, seed)
                    t0 = time.perf_counter()
                    res = run_variant(ctx, v, allv[v], n, protocol, budgets, with_test, community,
                                      surrogate_eval=not a.original_replay)
                    res.update({"phase": a.phase, "commit": commit, "hardware": hw, "cache_id": ctx.meta["cache_id"],
                                "wall_clock_s": time.perf_counter() - t0, "timing_grade": False,
                                "lock_sha": lock.get("sha") if lock else None})
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_text(json.dumps(res, indent=1, default=float))
                    append_ledger({"experiment_id": f"{a.phase.upper()}-{ds}-s{seed}-{v}-n{n}-{protocol}",
                                   "purpose": {"select": "teacher-size selection (validation only)",
                                               "final": "locked evaluation"}[a.phase],
                                   "dataset": ds, "protocol": protocol, "config": str(LOCK.relative_to(ROOT)) if lock else "N_GRID",
                                   "seeds": [seed], "budgets": budgets, "teacher_n": n, "beta": allv[v],
                                   "cache_id": ctx.meta["cache_id"], "commit": commit, "hardware": hw,
                                   "threads": torch.get_num_threads(), "wall_clock_s": res["wall_clock_s"],
                                   "peak_memory_mb": res["timings"].get("scorer_fit_peak_rss_mb"),
                                   "status": "completed", "output": str(path.relative_to(ROOT))})
                    vf = [r["val_macro_f1"] for r in res["rows"]]
                    print(f"{a.phase} {ds} s{seed} {v} n={n} {protocol} valF1={np.mean(vf):.4f} "
                          f"rho_ho={res['agreement']['holdout_scorer_spearman']:.3f} {res['wall_clock_s']:.0f}s", flush=True)


if __name__ == "__main__":
    main()
