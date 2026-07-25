# GitHub frozen-protocol audit (Stage A)

**Date:** 2026-07-25  
**Runner:** `scripts/run_full_grid.py`  
**Status:** Additive GitHub completion only — no methodological changes.

## 1. Dataset loader

- Registry: `src/data/datasets.py` key `"github"`
- Kind: `snap_zip`
- URL: `https://snap.stanford.edu/data/git_web_ml.zip`
- Edges/features/target: `git_web_ml/musae_git_*.csv/json`
- Processed artifact: `data/processed/github.pt`

## 2. Preprocessing path

`src/data/datasets.py::preprocess_graph` via download/audit pipeline:

- undirected conversion / symmetrization;
- self-loop removal;
- duplicate undirected-edge removal;
- multi-hot bag-of-words feature expansion;
- official binary class labels retained;
- scientific edge set = unique undirected edges;
- PyG `edge_index` stores both directions.

## 3. Verified processed statistics

| Statistic | Expected | Observed |
|---|---|---|
| Nodes | 37,700 | **37,700** |
| Unique undirected edges | 289,003 | **289,003** |
| Classes | 2 | **2** |
| Feature dim | ~4,005 | **4,005** |
| Self-loops in tensor | 0 | **0** |
| Raw directed records | — | 578,006 (= 2×289,003) |

**No discrepancy — grid permitted.**

## 4. Split-generation path

- Utility: `scripts/build_splits.py`
- Artifacts: `data/splits/github_seed{0..9}.pt`
- Policy: stratified 60/20/20
- Seed 0 fractions verified: 0.6 / 0.2 / 0.2; train/val/test disjoint and cover all nodes

## 5. CILP configuration (`GRID_CFG["github"]`)

| Parameter | Value | Source |
|---|---|---|
| teacher_n | 40 | GRID_CFG |
| teacher_epochs | 15 | GRID_CFG |
| train_epochs (scorer) | 15 | GRID_CFG |
| down_epochs | 35 | GRID_CFG |
| ilp_epochs | 15 | GRID_CFG |
| encoder | GCN, 2 layers, hidden 64, dropout 0.4 | `score_cailp` / CAILPConfig |
| fusion | concat | hardcoded |
| optimizer / lr | Adam / 1e-3 | `score_cailp` |
| β | (1.0, 0.5, 1.0, 0.3, 0.5, 0.5) | `CFCoefficients` defaults |
| λCE / λNLL / λrank | 0.5 / 1.0 / 0.2 | `score_cailp` |
| ranking margin / pairs | 0.1 / 256 | `ranking_hinge_loss` defaults |
| pruning | `budget_prune_unconstrained` by μ | eval path |
| device | CPU (`get_device` skips MPS) | `src/utils/io.py` |

**No code–document conflicts found.**

## 6. Baseline configurations

Same runner methods: `random`, `original_ilp`, `neuralsparse`, `ptdnet`, `dspar`, `effective_resistance` (Resistance proxy; deduped later), `cailp_multi`, `cailp_a31`.

## 7. Result-output schema

- Path: `results/raw/grid/{dataset}_seed{seed}_{method}.json`
- Content: list of per-budget dicts (length ≥ 9 when complete)
- Resume: `already_done` skips if ≥ 9 budget rows present

## 8. Validation / leakage

- Teacher CE on train∪val only (`exact_teacher.py`)
- Scorer / downstream use train (+ val early-select patterns as in existing code)
- Test used in final Macro-F1 evaluation only
- Same masks for all methods per seed

## 9. Exact commands

One seed (all methods, all budgets):

```bash
python scripts/run_full_grid.py --datasets github --only-seed 0
```

Diagnostic (seed 0, budget 0.5 only):

```bash
python scripts/run_full_grid.py --datasets github --only-seed 0 --budgets 0.5
```

All seeds:

```bash
python scripts/run_full_grid.py --datasets github --seeds 0 1 2 3 4 5 6 7 8 9
```

## 10. Pre-Stage-A inventory

- Authoritative GitHub rows: **0**
- Raw grid GitHub files: **0**
- LastFM + Facebook authoritative rows: **1260** (must remain unchanged)
