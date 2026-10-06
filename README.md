# CILP — Counterfactual Inverse Link Prediction for Graph Sparsification

Public **code and experimental results** for exact-budget graph sparsification via Counterfactual Inverse Link Prediction (CILP): the original Stage A results and the R1 revision (corrected protocol), which are reported as primary evidence in the revised manuscript.

**Method name:** CILP (multi-criteria counterfactual teacher)  
**Matched ablation:** Task-only  
**Parent method:** ILP-GCN (Bangian Tabrizi, Jalali & Houshmand, *Journal of Big Data*, 2025)

This repository does **not** include the journal manuscript sources.

## Repository contents

| Path | Description |
|---|---|
| `src/` | CILP teacher, scorer, pruning, baselines |
| `scripts/` | Data prep, full grid, evaluation, CSV table export |
| `configs/` | Experiment configs |
| `data/splits/` | Fixed train/validation/test masks (seeds 0–9; LastFM, Facebook, GitHub, PubMed) |
| `results/processed/` | Authoritative Stage A store (1,890 rows) |
| `results/raw/grid/` | Per-seed raw JSON outputs |
| `results/tables/` | Aggregated CSV tables (Stage A) |
| `results/processed/recent_baselines/` | Stage A results for GASSIP-adapted, MoG-adapted, PSGNN-reimpl (810 rows) |
| `src/revision/`, `scripts/revision/` | R1 revision pipeline (criterion cache, corrected protocol, ablations, timing, statistics) |
| `results/revised/` | R1 results: per-run JSON (`final/`, `select/`, `timing/`), tables, experiment ledger |
| `docs/revision/` | R1 audit, pre-declared analysis plan, reference verification |
| `docs/` | Freeze manifest and technical audits |
| `CHECKSUMS.sha256` | SHA-256 digests for splits, authoritative results, and local `.pt` graphs |

Large processed graphs (`data/processed/*.pt`) are **not** stored in Git (GitHub size limits). Rebuild them from public SNAP/MUSAE releases and verify against `CHECKSUMS.sha256`.

## Quick start

```bash
pip install -r requirements.txt
python scripts/download_data.py --all
# build processed graphs with the project data scripts, then:
python scripts/build_splits.py --datasets facebook lastfm github --seeds 0 1 2 3 4 5 6 7 8 9
# or reuse the tracked masks under data/splits/
pytest -q
```

Export summary tables from the authoritative store:

```bash
python scripts/make_tables.py
python scripts/make_grid_tables.py
```

## Stage A freeze

- Datasets: LastFM Asia, Facebook Page-Page, GitHub Developers
- Seeds 0–9; seven methods; budgets 0.1–0.9
- Authoritative rows: **1,890**
- Suggested tag: `stage-a-github-frozen`
- See `docs/stage_a_freeze_manifest.md`

## R1 revision (corrected protocol)

The revision re-runs the study under a corrected protocol: teacher targets use **training labels only**, the
validation set is used only for selection (teacher size; downstream epoch), and test labels were read only after
the configuration in `configs/revised/locked.json` was committed. ORIGINAL (Stage A) results are kept unchanged.
See `docs/revision/AUDIT.md` (code audit and findings) and `docs/revision/PREREGISTRATION.md` (analysis plan).

```bash
pip install -r requirements.txt            # adds networkit, python-igraph, leidenalg, psutil
python scripts/revision/build_pubmed.py   # PubMed transfer graph + splits (other graphs: scripts/download_data.py)
python scripts/revision/build_teacher_cache.py                     # cache criterion values per edge/dataset/seed
python scripts/revision/run_experiment.py --phase select           # validation-only teacher-size selection
python scripts/revision/lock_config.py                             # apply the pre-declared selection rule
python scripts/revision/run_experiment.py --phase final            # CILP, Task-only, ablation and weight variants
python scripts/revision/run_baselines.py                           # baselines incl. NetworKit Local Degree/Similarity
python scripts/revision/run_timing.py                              # timing-grade cost decomposition (run alone)
python scripts/revision/analyze.py && python scripts/revision/make_timing_tables.py
pytest -q tests/test_revision.py                                   # equivalence of revision pipeline and original code
```

Every run is recorded in `results/revised/ledger.jsonl` (experiment ID, protocol tag, configuration, seeds,
budgets, teacher size, weights, cache ID, commit, hardware, wall-clock time, peak memory, status, output path).
NetworKit is called in a separate process because it and PyTorch ship conflicting OpenMP runtimes on macOS.

## Licence

MIT — see `LICENSE`.

## Citation

Cite the original ILP-GCN paper when using this codebase. See `CITATION.cff`.
