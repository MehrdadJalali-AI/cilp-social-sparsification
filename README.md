# CILP — Counterfactual Inverse Link Prediction for Social Graph Sparsification

Public **code and experimental results** for exact-budget graph sparsification via Counterfactual Inverse Link
Prediction (CILP).

**Method name:** CILP (multi-criteria counterfactual teacher)  
**Matched control:** Task-only teacher  
**Parent method:** ILP-GCN (Bangian Tabrizi, Jalali & Houshmand, *Journal of Big Data*, 2025)

This repository does **not** include the journal manuscript sources.

## Repository contents

| Path | Description |
|---|---|
| `src/` | CILP teacher, scorer, pruning, baselines |
| `src/revision/`, `scripts/revision/` | Experiment pipeline: criterion cache, teacher-size selection, CILP / Task-only / ablation and weight variants, baselines (incl. NetworKit Local Degree / Local Similarity), community metrics, timing, statistics |
| `scripts/` | Data preparation, grid runner, evaluation, CSV table export |
| `configs/` | Experiment configurations (`configs/revised/locked.json`: selected teacher sizes) |
| `data/splits/` | Fixed train/validation/test masks (seeds 0–9; LastFM, Facebook, GitHub, PubMed) |
| `results/revised/` | Results reported in the article: per-run JSON (`final/`, `select/`, `timing/`), analysis tables, experiment ledger |
| `results/processed/`, `results/raw/grid/`, `results/tables/` | Stage A grid store (seven methods, three datasets) |
| `results/processed/recent_baselines/` | Stage A results for GASSIP-adapted, MoG-adapted, PSGNN-reimpl |
| `docs/revision/` | Code audit, pre-declared analysis plan, reference verification |
| `docs/` | Technical audits and freeze manifest |
| `CHECKSUMS.sha256` | SHA-256 digests for splits, result stores, and local `.pt` graphs |

Large processed graphs (`data/processed/*.pt`) are **not** stored in Git (GitHub size limits). Rebuild them from the
public SNAP/MUSAE and Planetoid releases and verify against `CHECKSUMS.sha256`.

## Protocol

Teacher targets are built from **training labels only**; the validation set is used only for selection (teacher size
and the training epoch of each downstream GCN); test labels are used only for the final evaluation, after the
configuration in `configs/revised/locked.json` was fixed. See `docs/revision/PREREGISTRATION.md` (analysis plan) and
`docs/revision/AUDIT.md` (code audit).

## Reproducing the results

```bash
pip install -r requirements.txt
python scripts/download_data.py --all                              # LastFM, Facebook, GitHub
python scripts/revision/build_pubmed.py                            # PubMed graph and splits
python scripts/revision/build_teacher_cache.py                     # cache criterion values per edge/dataset/seed
python scripts/revision/run_experiment.py --phase select           # validation-only teacher-size selection
python scripts/revision/lock_config.py                             # apply the pre-declared selection rule
python scripts/revision/run_experiment.py --phase final            # CILP, Task-only, ablation and weight variants
python scripts/revision/run_baselines.py                           # baselines
python scripts/revision/run_timing.py                              # timing-grade cost decomposition (run alone)
python scripts/revision/analyze.py && python scripts/revision/make_timing_tables.py
pytest -q                                                          # tests, incl. equivalence of the cached pipeline
```

Every run is recorded in `results/revised/ledger.jsonl` (experiment ID, configuration, seeds, budgets, teacher size,
weights, cache ID, commit, hardware, wall-clock time, peak memory, status, output path).
NetworKit is called in a separate process because it and PyTorch ship conflicting OpenMP runtimes on macOS.

## Stage A grid

- Datasets: LastFM Asia, Facebook Page-Page, GitHub Developers
- Seeds 0–9; seven methods; budgets 0.1–0.9; 1,890 rows
- Export tables: `python scripts/make_tables.py && python scripts/make_grid_tables.py`
- See `docs/stage_a_freeze_manifest.md`

## Licence

MIT — see `LICENSE`.

## Citation

Please also cite the ILP-GCN paper when using this codebase. See `CITATION.cff`.
