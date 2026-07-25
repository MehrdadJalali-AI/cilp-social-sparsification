# CILP — Counterfactual Inverse Link Prediction for Social-Network Sparsification

Public **code and Stage A experimental results** for utility-preserving social-network sparsification via Counterfactual Inverse Link Prediction (CILP).

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
| `data/splits/` | Fixed train/development/test masks (seeds 0–9) |
| `results/processed/` | Authoritative Stage A store (1,890 rows) |
| `results/raw/grid/` | Per-seed raw JSON outputs |
| `results/tables/` | Aggregated CSV tables |
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

## Licence

MIT — see `LICENSE`.

## Citation

Cite the original ILP-GCN paper when using this codebase. See `CITATION.cff`.
