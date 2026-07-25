# Result provenance audit

**Date:** 2026-07-24

## Sources

| Artifact | Path |
|---|---|
| Grid JSON | `results/raw/grid/{dataset}_seed{N}_{method}.json` |
| Ablations | `results/raw/grid/ablation_lastfm_seed{N}_*.json` |
| Aggregates | `results/tables/table3_multiseed_macro_f1.csv`, `table_surrogate_quality.csv`, `table14_paired_tests.csv` |
| Figure regen | `scripts/generate_paper_figures.py` |
| SI table regen | `scripts/generate_si_tables.py` |

## Seed coverage (core grid, excluding teacher-diagnostic files)

| Dataset | Method | Seeds |
|---|---|---|
| LastFM | all core methods | 0–9 (n=10) |
| Facebook | CAILP-Social multi-obj. | 0,1,5,6 (n=4) |
| Facebook | A31, baselines | 0,1,2,5,6,7 (n=6) |
| GitHub | — | none |

## Numbers in main tables

All Macro-F1 / AUC / structure / runtime means in main text are recomputed from JSON by the figure/SI scripts (mean ± s.d. over seeds; CI = mean ± 1.96×SEM).

## Leakage

Teacher-size selection used validation Macro-F1 (Facebook diagnostic). Test metrics reported only after selection. Documented in SI leakage-audit table.

## Not claimed

- Independent DSpar reproduction
- Complete Facebook 0–9 CAILP
- Any GitHub results
- Separate teacher wall-clock (not logged)
