# GitHub completion summary (Stage A — frozen protocol)

## Executive status

| Item | Value |
|---|---|
| All ten seeds completed | **yes** |
| Completed seed IDs | 0 1 2 3 4 5 6 7 8 9 |
| Method failures | **none** |
| Memory failures | **none** |
| Protocol changes | **none** |
| LastFM/Facebook altered | **no** (SHA identical) |
| GitHub decision-rule outcome | **Supported** (4/4 metrics) |
| Manuscript edited | **no** (Stage A restriction) |
| Stage B started | **no** |

## Runtime

| Scope | Wall-clock |
|---|---|
| Diagnostic baselines seed0 @0.5 | ~123 s |
| Diagnostic Task-only seed0 @0.5 | train_seconds 518.0 s |
| Diagnostic CILP seed0 @0.5 | train_seconds 590.9 s |
| Full grid (seeds 0–9, all methods, budgets 0.1–0.9) | **22,282,607 ms ≈ 6.19 h** |

## Authoritative GitHub rows

- **630** rows (10 × 7 scientific methods × 9 budgets)
- Total authoritative store: **1890** rows
- Path: `results/processed/authoritative_results.csv`

## GitHub predictive AUC (mean ± s.d., n=10)

| Method | AUC |
|---|---|
| ILP-GCN | 0.6504 ± 0.0035 |
| PTDNet | 0.6503 ± 0.0033 |
| CILP | 0.6480 ± 0.0039 |
| Random | 0.6470 ± 0.0034 |
| Resistance proxy | 0.6457 ± 0.0037 |
| Task-only | 0.6447 ± 0.0044 |
| NeuralSparse | 0.6395 ± 0.0045 |

## Paired CILP − Task-only (GitHub, n=10, Holm within 4-metric family)

| Metric | Δ | Holm p | Supported | W/T/L |
|---|---|---|---|---|
| sparsity–Macro-F1 AUC | +0.00326 | 0.0078125 | yes | 10/0/0 |
| giant-component ratio @50% | +0.2284 | 0.0078125 | yes | 10/0/0 |
| bridge retention @50% | +0.3195 | 0.0078125 | yes | 10/0/0 |
| minority-degree retention @50% | +0.1136 | 0.0078125 | yes | 9/0/1 |

**Decision:** Supported (≥2 metrics with positive Δ and Holm p<0.05).

## Regenerated artifacts (no manuscript text edits)

### Tables

- `paper/tables/paired_structural_tests.csv` (includes GitHub)
- `paper/tables/table4_structure.csv` / `.tex`
- `paper/tables/s14_full.tex`
- `paper/tables/s16_18_paired.tex`
- `paper/tables/s19_summary.tex`
- `paper/tables/s22_structure.tex`
- `paper/tables/s23_table.tex`
- `paper/tables/descriptive_macro_f1.csv`
- `results/processed/multi_dim_decision.json`

### Figures

- `paper/figures/fig2_sparsity_curves.pdf`
- `paper/figures/fig3_method_comparison.pdf`
- `paper/figures/fig4_teacher_compare.pdf`
- `paper/figures/fig5_structure.pdf`
- `paper/figures/fig7_runtime.pdf`

### Audits

- `docs/github_frozen_protocol_audit.md`
- `docs/github_leakage_audit.md`
- `docs/github_runtime_estimate.md`
- `docs/github_execution_log.md`
- `docs/github_result_merge_audit.md`
- `docs/github_figure_provenance.md`
- `docs/github_completion_summary.md` (this file)

## Identical-protocol assumptions

Held: same runner, GRID_CFG github block, β defaults, concat fusion, unconstrained μ ranking, CPU device, seeds 0–9, budgets 0.1–0.9, no retuning.

## Unresolved / next

- Manuscript Abstract/Results/Conclusions **not yet updated** (await Stage A review per protocol).
- Stage B (adaptive teacher sampling) **not started**.
