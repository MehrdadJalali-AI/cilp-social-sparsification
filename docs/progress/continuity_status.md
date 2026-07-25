# Continuity Status — Multi-seed Grid

**Updated:** 2026-07-24 (Facebook CAILP mid-run; GitHub queued)

## Completed
| Track | Status |
|---|---|
| LastFM seeds 0–9 × 8 core methods × budgets 0.1–0.9 | **Done** (80 JSON) |
| LastFM analysis, AUC, paired tests, figures, tables | **Done** |
| LastFM A1–A32 ablations (seeds 0–4; adversarial A26–A28 skipped) | **Done** (80 JSON) |
| Facebook teacher-size diagnostic | **Done** (teacher_n=120 val-selected; teacher40 failure retained) |
| Adversarial | **Excluded** (rejected; pilot F1 0.666 vs core 0.829 @ LastFM seed0/50%) |
| Abstract / novelty / contributions | **Deferred** (protocol) |

## In progress
- **Facebook** seeds 0–9 full grid (teacher_n=120): seeds **0 & 5 complete**; seeds **1 & 6 on `cailp_a31`**; others pending
- **Queue** `scripts/queue_after_facebook.sh`: starts GitHub grid + Facebook ablations when Facebook PIDs exit, then re-runs `analyze_grid.py`

## Pending
- GitHub seeds 0–9 × core methods
- Facebook ablations (A1–A32 subset, ≥1 larger dataset)
- Final RQ11 decision after ≥2 datasets with multi-seed paired significance
- Refresh curves / AUC / Pareto / runtime–memory tables

## Evidence snapshot (do not claim headline yet)

### LastFM (10 seeds) — multi vs A31
| | AUC Macro-F1 | Surrogate Spearman |
|---|---|---|
| CAILP multi | **0.593** | ~0.78 |
| A31 fidelity-only | 0.566 | ~0.32 |
| PTDNet (best overall AUC) | **0.616** | — |

Paired AUC multi>A31: Wilcoxon p≈0.004, d≈0.78; **4/4 dims** win → LastFM counts toward RQ11.

### Facebook (2 seeds so far: 0, 5) — directional only
| Method | AUC |
|---|---|
| original_ilp | 0.743 |
| PTDNet | 0.738 |
| **cailp_multi** | **0.735** |
| random | 0.734 |
| cailp_a31 | 0.718 |

Multi > A31 on mean AUC, but **n=2 → Wilcoxon not significant** (`dims_win=0`). Surrogate Spearman: multi ≈0.57 vs A31 ≈0.00.

### Ablations LastFM (seeds 0–4), fusion held = concat
| Contrast @ budgets | Result |
|---|---|
| A1_full_core − A31_single_obj_concat | multi wins 5/5 @30%, 4/5 @50%, 4/5 @70% |
| A8_concat − A31_single_obj_concat | **mixed** (4/5, 3/5, 2/5) |

RQ11 fusion-fixed ablation evidence is **weaker** than the full grid multi vs A31 contrast; report both honestly.

### RQ11 decision rule (current)
`retain_six_component_as_headline: **False**` — only LastFM has multi-dimension significant wins. Need Facebook (and preferably GitHub) with enough seeds for paired tests.

## Resume commands
```bash
# Already running for Facebook; resume-safe skips completed files
python scripts/run_full_grid.py --datasets facebook --seeds 0 1 2 3 4 5 6 7 8 9
# After Facebook (or via queue_after_facebook.sh):
python scripts/run_full_grid.py --datasets github --seeds 0 1 2 3 4 5 6 7 8 9
python scripts/run_ablations_grid.py --datasets facebook --seeds 0 1 2 --budgets 0.3 0.5 0.7
python scripts/analyze_grid.py && python scripts/make_grid_tables.py
```

## Manuscript (2026-07-24)
Restructured to Nature Communications–style Results-first article (`paper/main.pdf`, 11 pp) + SI (`paper/supplementary.pdf`).
Plan: `docs/manuscript_restructuring_plan.md`. Change log: `docs/manuscript_change_log.md`.
