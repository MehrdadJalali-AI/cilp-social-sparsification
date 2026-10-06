# Stage B provisional results store

**Status:** Complete — **810 / 810** rows validated (3 methods × 3 datasets × 10 seeds × 9 budgets).

Stage A authoritative results remain frozen:

- `results/processed/authoritative_results.csv` (1,890 rows — hash `a84cd74f…` unchanged)

## Artifacts

| Artifact | Content |
|----------|---------|
| `stage_b_results.csv` / `.json` | 810 rows |
| `../../paper/tables/stage_b_auc_comparisons.csv` | 9 CILP-vs-baseline AUC Holm contrasts (n=10 each) |

## Provisional AUC summary (CILP − baseline; positive = CILP higher)

| Dataset | vs GASSIP-adapted | vs MoG-adapted | vs PSGNN-reimpl |
|---------|-------------------|----------------|-----------------|
| LastFM | −0.0048 (ns) | +0.0839\* | +0.1360\* |
| Facebook | +0.0186\* | +0.0170\* | +0.0205\* |
| GitHub | +0.0076\* | +0.0048\* | +0.0049\* |

\*Holm-adjusted Wilcoxon p < 0.05. On LastFM, GASSIP-adapted is competitively tied with CILP (slightly higher mean AUC; not significant).

**Do not merge into authoritative store until authors approve.** Manuscript claims should remain cautious; report GASSIP-adapted LastFM tie transparently.

See `docs/stage_b_experimental_protocol.md`.
