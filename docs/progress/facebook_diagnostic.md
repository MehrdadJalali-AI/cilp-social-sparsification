# Facebook Negative-Result Investigation

**Protocol:** Variants ranked by **validation** Macro-F1 at 50% removal. Test metrics reported for all variants (no test-based selection).

## Finding
The earlier Facebook lite pilot (teacher_n=40) underperformed random/ILP because the **teacher sample was too small**, not because CAILP is intrinsically worse on Facebook.

| Variant (val-ranked) | Val F1 @50% | Test F1 @50% | Surrogate Spearman | Top-k P/R |
|---|---|---|---|---|
| teacher120 (selected) | 0.936 | 0.938 | 0.48 | 0.62 |
| teacher80 | 0.934 | 0.939 | 0.59 | 0.84 |
| teacher40 (pilot) | 0.911 | 0.906 | 0.65 | 1.00* |
| teacher20 | ~0.90 | 0.900 | 0.69 | — |

\*Top-k on tiny labeled sets can be optimistically high.

## Action
Facebook full grid uses **teacher_n=120** (validation-selected). Unfavorable pilot result is retained in the record; not hidden.

## Artifact
`results/raw/grid/facebook_diagnostic_seed0.json`
`results/processed/facebook_diagnostic_seed0_summary.json`
