# Statistical audit — multi-criteria vs task-only

## Family definition

For each dataset (LastFM, Facebook), one family of four paired metrics (CILP − Task-only) on common seeds 0–9:

1. sparsity–Macro-F1 AUC
2. giant-component ratio @ 50%
3. bridge retention @ 50%
4. minority-degree retention @ 50%

Holm adjustment is applied to Wilcoxon *p*-values **within each dataset family**.

CI: paired mean difference ± \(t_{0.975,n-1}\mathrm{SEM}\).  
Effect size: \(d_z = \mathrm{mean}(\mathrm{diff})/\mathrm{sd}(\mathrm{diff})\).

## Decision rule

Multi-criteria supervision is supported across datasets when ≥2 metrics are supported (positive Δ and Holm *p*<0.05) on ≥2 datasets under identical concatenation fusion.

## Decision outcome

**Supported.** LastFM: 4/4 metrics. Facebook: 4/4 metrics.

Artifacts:

- `results/processed/paired_tests.json`
- `results/processed/multi_dim_decision.json`
- `results/processed/multi_dim_decision.json`

## Predictive AUC (unchanged numerically)

| Dataset | ΔAUC | 95% CI | Wilcoxon p | Holm p (4-metric family) | \(d_z\) | W/T/L |
|---|---|---|---|---|---|---|
| LastFM | +0.0265 | [0.0022, 0.0508] | 0.003906 | 0.007812 | 0.781 | 9/0/1 |
| Facebook | +0.0142 | [0.0053, 0.0231] | 0.003906 | 0.015625 | 1.137 | 9/0/1 |

## Structural interpretation (vs other methods)

CILP ≫ Task-only on GC/bridge/minority.  
ILP-GCN is stronger overall on reported structural metrics (especially Facebook).  
PTDNet slightly exceeds CILP on LastFM minority-degree retention.  
No direct community-preservation metric is reported.
