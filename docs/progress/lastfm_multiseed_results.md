# LastFM Complete Multi-seed Results (seeds 0–9)

**Budgets:** 0.1–0.9 | **Downstream:** GCN | **Fusion for RQ11:** concat (fixed)  
**Adversarial:** excluded (rejected)

## Macro-F1 (mean ± std)

| Method | 30% | 50% | 70% | AUC |
|---|---|---|---|---|
| PTDNet | 0.795±0.025 | 0.788±0.013 | 0.760±0.018 | **0.616±0.012** |
| CAILP multi | 0.786±0.040 | 0.769±0.061 | 0.715±0.071 | 0.593±0.038 |
| Original ILP-GCN | 0.801±0.019 | 0.776±0.023 | 0.702±0.032 | 0.590±0.017 |
| Random | 0.773±0.016 | 0.745±0.036 | 0.696±0.036 | 0.580±0.018 |
| CAILP A31 (fidelity-only) | 0.758±0.062 | 0.730±0.083 | 0.673±0.081 | 0.566±0.050 |
| NeuralSparse | 0.665±0.022 | 0.602±0.013 | 0.551±0.021 | 0.489±0.010 |
| DSpar / eff-resistance proxy | 0.613±0.031 | 0.558±0.027 | 0.524±0.020 | 0.458±0.014 |

## Surrogate quality (mean over seeds)
| Teacher | MAE | Spearman | Kendall | ECE |
|---|---|---|---|---|
| Multi-objective | 0.076 | **0.775** | 0.632 | 0.077 |
| A31 single-objective | 0.115 | 0.318 | 0.225 | 0.147 |

## RQ11 on LastFM alone
Multi-objective beats A31 on AUC and most budgets; surrogate ranking quality is substantially better. **Headline retention still requires ≥2 datasets** under the stated decision rule — pending Facebook/GitHub.

## Notes
- DSpar and effective-resistance currently share the same resistance-style proxy implementation in this codebase (reported separately for completeness).
- Figures: `results/figures/sparsity_f1_lastfm.png`
- Tables: `results/tables/table3_multiseed_macro_f1.csv`
