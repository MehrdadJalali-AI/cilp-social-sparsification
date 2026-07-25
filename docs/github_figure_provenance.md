# GitHub figure provenance (Stage A)

Source CSV: `results/processed/authoritative_results.csv` (1890 rows after GitHub merge).

CI: mean ± t₀.₉₇₅,ₙ₋₁ × SEM over seeds (n=10 for all three datasets).

Method order / colors unchanged from prior LastFM/Facebook figures.

| Figure | Datasets | Filters | Output |
|---|---|---|---|
| Fig. 2 sparsity curves | lastfm, facebook, **github** | all 7 methods; budgets 0.1–0.9 | `paper/figures/fig2_sparsity_curves.pdf` |
| Fig. 3 AUC bars | lastfm, facebook, **github** | per-seed AUC then mean±CI | `paper/figures/fig3_method_comparison.pdf` |
| Fig. 4 teacher compare | lastfm, facebook, **github** | CILP vs Task-only paired seeds | `paper/figures/fig4_teacher_compare.pdf` |
| Fig. 5 structure heatmaps | lastfm, facebook, **github** | 50% removal; 5 methods | `paper/figures/fig5_structure.pdf` |
| Fig. 7 runtime | lastfm, facebook, **github** | train_seconds means | `paper/figures/fig7_runtime.pdf` |

LastFM and Facebook visual values regenerated from the same authoritative rows (fingerprint unchanged for LF/FB Macro-F1 keys).
