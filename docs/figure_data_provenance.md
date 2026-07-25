# Figure data provenance

**Authoritative source:** `results/processed/authoritative_results.csv`  
**SHA-256:** `a84cd74f21c7fb419480a1abc7fdce2709c2aa8ed891677459bf6fc9c138a9d5`  
**Rows:** 1,890 (LastFM 630 + Facebook 630 + GitHub 630)  
**CI convention:** mean ± \(t_{0.975,n-1}\times\mathrm{SEM}\) over seeds  
**Stage A freeze:** `docs/stage_a_freeze_manifest.md`

Method display order / colors (Figs. 2–5, 7):  
CILP, Task-only, ILP-GCN, PTDNet, NeuralSparse, Random, Resistance proxy.

---

## Fig. 2 — Sparsity–Macro-F1 curves
- Filter: all methods, budgets 0.1–0.9
- Datasets: LastFM, Facebook, GitHub (three panels)
- \(n=10\) seeds each
- File: `paper/figures/fig2_sparsity_curves.pdf`

## Fig. 3 — Paired CILP AUC forest plot
- Paired mean Δ AUC (CILP − Task-only / ILP-GCN / PTDNet) with 95% t-CI, n=10
- Source table: `paper/tables/fig3_forest_paired_auc.csv`
- Datasets: LastFM, Facebook, GitHub
- File: `paper/figures/fig3_method_comparison.pdf`

## Fig. 4 — Multi-criteria vs Task-only
- Paired common seeds only; concatenation fusion
- Datasets: LastFM, Facebook, GitHub
- File: `paper/figures/fig4_teacher_compare.pdf`

## Fig. 5 — Structural / group metrics
- Budget 0.5 only; mean over seeds 0–9
- Heatmap: rows = methods, columns = metrics; shared colour bar encodes values in [0.35, 1.0]
- Datasets: LastFM, Facebook, GitHub
- File: `paper/figures/fig5_structure.pdf`

## Fig. 6 — LastFM ablations
- Source: ablation_lastfm_*.json seeds 0–4 (\(n=5\))
- Fusion: Concatenation / Gated / Cross-attn (multi-criteria teacher)
- Teacher: Multi-criteria vs Task-only (concatenation)
- Context: CILP vs ILP-GCN (not a matched ablation)
- File: `paper/figures/fig6_ablations.pdf`

## Fig. 7 — Runtime
- Field: `train_seconds` from authoritative CSV
- CILP / Task-only include teacher construction (not separable in logs)
- Datasets: LastFM, Facebook, GitHub
- GitHub full-grid orchestration wall-clock ≈ 6.19 h is separate from per-method means
- File: `paper/figures/fig7_runtime.pdf`

## SI teacher-size figure
- Source: facebook_diagnostic_seed0.json
- Selection rule: max validation Macro-F1 at 50% removal
- File: `paper/figures/fig_s8_teacher_size.pdf`
