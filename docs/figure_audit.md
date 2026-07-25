# Figure audit

**Date:** 2026-07-24

## Shared style contract (Figs 2–7)

| Item | Spec |
|---|---|
| Method order | CAILP-Social (multi-obj.); fidelity-only A31; original ILP-GCN; PTDNet; NeuralSparse; Random; resistance-style proxy |
| Colors | Fixed hex map in `scripts/generate_paper_figures.py` |
| CI | Mean ± 1.96 × SEM (SEM = sample SD / √n over seeds); stated in every caption |
| X-axis | edge-removal rate |
| Y predictive | Macro-F1 |
| Seed counts | Stated per panel/dataset in caption |

## Figure status

| Fig | Status | Notes |
|---|---|---|
| 1 | Redesign | TikZ two-panel training / sparsification; vector PDF via main compile |
| 2 | Regen | Curves + CI bands; seed counts in caption |
| 3 | Regen | AUC bars; same order/colors |
| 4 | Regen | Multi-obj. vs A31 only; fusion fixed |
| 5 | Redesign | Two panels; shared colorbar; short labels |
| 6 | Redesign | Grouped ablations; short labels; ILP separated |
| 7 | Redesign | train_seconds + prune_seconds from JSON; log y; CPU |

## Timing provenance

Raw JSON provides `train_seconds` (includes teacher construction + scorer training for CAILP) and `prune_seconds`.
No separate teacher-only timer is logged; SI states this limitation explicitly.
