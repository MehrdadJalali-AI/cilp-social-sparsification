# Preliminary Results (Pilot — not final claims)

**Scope:** LastFM Asia, seed 0, removal rates {0.3, 0.5, 0.7}, GCN downstream Macro-F1.  
**Status:** Exploratory. No multi-seed statistics yet. Do not treat as paper conclusions.

## Original graph
Macro-F1 ≈ **0.848**

## Track A (edge sparsification)

| Method | 0.3 | 0.5 | 0.7 |
|---|---|---|---|
| Random | 0.793 | 0.762 | 0.712 |
| DropEdge-export | 0.817 | 0.762 | 0.706 |
| Jaccard | 0.751 | 0.741 | 0.693 |
| Degree-sum | 0.666 | 0.614 | 0.534 |
| DSpar-proxy | 0.623 | 0.577 | 0.540 |
| Original ILP-GCN | 0.833 | 0.802 | 0.747 |
| CAILP multi (budget) | 0.833 | 0.829 | 0.796 |
| CAILP multi (constrained) | 0.833 | **0.834** | 0.792 |
| A31 single-obj CF (budget) | **0.841** | 0.801 | 0.785 |
| A31 single-obj CF (constrained) | 0.836 | 0.830 | 0.784 |

## RQ11 (pilot only)
Mixed: A31 wins at 30% removal (budget); multi-objective wins at 50–70% removal. **Decision rule not yet met** (needs ≥2 datasets, multi-seed, fusion held fixed). Retain six-component design as a hypothesis, not a confirmed headline.

## Track B (node sampling) — not comparable to edge %
Black Hole / degree / PageRank node retention runs completed (`results/raw/node_sample_*`). Report separately from Track A.

## Facebook lite (50% removal, seed 0)
| Method | Test Macro-F1 |
|---|---|
| Random | 0.923 |
| DropEdge-export | 0.922 |
| Original ILP-GCN | **0.933** |
| CAILP multi (budget, lite teacher) | 0.904 |

Lite CAILP underperformed random/ILP on Facebook under the reduced teacher/training budget. Treat as a **negative/mixed** pilot signal requiring fuller hyperparameters before any claim.

## Optional modules
- **Adversarial (LastFM, 50% removal, seed 0):** Macro-F1 ≈ **0.666** vs core CAILP ≈ **0.829**. **Reject** for final model under current decision rules (worse utility; no Pareto gain).
- Node-mass (A32) dedicated edge-prior ablations still pending beyond Track B reproduction.
