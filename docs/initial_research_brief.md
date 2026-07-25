# CAILP-Social — Initial Research Brief (Pre-Implementation)

**Date:** 2026-07-24  
**Rule:** Expensive experiments deferred until pilot, unit tests, and leakage audit succeed.  
**Novelty gate:** Checked against `docs/foundational_analysis.md` §0 and §6.

---

## 1. Original ILP (concise)

ILP-GCN (Bangian Tabrizi, Jalali & Houshmand, *J Big Data* 2025) sparsifies MOF similarity graphs by (i) training a GCN link scorer \(S_{\mathrm{GCN}}(e_{ij})\), (ii) forming inverse weights \(W_{\mathrm{ILP}}=\alpha/(S_{\mathrm{GCN}}+\varepsilon)\), (iii) fusing with initial similarity weights, and (iv) pruning by a threshold chosen where modularity peaks. Edge importance is **derived**, not supervised by measured deletion harm. Modularity guides **threshold selection / evaluation**, not GCN training. Nodes may disappear when isolated — Track A must avoid that coupling.

## 2. Black Hole Strategy (concise)

Black Hole (Jalali et al., *JCIM* 2025) is gravity-inspired **representative node sampling** for frugal MOF learning. It is **not** an ILP edge scorer. In CAILP it belongs to Track B and Optional Module A; the unmodified formulation is a mandatory comparison point for any new mass prior (RQ8, A32).

## 3. External prior art (concise contrast)

**CF-GNNExplainer family:** delete few edges to **flip one prediction** (local explanation).  
**CAILP:** score every edge by **aggregated multi-dimensional harm** and prune under a **global budget** with structural/fairness constraints. Shared mechanic; different objective, scope, and supervision. Mechanic ≠ novelty.

**DropEdge / NeuralSparse / PTDNet / DSpar:** already instantiate random, task-feedback, topological-denoising, or degree/resistance importance. “Learn importance and prune” is not new; CAILP must beat these on the broader evaluation battery, especially vs fidelity-only CF (A31).

## 4. Main framework vs optional hypotheses

| Component | Status |
|---|---|
| Counterfactual multi-objective ILP teacher + surrogate | **Core** |
| Node + edge views, fusion (incl. cross-attention as one option) | **Core architecture; fusion choice empirical** |
| Uncertainty-aware scoring | **Core** |
| Constrained budgeted pruning | **Core** |
| Black Hole / node-mass / gravity priors | **Optional Module A — rejectable** |
| Adversarial representation / mask GAN | **Optional Module B — rejectable** |

Final inference must always emit explicit ILP scores \(s_{ij}\in[0,1]\).

## 5. Mathematical formulation (core)

**Graph:** \(G=(V,E)\), attributes \(X\), adjacency \(A\).

**Node encoder:** \(H = f_\theta(X,A)\), \(h_i\in\mathbb{R}^d\).

**Node-centric edge features:**
\[
z_{ij}^{\mathrm{node}} = \big[h_i+h_j \,\|\, |h_i-h_j| \,\|\, h_i\odot h_j \,\|\, \cos(h_i,h_j) \,\|\, \phi_{\mathrm{struct}}(i,j)\big]
\]
(undirected: average both orientations if needed).

**Edge-centric:** line-graph GNN or memory-guarded local edge encoder → \(z_{ij}^{\mathrm{edge}}\).

**Counterfactual teacher** (full model — all six terms required):
\[
\begin{aligned}
\Delta_{\mathrm{task}}(e) &= L_{\mathrm{val}}(G^{-e}) - L_{\mathrm{val}}(G), \\
y_e^{\mathrm{cf}} &= \mathrm{Normalize}\Big(
\beta_{\mathrm{task}}\Delta_{\mathrm{task}}
+\beta_{\mathrm{comm}}\Delta_{\mathrm{comm}}
+\beta_{\mathrm{conn}}\Delta_{\mathrm{conn}}
+\beta_{\mathrm{spec}}\Delta_{\mathrm{spec}}
+\beta_{\mathrm{repr}}\Delta_{\mathrm{repr}}
+\beta_{\mathrm{group}}\Delta_{\mathrm{group}}
\Big).
\end{aligned}
\]
\(\beta\) selected on validation only. Surrogate extends labels from stratified teacher subset.

**A31 / RQ11 control:** \(y_e^{\mathrm{cf,single}} = \mathrm{Normalize}(\Delta_{\mathrm{fidelity}})\) only.

**Fusion (cross-attention variant):**
\[
u_{ij}=\mathrm{CrossAttn}(Q(z_{ij}^{\mathrm{node}}),K(z_{ij}^{\mathrm{edge}}),V(z_{ij}^{\mathrm{edge}})),\quad
z_{ij}=\mathrm{LN}\big(P_n(z_{ij}^{\mathrm{node}})+P_u(u_{ij})\big).
\]

**Decoder:** \((\mu_e,\log\sigma_e^2)=\mathrm{Dec}(z_{ij})\); importance \(s_e=\sigma(\mu_e)\) or calibrated map to \([0,1]\).

**Conservative removal (ablation):** remove if \(\mu_e+\kappa\sigma_e < \tau\); main comparison uses **equal edge-retention budgets**.

**Loss (configurable weights on val):**
\[
\mathcal{L}=
\lambda_{\mathrm{cf}}\mathcal{L}_{\mathrm{cf}}
+\lambda_{\mathrm{rank}}\mathcal{L}_{\mathrm{rank}}
+\lambda_{\mathrm{task}}\mathcal{L}_{\mathrm{task}}
+\lambda_{\mathrm{comm}}\mathcal{L}_{\mathrm{comm}}
+\lambda_{\mathrm{conn}}\mathcal{L}_{\mathrm{conn}}
+\lambda_{\mathrm{spec}}\mathcal{L}_{\mathrm{spec}}
+\lambda_{\mathrm{repr}}\mathcal{L}_{\mathrm{repr}}
+\lambda_{\mathrm{sparse}}\mathcal{L}_{\mathrm{sparse}}
+\lambda_{\mathrm{unc}}\mathcal{L}_{\mathrm{unc}}
+\lambda_{\mathrm{cal}}\mathcal{L}_{\mathrm{cal}}.
\]

**Constrained prune:** rank ascending by importance; delete iff constraints (MSF/bridges/min-degree/community backbone/minority connectivity/…) hold; stop at budget; report shortfalls.

## 6. Pipeline diagram

```text
Original Social Graph
        │
        ▼
Graph / Feature Preprocessing  (splits, leakage firewall)
        │
        ▼
Node Encoder (GCN | GraphSAGE | GATv2)
        │
        ├──────────────────────┐
        ▼                      ▼
Node-Centric Edge z_ij^node   Edge-Centric Encoder (line-graph | local)
        │                      │
        └──────────┬───────────┘
                   ▼
        Counterfactual Teacher (6-comp) ──► Surrogate
                   │
                   ▼
        Dual-View Fusion (concat | gate | bilinear | cross-attn)
                   │
                   ▼
        Importance μ_e + Uncertainty σ_e
                   │
                   ▼
        Connectivity / Community / Fairness-Constrained Ranking
                   │
                   ▼
        Sparsified Graph G_ρ
                   │
                   ▼
Predictive | Structural | Community | Centrality | Fairness | Robustness | Efficiency
```

Optional: Node-mass prior → edge prior \(g_{ij}\); Adversarial D / mask generator — **outside** core path unless decision rules pass.

## 7. Baseline-selection table

| # | Method | Provenance |
|---|---|---|
| 1 | Original graph | Control |
| 2–13 | Random, degree-sum/product, CN, Jaccard, AA, RA, PA, betweenness, local similarity, spanning-tree/backbone, effective-resistance/spectral | **External classical** |
| 14 | DropEdge | **External** (Rong et al. 2020) |
| 15 | Original ILP-GCN | **Team prior** (J Big Data 2025) |
| 16 | GAT-ILP w/o CF | **New for this project** (ablation architecture) |
| 17 | NeuralSparse | **External** (Zheng et al. 2020) |
| 18 | PTDNet | **External** |
| 19 | DSpar | **External** |
| 20 | CF-ILP w/o attention | **New** (ablation) |
| 21 | Single-objective CF (CF-GNNExplainer-style) | **External mechanic, adapted** (RQ11 / A31) |
| 22 | CAILP-Social core | **New** (this project) |
| 23–25 | + mass / gravity / unmodified Black Hole mass | **Optional; 25 = team prior reused** |
| 26–28 | + adversarial variants / full | **Optional / new** |
| Track B | Random/degree/PR/k-core/betweenness/k-center/clustering/coreset + **Black Hole** + learnable mass | Mixed; Black Hole = **team prior** |

## 8. Experimental matrix (incl. RQ11)

| Track | Question | Key comparisons |
|---|---|---|
| A | Edge sparsification @ ρ∈{0.1…0.9} retention | All edge baselines vs CAILP; equal budgets |
| B | Node sampling | Black Hole vs classical node samplers |
| C | Hybrid orderings | node→ILP, ILP→node, edge-only, node-only, alternating |
| RQ11 / A31 | Multi- vs single-objective CF | Full 6-comp teacher vs fidelity-only; **hold fusion fixed** |
| A32 / RQ8 | Mass priors | none / degree / PR / analytical / learnable / gravity / **unmodified BH** |
| RQ9–10 | Adversarial | none / D / mask GAN / GAN-only / both |
| Ablation A1–A32 | Architecture & losses | LastFM + ≥1 larger graph |

Seeds: preferably 10; ≥5 for expensive runs. Primary predictive metric: Macro-F1.

## 9. Leakage-risk analysis

| Risk | Mitigation |
|---|---|
| Test labels in CF teacher / pruning | Teacher and constraints use train+val only; test labels at final eval |
| Test edges in LP sparsification | Sparsify **training graph only**; hold out val/test positives |
| Threshold / β / λ tuned on test | Val-only selection; freeze before test |
| Feature norm using test stats | Fit scalers on train (or train+val if pre-registered); document |
| Seed picking / discarded failures | Fixed seed list; report all runs |
| Unequal tuning budgets | Same search protocol for all methods |
| Surrogate trained with test edges | Stratified teacher edges from train graph only |

Full ledger: `docs/leakage_audit.md` (Phase 9; stub early).

## 10. Scalability-risk analysis

| Risk | Mitigation |
|---|---|
| Exact CF on all edges | Stratified teacher subset + surrogate; cache Δ components |
| Line-graph blow-up | Estimate \(|E_L|\); memory guard; auto-switch to local edge encoder |
| Spectral Δ per edge | Lanczos / few eigenvalues / Nyström-style approx on subset |
| Community Δ per edge | Local modularity contribution / sampled recomputes |
| GAN instability | WGAN-GP option; early reject on seed variance |
| GPU OOM | Mini-batch edges; CPU fallback; checkpointing |

## 11. Staged implementation plan

Aligned with project Phases 1–17. **Current:** Phase 1 done.  
Next: Phase 2 math/hypotheses → Phase 3 dataset audit → Phase 4 ILP reproduce → Phase 5 BH Track B → Phase 6 baselines incl. A31 → Phase 7 core CAILP → Phase 8 LastFM pilot → Phase 9 tests+leakage → Phase 10 protocol freeze → full runs → optional modules → ablations → paper **after** results.

## 12. Resource estimate (order-of-magnitude)

| Item | Estimate |
|---|---|
| Hardware | 1× GPU ≥24GB (A5000/A6000/4090-class) preferred; CPU-only possible for small pilots |
| Host RAM | 32–64 GB |
| Disk | 20–50 GB (raw+processed+results) |
| LastFM pilot (core, 5 seeds, few budgets) | ~0.5–2 GPU-days |
| Full Track A three datasets × budgets × seeds × baselines | ~2–6 GPU-weeks depending on teacher size |
| Exact teacher on all edges | **Infeasible** at full scale — surrogate mandatory |
| Line-graph on GitHub (high degree) | High risk — local encoder default likely |

Refine after Phase 3 audit and Phase 8 pilot timings.

## 13. Conservative novelty statement (pre-result)

CAILP-Social is an **extension** of the team’s Inverse Link Prediction framework (ILP-GCN, *J Big Data* 2025) to attributed social-network sparsification. It adapts ILP’s question — *how harmful is removing an existing edge?* — by supervising edge scores with a **multi-dimensional counterfactual deletion-harm** target and pruning under uncertainty-aware structural and group constraints.

Edge-deletion counterfactuals are **borrowed** from the GNN explanation literature (e.g., CF-GNNExplainer) and are **not** claimed as novel. Learning to score and drop edges is **not** claimed as novel relative to NeuralSparse, PTDNet, DSpar, or DropEdge. Dual-view attention is a **design choice** pending ablation. Node-mass (including Black Hole) and adversarial modules are **optional hypotheses**.

Defensible novelty, **if and only if** experiments support it, is limited to: (a) multi-dimensional CF harm for global budgeted sparsification vs single-objective CF; (b) uncertainty-aware constrained pruning operationalizing that signal; (c) evidence-based accept/reject of optional modules. The final paper will claim no more than the evidence and this positioning jointly allow.
