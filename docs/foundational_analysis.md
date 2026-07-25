# Foundational Analysis: CAILP-Social

**Project:** Counterfactual Attention-Based Inverse Link Prediction for Social Network Sparsification  
**Date:** 2026-07-24  
**Status:** Phase 1 complete — explicit prior-art contrast required before Phase 2  
**Scope:** Analysis only. No novelty claims beyond what this differentiation supports. Abstract, introduction novelty paragraph, and candidate contributions list are deferred until Phase 15 results and this section remain consistent.

---

## 0. Novelty Positioning Checklist (must remain satisfied)

| Constraint | Status in this document |
|---|---|
| ILP framed as extension of Bangian Tabrizi, Jalali & Houshmand (J Big Data 2025), not independent invention | Yes — §1 |
| Black Hole (JCIM 2025) treated as optional prior, compared unmodified as baseline | Yes — §2 |
| Edge-deletion counterfactuals **not** claimed as novelty; CF-GNNExplainer contrast explicit | Yes — §3 |
| “Learn edge importance and prune” **not** claimed as novelty; NeuralSparse/PTDNet/DSpar/DropEdge named | Yes — §4 |
| Dual-view / cross-attention framed as design choice pending ablation | Yes — §5 comparison table & §5.4 |
| Surviving candidate novelty claims limited to (a)–(c) in project brief | Yes — §6 |

---

## 1. Original Inverse Link Prediction (ILP-GCN)

**Citation:** Bangian Tabrizi, E., Jalali, M., & Houshmand, M. (2025). Inverse link prediction with graph convolutional networks for knowledge-preserving sparsification in cheminformatics. *Journal of Big Data*, 12:176. https://doi.org/10.1186/s40537-025-01220-8  
**Code:** https://github.com/MehrdadJalali-AI/InverseLinkPredcition  
**Domain:** Metal–Organic Framework (MOF) similarity graphs (MOFGalaxyNet).

### Research objective
Sparsify dense chemical-similarity networks by identifying and removing redundant edges while preserving community structure and downstream property-prediction utility (e.g., pore-limiting diameter classification).

### Mathematical formulation (as published)
Given \(G=(V,E)\) with initial similarity weights \(W_{\mathrm{initial}}(e_{ij})\):

1. **GCN link score**  
   \[
   S_{\mathrm{GCN}}(e_{ij}) = \mathrm{GCN}(v_i, v_j \mid \theta)
   \]

2. **Inverse / ILP-derived weight** (high predicted link score ⇒ low retention priority under the published inverse mapping)  
   \[
   W_{\mathrm{ILP\text{-}GCN}}(e_{ij}) = \frac{\alpha}{S_{\mathrm{GCN}}(e_{ij}) + \varepsilon}
   \]

3. **Dual-weight fusion**  
   \[
   W_{\mathrm{final}}(e_{ij}) = \gamma\, W_{\mathrm{initial}}(e_{ij}) + (1-\gamma)\, W_{\mathrm{ILP\text{-}GCN}}(e_{ij})
   \]

4. **Modularity-guided thresholding**  
   Edges with \(W_{\mathrm{final}}(e_{ij}) < \varphi\) are pruned. Threshold \(\varphi^*\) is selected where modularity \(Q(\varphi)\) is maximized or plateaus (post-scoring selection / sparsification criterion, not a GCN training loss).

5. **Weighted modularity used in evaluation / pruning guidance**  
   \[
   Q = \frac{1}{2m}\sum_{ij}\Big(A_{ij}W_{\mathrm{final}}(e_{ij}) - \frac{k_i^{W} k_j^{W}}{2m}\Big)\delta(c_i,c_j)
   \]

### Model input and output
- **Input:** MOF similarity graph (node chemical/structural descriptors; edge similarity weights from linker fingerprints + metal cosine similarity).  
- **Output:** sparsified graph \(G'=(V',E')\) with \(E'\subset E\); disconnected nodes may also be dropped (\(V'\subseteq V\)).

### Role of node embeddings
GCN embeddings of MOF nodes drive the pairwise link score \(S_{\mathrm{GCN}}\). Embeddings are task-agnostic w.r.t. a dedicated edge-harm target: they support standard link scoring.

### Role of edge features
Initial edge weights encode domain similarity. No dedicated edge GNN / line-graph encoder. No explicit structural edge-feature vector beyond similarity.

### Scoring mechanism
Edge importance for retention is **not** a directly supervised deletion-harm score. It is a **composite of** (i) inverse of GCN link probability / score and (ii) initial similarity weight. Low \(W_{\mathrm{final}}\) ⇒ candidate for removal.

### Loss functions
Standard supervised / link-prediction-style GCN training on the similarity graph (implementation details in ESI / notebook). Modularity is **not** the GCN training objective; it guides threshold selection and structural evaluation.

### Threshold-selection mechanism
Sweep \(\varphi\); choose values where modularity peaks/stabilizes while allowing substantial edge reduction (e.g., reported operating points 0.90, 0.95, 0.98).

### Node or edge reduction strategy
Primarily **edge pruning** by weight threshold; **nodes may be removed** when left without retained edges. This mixes edge sparsification and incidental node reduction — CAILP Track A must **not** copy this coupling.

### Structural-preservation mechanism
Modularity-aware thresholding; community structure emphasized as a preservation target in evaluation.

### Downstream tasks
MOF property prediction (PLD categories) with GCN, GraphSAGE/GraphRAGE, DNN, GBT, LR, NB on original vs sparsified graphs.

### Evaluation metrics
Accuracy, Kappa, confusion matrices, modularity, community count, degree/density reductions, runtime.

### Limitations (for transfer)
- Domain-specific similarity construction (SMILES fingerprints, metal descriptors).  
- Inverse-of-link-score ≠ measured deletion harm.  
- Threshold/modularity coupling can favor community separation without guaranteeing predictive or fairness preservation.  
- Node drop coupled to edge prune complicates fair comparison to pure edge-budget methods.  
- No uncertainty-aware pruning; no multi-dimensional counterfactual teacher; no dual-view edge encoder.

### Scalability challenges
GCN link scoring over dense similarity graphs; threshold sweeps; large MOFGalaxyNet (~12.5k nodes, ~415k edges at similarity 0.9).

### Reproducibility concerns
Notebook-centric original code; need pinned commits, identical seeds, and a faithful reimplementation under CAILP’s shared protocol.

### Transferable to social networks
- Core ILP *question*: “which existing edges are least important?”  
- GCN (or other MPNN) edge scoring from node features.  
- Dual-weight idea (combine learned score with structural prior).  
- Budgeted pruning + community-aware evaluation.

### Domain-specific — do **not** transfer directly
- MOF similarity matrix construction and chemical fingerprint features.  
- PLD-specific label taxonomy.  
- Concurrent node dropping as default sparsification behavior.  
- Similarity-threshold graph construction as the primary data model (social benchmarks arrive as attributed graphs).

### Explicit determination (project checklist)
| Question | Answer |
|---|---|
| How was the original ILP score calculated? | Inverse of GCN link score, fused with initial similarity weight: \(W_{\mathrm{final}}=\gamma W_{\mathrm{initial}}+(1-\gamma)\alpha/(S_{\mathrm{GCN}}+\varepsilon)\). |
| Was edge importance directly supervised? | **No** — not with measured deletion harm. Supervision is link-prediction-style; importance is derived by inversion + fusion. |
| Was modularity part of training or post-processing? | **Post-scoring / sparsification selection and evaluation**, not the GCN training loss. |

---

## 2. Black Hole Strategy (JCIM 2025)

**Citation:** Jalali, M., Dinga Wonanke, A. D., Friederich, P., & Wöll, C. (2025). The Black Hole Strategy: Gravity-Based Representative Sampling for Frugal Graph Learning on Metal–Organic Framework Networks. *Journal of Chemical Information and Modeling*, 65(20), 10885–10902. https://doi.org/10.1021/acs.jcim.5c01518  
**Code:** https://github.com/MehrdadJalali-AI/BlackHole  
**Domain:** MOF networks; frugal / representative learning.

### Research objective
Construct compact, informative **node subsets** (representative sampling) via a gravity-inspired scoring mechanism, preserving structural and property diversity for downstream learning under reduced memory/compute.

### Mathematical formulation (conceptual)
Node “mass” / influence from structural (and possibly property) descriptors; gravity-like attraction between nodes; community-aware retention (Louvain). Exact mass formulae and exponents are as published in JCIM 2025 / repository — CAILP must reimplement the **unmodified** formulation as Optional Module A baseline (variant 7 / A32).

Illustrative gravity prior (CAILP optional variant, *not* claimed identical to the paper until matched):  
\[
g_{ij} = \frac{m_i m_j}{(d_{ij}+\varepsilon)^p}
\]

### Model input and output
- **Input:** graph + node descriptors.  
- **Output:** reduced **node set** (and induced edges). Primary reduction unit is the **node**.

### Role of node embeddings / edge features
Mass and gravity act on nodes; edges are retained incidentally via the induced subgraph. Not an edge-importance ILP scorer.

### Scoring mechanism
Gravity / mass ranking of nodes (and associated influential connections in some repository descriptions). Distinct from ILP edge scoring.

### Loss functions
Sampling heuristic / scoring procedure; not a counterfactual edge-harm regression loss.

### Threshold / reduction strategy
Retain top-mass or community-representative nodes to a node budget.

### Structural-preservation mechanism
Community detection (Louvain) + preferential retention of influential nodes.

### Downstream tasks
Graph learning on frugal MOF subsets (e.g., GraphSAGE).

### Evaluation metrics
Accuracy under reduced data, modularity, structural metrics, runtime/memory.

### Limitations
- **Node sampling ≠ edge sparsification** (Track B vs Track A).  
- Gravity priors can introduce **hub bias**.  
- Chemical-network justification does not automatically transfer to social graphs.  
- Must not be renamed as the main CAILP method.

### Scalability
Cheaper than full counterfactual teachers; still needs careful community computation on large graphs.

### Reproducibility
Pin JCIM formulation + git commit; compare CAILP mass variants against **unmodified** Black Hole mass.

### Transferable
Idea of node representativeness as an *optional auxiliary prior* for edge scoring; community-aware retention.

### Domain-specific — do not transfer blindly
MOF property mass terms; chemical similarity distances; any assumption that gravity is theoretically justified for social networks without empirical test (RQ8, RQ10).

### Explicit determination
| Question | Answer |
|---|---|
| Node sampling or edge pruning? | **Primarily node sampling / representative selection** (Track B). |
| Can node mass help ILP beyond original Black Hole? | **Unknown a priori** — must test analytical/learnable mass vs degree, PageRank, and unmodified Black Hole (RQ8, A19–A24, A32). |
| Gravity theoretically justified for social nets? | **Not assumed.** Social networks have different generative mechanisms; treat as hypothesis. |
| Hub bias risk? | **Yes** — high-degree / high-PageRank nodes may be overprotected; measure class-wise degree retention and centrality skew. |

---

## 3. Counterfactual GNN Explanation Methods (critical contrast)

**Primary citation:** Lucic, A., et al. (2022). CF-GNNExplainer: Counterfactual Explanations for Graph Neural Networks. *AISTATS*, PMLR 151:4499–4511.  
**Related:** RCExplainer, CF², INDUCE-style inductive counterfactual reasoning, and other CF explanation methods that delete or mask edges to alter a local prediction.

### Research objective (CF-GNNExplainer family)
Produce a **minimal, instance-level** counterfactual explanation: the smallest edge-deletion set such that a GNN’s prediction for a **target node (or instance)** changes.

### Mathematical formulation (CF-GNNExplainer sketch)
Learn a perturbation / mask \(P\) over the (sub)graph adjacency so that  
\[
A' = A \odot P
\]  
minimizes a combination of:
- **prediction-change / fidelity** objective (encourage flipped prediction), and  
- **proximity / size** objective (prefer few deletions).

Success metrics: fidelity, explanation size, sparsity, accuracy of the explanation — **not** global structural preservation under a fixed global edge budget.

### Model I/O
- **Input:** trained GNN + instance subgraph.  
- **Output:** small set of edges whose removal flips the instance prediction.

### Role of embeddings / edge features
Uses the trained GNN’s behavior; edge importance is defined by contribution to **that instance’s prediction**, not by multi-criteria global harm.

### Scoring / supervision
Optimization toward prediction flip + minimality. No aggregation of modularity, spectral gap, group F1, etc., into a global teacher target.

### What CAILP-Social **borrows**
The **mechanics** of measuring the effect of edge deletion (counterfactual perturbation of \(A\)). This mechanic is established; **it is not the novelty**.

### What CAILP-Social does **differently** (must appear in paper §5.4)

| Dimension | CF-GNNExplainer / CF explanation | CAILP-Social |
|---|---|---|
| Objective | Flip one prediction (local explanation) | Rank edges for **global budgeted sparsification** |
| Scope | Per-instance subgraph | Whole graph under retention budget \(\rho\) |
| Supervision target | Fidelity + minimality | Multi-dimensional deletion harm: task, community, connectivity, spectral, representation, group |
| Success metric | Explanation fidelity / size | Predictive + structural + fairness + robustness + efficiency under equal budgets |
| Aggregation | Single (or few) prediction-centric terms | Explicit six-component weighted aggregate \(y_e^{\mathrm{cf}}\) |
| Constraints | Minimal perturbation | Connectivity, community backbone, minority neighborhood, etc. |

**Point-by-point contrast (required):**
1. Shared mechanic: remove edge(s), observe model/graph change.  
2. Different objective: explanation vs sparsification.  
3. Different scope: single instance vs whole graph.  
4. Different supervision: fidelity/minimality vs multi-dimensional harm.  
5. Empirical safeguard: implement **single-objective CF-style teacher** (fidelity / prediction-flip only) as named baseline A31 / RQ11; do not allow prose-only differentiation.

### Limitations of treating CF methods as sparsifiers
Optimizing for prediction flip can delete *locally* critical edges while ignoring bridges, minority connectivity, or spectral structure; minimality is the opposite of operating at 10–90% removal budgets.

### Transferable
Deletion-based counterfactual measurement; mask learning ideas (carefully, as optional GAN variants).

### Not transferable as novelty claims
“We use edge-deletion counterfactuals” as a contribution statement.

---

## 4. Learned and Classical Graph Sparsification Baselines

### 4.1 DropEdge (Rong et al., ICLR 2020)
- **Signal:** random / distribution-based edge dropping during training (regularization).  
- **Not** a persistent learned importance ranking for a fixed sparsified graph export in the original sense; include as a strong random/regularization-related baseline under equal retention protocols where applicable.  
- **Differs from CAILP:** no multi-dimensional counterfactual teacher; no constrained global pruning for structural/fairness criteria.

### 4.2 NeuralSparse (Zheng et al., ICML 2020)
- **Signal:** sparsification network selects edges from one-hop neighborhoods under a budget; optimized by **downstream task feedback**.  
- **Differs from CAILP:** task-driven selection ≠ six-component counterfactual harm; typically local \(k\)-neighbor style budgets, not global multi-criteria constrained pruning with uncertainty.

### 4.3 PTDNet (Luo et al.; “Learning to Drop” / topological denoising)
- **Signal:** parameterized topological denoising; relaxes NeuralSparse’s strict \(k\)-neighbor assumption; task-supervised edge dropping with structural regularizers (e.g., low-rank / denoising biases).  
- **Differs from CAILP:** still primarily task (+ topological denoising) driven; does not define CAILP’s multi-dimensional CF harm teacher or fairness/community constrained budgeted export protocol.

### 4.4 DSpar (degree-based / effective-resistance-motivated efficiency sparsification)
- **Signal:** structural / degree / resistance-style importance for efficient approximation.  
- **Differs from CAILP:** classical or efficiency-oriented structural signal; no learned multi-objective CF target.

### 4.5 Effective-resistance / spectral sparsification (Spielman–Srivastava and descendants)
- **Signal:** approximate spectral properties via resistance sampling.  
- **Differs from CAILP:** spectral guarantee focus; not social-task + fairness multi-objective supervision.

### Plain statement (required for Related Work and baselines table)
> “Learn edge importance from data and prune accordingly” is **not** new. NeuralSparse, PTDNet, DSpar, DropEdge, and classical spectral/resistance methods already instantiate variants of that idea. CAILP’s *candidate* contribution is the **specific counterfactual, multi-objective, uncertainty-aware formulation** of the edge-importance signal and its evaluation under a **broader** predictive/structural/fairness/robustness battery — contingent on experiments, especially vs A31 and learned baselines.

### What CAILP borrows
Budgeted edge retention; task-aware learning signals; structural heuristics as baselines; spectral/resistance ideas as comparison points.

### What CAILP does differently
Explicit six-component CF teacher + surrogate; uncertainty-aware conservative removal rules; multiply constrained global pruning; dual-view fusion as *ablated design choice*; optional mass/GAN as *rejectable* hypotheses.

---

## 5. Comparison Table

| Aspect | Original ILP (J Big Data 2025) | Black Hole Strategy (JCIM 2025) | CF-GNNExplainer / CF explanation | Learned sparsification (NeuralSparse / PTDNet / DSpar) | Proposed CAILP-Social |
|---|---|---|---|---|---|
| Primary goal | Knowledge-preserving edge prune on MOF graphs | Representative **node** sampling | Minimal CF explanation (flip prediction) | Task-robust / efficient sparsification | Global budgeted edge sparsification on social graphs |
| Reduction unit | Edges (+ incidental nodes) | Nodes (induced edges) | Few edges per instance | Edges (local or global) | Edges (Track A); nodes separate (Track B) |
| Edge score source | Inverse GCN link score ⊕ similarity | N/A (node mass/gravity) | Contribution to prediction flip | Task feedback / topo denoising / degree–resistance | Multi-dimensional CF deletion harm (6 components) |
| Direct CF supervision? | No | No | Yes (fidelity-centric) | No (except indirectly via task loss) | Yes (aggregated harm; + A31 single-objective control) |
| Modularity in training? | Threshold/eval, not GCN loss | Community-aware sampling | No | Usually no | Optional loss term + hard constraints |
| Dual-view / attention | No | No | No | Architecture-specific | Design choice; must beat concat/gate ablations |
| Uncertainty | No | No | No | Rare | Heteroscedastic / optional MC dropout, ensembles |
| Fairness / group effects | Not primary | Not primary | Not primary | Not primary | Explicit \(\Delta_{\mathrm{group}}\) + constraints |
| Domain | Cheminformatics (MOF) | Cheminformatics (MOF) | GNN explainability benchmarks | General / efficiency / robustness | Attributed social networks |
| Novelty relation to CAILP | **Direct ancestor** (extend, cite first) | **Optional prior** (Track B + Module A) | **Shared mechanic, different problem** | **Baselines; learning-to-prune not new** | Candidate claims (a)–(c) only if evidenced |

---

## 6. Defensible Candidate Novelty (pre-experiment; contingent)

After contrasts (1)–(4), only these claims may be argued — **and only if experiments support them**:

**(a)** Multi-dimensional counterfactual deletion-harm target aggregating task, community, connectivity, spectral, representation, and group effects for **global** sparsification (distinct from CF explanation).  

**(b)** Uncertainty-aware, multiply constrained pruning under a fixed global budget.  

**(c)** Ablation-driven determination of whether dual-view attention fusion, node-mass priors (incl. original Black Hole), or adversarial learning justify their complexity — with explicit expectation that any may be **rejected**.

**Not novelty:** edge-deletion counterfactuals per se; GCN→GAT; dual-view/cross-attention alone; “learn importance and prune”; Black Hole or GAN unless they clear decision rules.

---

## 7. Implications for CAILP-Social Design

1. Core method remains an **ILP** descendant even if Modules A/B fail.  
2. Teacher must compute **all six** harm components in the full model; A31 must collapse to fidelity-only for RQ11.  
3. Track A never silently drops nodes; Track B never pretends node% ≡ edge%.  
4. Reproduce original ILP-GCN scoring (inverse link score + dual weight + modularity-aware threshold analogue under **equal edge budgets**) as baseline 15.  
5. Do not draft abstract / novelty paragraph / contributions list until Phase 15 and this file still pass the §0 checklist.

---

## 8. References (seed)

1. Bangian Tabrizi, Jalali, Houshmand (2025). *J Big Data* 12:176.  
2. Jalali et al. (2025). Black Hole Strategy. *JCIM* 65(20):10885–10902.  
3. Lucic et al. (2022). CF-GNNExplainer. *AISTATS*.  
4. Rong et al. (2020). DropEdge. *ICLR*.  
5. Zheng et al. (2020). NeuralSparse. *ICML*.  
6. Luo et al. PTDNet / Learning to Drop (topological denoising).  
7. DSpar and classical effective-resistance / spectral sparsification literature (Spielman–Srivastava and descendants).  

*Full BibTeX will live in `paper/references.bib` during Phase 2+.*
