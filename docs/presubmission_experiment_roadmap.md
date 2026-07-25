# Pre-submission experiment roadmap (Journal of Big Data)

Stage A remains frozen. These items strengthen systems evidence and multi-criteria methodology; they are **not** Stage B adaptive teacher sampling.

## Priority order

| # | Item | Impact | Status |
|---|---|---|---|
| 1 | Validation / development-set reuse | Reviewer risk | **Addressed in text** (development-set naming + limitation). Stronger fix: retrain teacher $\Delta_{\mathrm{task}}$ on train-only. |
| 2 | Runtime attribution | Reviewer risk | **Addressed in text**. Stronger fix: separate teacher vs scorer timers. |
| 3 | Downstream GCN speed / memory / break-even | Highest JBD impact | **Not run** |
| 4 | Leave-one-criterion-out + criterion correlations | Highest methods impact | **Not run** |
| 5 | Direct community metrics | Methods clarity | **Not run** |
| 6 | Structural multi-budget | Evaluation symmetry | **Partial:** 30/50/70 table added from frozen results |
| 7 | Full-graph 0% / utility retention | Presentation | **Not run** (grid starts at 0.1) |
| 8 | Weight-robustness mini-grid | Sensitivity | **Not run** |
| 9 | Surrogate ranking diagnostics | Surrogate section | **Interpretation updated**; extra metrics not computed |
| 10 | Authors / DOI / licence / tag | Submission hygiene | Pending |

---

## 3. Downstream break-even protocol (to implement)

For each dataset, seed $\in\{0,\ldots,9\}$, method $\in\{\mathrm{CILP},\mathrm{ILP\text{-}GCN},\mathrm{PTDNet},\mathrm{Random}\}$, removal $r\in\{0,0.3,0.5,0.7,0.9\}$:

1. Build sparsified adjacency from frozen importance ranking (or random).
2. Train the standard downstream GCN under the same epoch budget as the Stage A downstream evaluator.
3. Log:
   - seconds / epoch;
   - total training seconds;
   - peak RSS / CUDA memory if applicable;
   - adjacency storage bytes;
   - retained Macro-F1;
   - $|E_r|/|E|$.
4. Break-even:
   \[
   N_{\mathrm{break\text{-}even}}
   =
   \frac{\text{CILP construction cost (train\_seconds)}}
   {\text{downstream time}(r{=}0)-\text{downstream time}(r)}
   \]
5. Report mean$\pm$s.d.\ over seeds; do not retune Stage A sparsifiers.

Suggested script: `scripts/run_downstream_breakeven.py`  
Outputs: `results/processed/downstream_breakeven.csv`, SI table S30, Fig.~S-breakeven.

## 4. Leave-one-criterion-out protocol

On LastFM seeds $0$–$4$ (extend later):

- Full CILP teacher vs six ablations with one $\beta_k{=}0$ (renormalize remaining weights or keep others fixed—**pre-declare**).
- Pairwise Spearman among the six normalized $\Delta$ components on sampled teacher edges.
- Heatmap: criterion removed × $\{\Delta$ AUC / GC@50 / bridge@50 / min-deg@50}.

Suggested script: `scripts/run_loo_teacher_ablation.py`

## 5. Community metrics (post-hoc on frozen sparsified graphs)

At $r\in\{0.3,0.5,0.7\}$ for CILP / Task-only / ILP-GCN / PTDNet / Random:

- modularity retention $Q(G_r)/Q(G)$;
- NMI / ARI between Louvain on $G$ and $G_r$;
- intra-community edge retention.

No teacher recompute required if sparsified edge sets can be reconstructed from stored rankings / raw JSON.

## 8. Weight robustness (mini)

Five $\beta$ vectors on LastFM seeds $0$–$4$ at budgets $0.3,0.5,0.7$: current / equal / task-heavy / structure-heavy / group-heavy.  
Claim only robustness of CILP$>$Task-only direction, not optimality of $\beta$.

---

## Already completed in this revision (no new Stage A runs)

- Replaced “dense” with edge-rich / substantial edge volume.
- Development-set reuse disclosed; not claimed fully validation-isolated.
- Runtime claim softened to CILP training (teacher+scorer).
- Abstract AUC deltas; GitHub CILP below ILP-GCN/PTDNet stated directly.
- Holm families (i)/(ii) distinguished; “preregistered” avoided.
- Fig.~3 replaced by paired forest plot.
- Structural 30/50/70 table from frozen CSV.
- S13 CAILP→CILP; surrogate ranking interpretation; figure captions clarified.
