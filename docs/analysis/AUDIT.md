# Code audit and implementation notes

Everything below was established from code inspection or measurements on the machine used for the experiments
(Apple M2 Pro, 10 cores, 32 GB RAM, macOS 26.6.2, CPU only; Python 3.9.6, PyTorch 2.4.0, PyTorch Geometric 2.6.1,
NetworkX 3.2.1, NumPy 1.26.4, SciPy 1.13.1, scikit-learn 1.6.1, python-louvain 0.16, NetworKit 11.0.1,
python-igraph 1.0.0, leidenalg 0.12.0).

## 1. Components

| Item | Location | Notes |
|---|---|---|
| Stage A grid runner (reference implementation) | `scripts/run_full_grid.py` (`GRID_CFG`, `score_cailp`, `eval_importance`) | CILP = `cailp_multi`, Task-only = `cailp_a31` |
| Experiment pipeline | `src/pipeline/`, `scripts/pipeline/` | criterion cache, selection, final runs, baselines, timing, analysis |
| Teacher | `src/counterfactual/exact_teacher.py`, `sampling.py`, `surrogate.py`; `src/pipeline/teacher.py` | |
| Scorer | `src/models/cailp.py` (+ encoders, decoder); `src/pipeline/scorer.py` | |
| Downstream | `src/tasks/node_classification.py::train_node_classifier` | 2-layer GCN, validation-selected epoch |
| Baselines | `src/sparsification/`, `scripts/run_full_grid.py`, `scripts/pipeline/run_baselines.py` | incl. NetworKit Local Degree / Local Similarity |
| Splits | `data/splits/{dataset}_seed{0..9}.pt`, stratified 60/20/20 | checksums in `CHECKSUMS.sha256` |
| Results | `results/main/` | per-run JSON, tables, ledger |

## 2. Criterion cache

`scripts/pipeline/build_teacher_cache.py` stores, for each dataset and seed, the raw criterion values Δ_k(e) for
the union of the stratified teacher samples n ∈ {20, 40, 80, 120, 240} ∪ {Stage A n} (≈260 edges) and for 200
uniformly drawn held-out probe edges, together with the teacher encoder. Ablation and weight variants change only
the target y_e built from these values. On LastFM seed 0 the cached values reproduce the Stage A teacher exactly;
`tests/test_pipeline.py` checks per-edge equivalence with the reference teacher and scorer.

## 3. Label access

In the pipeline (`train` setting, used for all reported results) Δ_task is the change in mean cross-entropy on
**training** nodes and Δ_group the drop in worst-class F1 on **training** nodes; the validation set is used only for
teacher-size selection and downstream epoch selection. The Stage A teacher computed Δ_task on train∪validation and
Δ_group on validation nodes; this is available as the `trainval` sensitivity setting
(`scripts/pipeline/run_experiment.py --stage-a-setting`). Train-only and train∪validation Δ_task rank probed
edges similarly but not identically (Spearman 0.84 LastFM, 0.82 Facebook, 0.75 GitHub, 0.83 PubMed).

## 4. Implementation details checked

1. **Surrogate prediction mode.** The Stage A runner predicted the extended targets with the surrogate still in
   train mode (dropout p = 0.2 active), so the scorer's targets were stochastic. The pipeline predicts in
   evaluation mode.
2. **Louvain seed.** The community-boundary proxy uses a Louvain partition; the pipeline seeds it with the split
   seed (the Stage A call was unseeded).
3. **Group criterion.** Δ_group is zero on every probed edge of LastFM and non-zero on at most 0.9% (Facebook),
   0.7% (GitHub) and 1.5% (PubMed) of probed edges, so it contributes almost nothing to the composite target.
4. **Connectivity criterion.** Δ_conn is non-zero only for bridges (7.4% of probed edges on LastFM, 1.7% on
   Facebook and GitHub, 21.5% on PubMed).
5. **Numerical precision.** Δ_task in float32 and float64 agree (Spearman ≥ 0.998).
6. **Surrogate agreement.** The pipeline reports out-of-sample agreement on 200 held-out probed edges in addition
   to in-sample agreement.
7. **Bridge retention.** |bridges(G) ∩ bridges(G_r)| / |bridges(G)| equals the fraction of original bridges kept,
   since a bridge kept in a subgraph remains a bridge.
8. **Minority group.** Smallest class by node count over the full label vector (used only for evaluation).

## 5. Runtime

The Task-only and CILP teachers perform the same computations (all six criteria are evaluated for both; only the
final aggregation differs), so their cost differs only by run-to-run variance. Profiling the reference runner
(LastFM seed 0) showed that most time went to per-epoch recomputation of parameter-independent quantities (line
graph, the dense |x_i − x_j| ⊕ x_i ⊙ x_j edge-feature matrix, structural features) and to NetworkX graph copies in
Δ_conn. The pipeline computes these once (identical outputs) and stores the Facebook/GitHub edge-feature matrix
sparsely (≈99.6% zeros; outputs equal within 1e-4 after training, tested). Reported costs come from dedicated
runs executed alone on the machine (`scripts/pipeline/run_timing.py`).

## 6. Baseline implementations

In the NeuralSparse and PTDNet implementations (`scripts/run_full_grid.py::score_learned_baseline`) and in
GASSIP-adapted and MoG-adapted (`src/sparsification/baselines/`), the node encoder runs on the unmasked graph and
the classification loss is computed from those embeddings; the edge scores enter the loss only through a
mean-score term. No task gradient reaches the edge scorer, so these four rankers are task-decoupled and do not
reproduce the task-driven edge masks of the published methods. PSGNN-reimpl ranks by predicted-label similarity.
ILP-GCN ranks by γ·1 + (1−γ)·minmax(1/s), i.e. edges with low predicted link probability are retained
preferentially. NetworKit Local Degree and Local Similarity are verified library implementations, ranked by score
with seeded tie-breaking to meet the exact budget.

## 7. Teacher sizes

The pipeline selects the teacher size for each dataset and teacher by one rule (validation Macro-F1 at 50% removal
over n ∈ {20, 40, 80, 120, 240}, seeds 0–9; `docs/analysis/PREREGISTRATION.md`). The Stage A grid used fixed sizes
(LastFM 50, Facebook 120 from a seed-0 diagnostic, GitHub 40); no selection procedure is recorded for 50 and 40.
