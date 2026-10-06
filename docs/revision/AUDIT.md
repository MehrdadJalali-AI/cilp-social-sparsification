# R1 Step 0 audit (NEUCOM-D-26-17771)

Audit date 2026-10-04. Everything below was established from code inspection or new measurements on
this machine; nothing is inferred from the manuscript text alone.

## 0.1 Inventory

| Item | Location | Notes |
|---|---|---|
| ORIGINAL grid entry point | `scripts/run_full_grid.py` (`GRID_CFG`, `score_cailp`, `eval_importance`) | CILP = `cailp_multi`, Task-only = `cailp_a31` |
| Stage B baselines | `scripts/run_stage_b_baselines.py`, `src/sparsification/baselines/{gassip_adapted,mog_adapted,psgnn_reimpl}.py` | **not in the public repository** (public HEAD 664d45f); now committed locally (5356ddc) |
| Teacher | `src/counterfactual/exact_teacher.py`, `sampling.py`, `surrogate.py` | |
| Scorer | `src/models/cailp.py` + `edge_encoder.py`, `line_graph_encoder.py`, `importance_decoder.py` | |
| Downstream | `src/tasks/node_classification.py::train_node_classifier` | 2-layer GCN, validation-selected epoch |
| Configs | `GRID_CFG` in `run_full_grid.py` (authoritative); YAMLs under `configs/` were pilots only | |
| Seeds / splits | `data/splits/{dataset}_seed{0..9}.pt`, stratified 60/20/20 | checksums match `CHECKSUMS.sha256` |
| Processed graphs | `data/processed/{lastfm,facebook}.pt` present; `github.pt` was missing | rebuilt from `data/raw/github` — SHA-256 identical to the recorded checksum (99e25d20…) |
| Cached teacher outputs | **none** in the ORIGINAL pipeline (Δ recomputed inside every run) | now `results/cache/teacher_deltas/` |
| Result stores | `results/processed/authoritative_results.{csv,json}` (1,890 rows), `results/processed/recent_baselines/stage_b_results.csv` (810 rows), `results/raw/grid/*.json` | archived read-only in `results/archive_original_protocol/` with `MANIFEST.sha256` (511 files) |
| Table / figure scripts | `scripts/generate_paper_tables.py`, `generate_paper_figures.py`, `regenerate_si_tables.py`, `generate_stage_b_tables.py`, `build_authoritative_results.py` | |

## 0.2 Environment

* Code: public `664d45f` + local snapshot `5356ddc` (tag `r1-original-protocol`, local only); revision commits on top (see ledger).
* Hardware: Apple M2 Pro, 10 logical cores, 32 GB RAM; macOS 26.6.2; CPU execution only.
* Python 3.9.6; PyTorch 2.4.0; PyTorch Geometric 2.6.1; NetworkX 3.2.1; NumPy 1.26.4; SciPy 1.13.1;
  scikit-learn 1.6.1; python-louvain 0.16; added in R1: networkit 11.0.1, python-igraph 1.0.0, leidenalg 0.12.0.
* The hardware that produced the ORIGINAL rows is not recorded in those rows (S27 already says so).

## 0.3 Δ caching

The ORIGINAL code had no Δ cache. R1 adds `scripts/revision/build_teacher_cache.py`: for each
(dataset, seed) it stores raw Δ for the union of stratified teacher samples n ∈ {20,40,80,120,240} ∪
{ORIGINAL n} (≈260 edges) and 200 held-out uniformly drawn probe edges, plus the teacher encoder.
Validation: on LastFM seed 0 the cached means over the ORIGINAL n=50 sample reproduce the archived
`effects_means` of the submitted run for all six criteria to the printed precision
(`tests/test_revision.py` adds exact per-edge equivalence tests).
Cost of the cache (≈460 probes per seed, run concurrently with other jobs): LastFM 18–52 s,
Facebook 106–151 s, GitHub 170–236 s, PubMed 53–65 s per seed.

## 0.4 Why Task-only was "slower" than CILP (255 vs 198 s LastFM; 1635 vs 1437 s Facebook)

From code: the Task-only path (`SingleObjectiveTeacher`) executes **exactly the same work** as CILP — it
computes all six criteria (`ExactCounterfactualTeacher.score_edges`) and only changes the final
aggregation line — and the scorer training loop is identical. There is no algorithmic reason for a difference.

From the archived run files: the grid ran as **two concurrent processes** (seeds 0–4 and 5–9; e.g. LastFM
seeds 0 and 5 finished in the same minute), alongside other jobs (Facebook diagnostic, ablations). The same
code took 92–321 s (CILP, LastFM) and 100–491 s (Task-only, LastFM) across seeds. Within each process
Task-only ran immediately before CILP. Measured in isolation on seed 0 LastFM (single process, this machine):
CILP 45.3 s, Task-only 55.0 s — a difference located entirely in the scorer's `build_edge_views`, whose work
does not depend on the teacher. **Conclusion:** the ORIGINAL runtime differences between CILP and Task-only
are not attributable to the method; they reflect machine contention and run-to-run variance. The REVISED
timing-grade runs are executed solo with component timers.

Where the ORIGINAL time actually went (cProfile, LastFM seed 0, CILP, 45.3 s): scorer epochs 27.3 s
(per-epoch recomputation of line graph 7.8 s, edge feature matrix 8.2 s, backward 7.8 s); teacher probes 8.2 s
(of which 4.4 s are NetworkX graph copies in Δ_conn). On Facebook/GitHub the per-epoch rebuilding of the
dense |x_i−x_j|⊕x_i⊙x_j edge-feature matrix (6.4 GB / 9.3 GB) dominates. R1 caches these parameter-independent
views (identical outputs) and stores the Facebook/GitHub edge-feature matrix sparsely (≈99.6% zeros; outputs
equal to ≤1e-4 after training, tested).

## 0.5 Further findings (all verified)

1. **Label access (R2.1).** Δ_task used cross-entropy on train∪validation and Δ_group used worst-class F1 on
   validation only — the manuscript mentioned only the former. Switching Δ_task to train-only labels changes the
   teacher ranking moderately (see item 2 for per-dataset Spearman correlations).
2. **Δ_group is almost always zero.** Across all cached probes (≈460 per seed) it is zero on every edge of
   LastFM (both protocols) and non-zero on at most 0.9% (Facebook), 0.7% (GitHub) and 1.5% (PubMed) of
   probed edges in any seed: deleting a single edge rarely lowers the frozen teacher's worst-class F1. After
   min–max normalisation the group term is therefore a sparse indicator of a handful of edges (and exactly
   zero on LastFM, where `loo_group` equals full CILP by construction).
   **Δ_conn is non-zero only for bridges** (7.4% of probed edges on LastFM, 1.7% on Facebook and GitHub,
   21.5% on PubMed), i.e. it acts as a bridge indicator plus component-size terms.
   Train-only vs train∪validation Δ_task: Spearman 0.84 (LastFM), 0.82 (Facebook), 0.75 (GitHub),
   0.83 (PubMed), averaged over seeds.
3. **Surrogate predicted in train mode.** `score_cailp` called the surrogate without `.eval()`, so dropout
   (p=0.2) was active when y_full was generated: the scorer's regression targets were stochastic. REVISED
   runs use eval mode; ORIGINAL-replay runs reproduce train mode for attribution.
4. **Unseeded Louvain** in the community-boundary proxy; REVISED seeds it per split seed.
5. **Float32 precision is not an issue:** Δ_task in float32 vs float64 agrees with Spearman ≥ 0.998
   (a hypothesis we tested and rejected).
6. The "surrogate agreement" reported in S19 is **in-sample** (on the n edges the surrogate was fit to).
   R1 adds out-of-sample agreement on 200 held-out probed edges.
7. `structural_metrics` bridge preservation = |bridges(G) ∩ bridges(G_r)| / |bridges(G)|, which equals the
   fraction of original bridges retained (a bridge kept in a subgraph remains a bridge).
8. Minority group = smallest class by node count over the full label vector (evaluation-only use).

## 0.6 Teacher sizes (R6 Q2)

* Facebook n = 120: `scripts/run_facebook_diagnostic.py`, seed 0, n ∈ {20,40,80,120}, ranked by validation
  Macro-F1 at r = 0.5 (log `results/logs/facebook_diagnostic.log`) — under the ORIGINAL label access.
* LastFM n = 50 and GitHub n = 40: **no rationale is recorded** in code, configs, logs or docs. The pilot
  YAML used 60 and `facebook_lite.yaml` used 40; `docs/github_leakage_audit.md` states GitHub's value was
  "frozen a priori". We do not reconstruct a rationale. R1 replaces all three by one documented rule
  (`docs/revision/PREREGISTRATION.md` §2).

## 0.7 Baseline fidelity audit (R2.5, R5.4)

In `scripts/run_full_grid.py::score_learned_baseline` (NeuralSparse, PTDNet) and in
`src/sparsification/baselines/{gassip_adapted,mog_adapted}.py`, the node encoder runs on the *unmasked*
graph and the classifier loss is computed from those embeddings; the edge scores enter the loss only through
a mean-score term (NeuralSparse −0.01·mean(s); PTDNet +0.1·mean(s); GASSIP curriculum MSE on mean(s) and
−0.01·mean(s·(1−difficulty)); MoG −0.005·mean(s)). Hence no task gradient reaches the edge head: these
four rankers are **task-decoupled** functions of GCN embeddings (plus a degree prior for GASSIP) and do not
reproduce the defining mechanism (task-driven edge masks that gate aggregation) of the original methods.
PSGNN-reimpl ranks by predicted-label similarity and is task-informed. ILP-GCN trains a link predictor and
ranks by γ·1 + (1−γ)·minmax(1/s), i.e. *low* predicted link probability ⇒ *higher* retention weight.
The REVISED runs keep these implementations unchanged (continuity with the ORIGINAL rows) and disclose the
defect; NetworKit Local Degree / Local Similarity are added as verified-library comparators.
