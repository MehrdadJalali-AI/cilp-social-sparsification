# Analysis plan (fixed before the final test evaluation)

Committed while only the criterion cache (no labels beyond training) and the blinded selection phase
(validation only; test mask emptied) were running; no test metric of this pipeline existed at commit time.

## 1. Protocol

| Stage | Labels | Edges | Node features |
|---|---|---|---|
| Teacher encoder pre-training | train | full graph | all nodes (transductive) |
| Teacher Δ_task / Δ_group | **train only** | full graph ± probed edge | all nodes |
| Δ_conn, Δ_comm, Δ_spec, Δ_repr | none | full graph ± probed edge | all nodes (Δ_repr) |
| Surrogate + scorer fitting | train (scorer CE term) + teacher targets | full graph | all nodes |
| Teacher-size selection | **validation** Macro-F1 at r = 0.5 | sparsified graph | all nodes |
| Downstream GCN training | train; validation selects the epoch | sparsified graph | all nodes |
| Test evaluation | test (once, after lock) | sparsified graph | all nodes |
| Structural / community / minority metrics | labels only for the minority-class definition (evaluation) | original and sparsified graphs | — |

The held-out partition is called **validation** everywhere.

## 2. Teacher-size selection rule

For each dataset and each teacher t ∈ {CILP, Task-only} separately (equal tuning budget):
n*_t = argmax_{n ∈ {20, 40, 80, 120, 240}} mean over seeds 0–9 of validation Macro-F1 at r = 0.5.
Ties within 1e-4 → smaller n (cheaper). The primary comparison uses each teacher at its own n*.
Secondary (sensitivity): both teachers evaluated on test at every n (matched-n curves), after the lock.
Ablation and weight variants use CILP's n* per dataset.

## 3. Endpoints

* **Primary:** sparsity–Macro-F1 AUC = trapezoidal area of test Macro-F1 over r ∈ {0.1,…,0.9}
  (the Stage A definition). Contrast: CILP − Task-only, paired by seed.
* **Smallest effect size of interest (SESOI):** ΔAUC = 0.008 (= 1 percentage point of Macro-F1
  averaged over the 0.8-wide removal range). Effects with |Δ| < SESOI are reported as practically
  negligible regardless of significance.
* **Secondary:** giant-component ratio, bridge retention, minority-degree retention at r = 0.5;
  community metrics at r = 0.5 (conductance ratio of the reference Leiden partition, NMI and ARI of
  re-detected Leiden partitions, modularity retention). Reported per budget in the supplement.

## 4. Hypothesis families and multiplicity

* **Family P (per dataset, confirmatory):** CILP vs Task-only on AUC, GC@0.5, bridge@0.5,
  minority@0.5 (4) + CILP vs each baseline on AUC (Random, ILP-GCN, NeuralSparse, PTDNet,
  Resistance proxy, GASSIP-adapted, MoG-adapted, PSGNN-reimpl, NetworKit Local Degree,
  NetworKit Local Similarity: 10) = **14 hypotheses**, one Holm correction per dataset.
* **Sensitivity:** one Holm correction across all 42 hypotheses (3 datasets).
* **Family C (secondary):** community metrics at r = 0.5, CILP vs Task-only and CILP vs ILP-GCN
  (4 metrics × 2 = 8 per dataset), separate Holm per dataset.
* **Ablation family (exploratory):** each variant vs full CILP on AUC; Holm per dataset; reported
  as exploratory.
* Test: two-sided exact Wilcoxon signed-rank on the 10 seed-aligned differences (SciPy,
  zero_method="wilcox"). Reported for every comparison: mean paired difference, 95% paired t-interval,
  d_z, wins/ties/losses, raw p, Holm-adjusted p.
* **Floor:** with n = 10 the smallest exact two-sided p is 2/2^10 = 0.00195. In a 14-hypothesis Holm
  family the first step needs p ≤ 0.05/14 = 0.00357, which only a 10/0 sign pattern attains; in the
  42-hypothesis family the first step needs p ≤ 0.00119 < 0.00195, so **no hypothesis can be rejected**
  in the cross-dataset sensitivity family with n = 10. This is stated, not worked around.

## 5. Decision rule

On a dataset, "CILP improves on Task-only" requires Holm-adjusted p < 0.05 in Family P **and**
mean ΔAUC > 0. It is called practically relevant only if the 95% CI lower bound exceeds 0 and the
mean exceeds the SESOI. Structural claims are made metric by metric. Statements about baselines use
"comparison under disclosed constraints" for adapted / simplified methods.

## 6. Order of operations

1. Δ cache (done before this file) → 2. selection (validation only) → 3. write and commit
`configs/selected/locked.json` → 4. final runs (test) for CILP, Task-only, ablations, weights,
baselines, full graph → 5. timing-grade solo runs → 6. analysis per this plan.
