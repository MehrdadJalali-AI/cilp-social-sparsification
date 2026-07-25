# Leakage Audit — CAILP-Social

**Status:** Protocol frozen for pilot (Phase 9/10). Enforce via `LeakageGuard` in sparsification scripts.

## Data roles

| Asset | Allowed use |
|---|---|
| Train node labels | Teacher task/group effects; CAILP task loss; downstream training |
| Val node labels | Teacher effects; early stopping; β/λ/τ selection; surrogate quality |
| Test node labels | **Final evaluation only** (after sparsifier frozen) |
| Full graph topology for Track A NC | Available for sparsification (transductive NC setting) |
| Link-prediction positives | Train edges only for sparsifier; val/test edges held out |
| Val/test LP negatives | Split-specific; never reused across splits |

## Graphs

| Stage | Graph |
|---|---|
| CAILP / baseline training (NC Track A) | Full processed graph with train/val supervision only |
| Sparsification input | Same; scores must not use test labels |
| Downstream NC eval | Sparsified graph; test labels for metrics only |
| LP sparsification | **Training edge graph only** (`train_edge_index`) |

## Threshold / hyperparameter selection

- Selected on **validation** only.
- Test performance must not choose seeds, thresholds, or budgets for reporting.

## When test labels are first accessed

After sparsified graph is produced and locked; first call in `train_node_classifier` / `evaluate_*` during final eval. `LeakageGuard.unlock_test_for_eval()` marks this phase.

## When test edges are first accessed (LP)

Only in LP evaluation after sparsifying the training graph.

## Forbidden practices (checklist)

- [ ] Test labels in edge scoring / CF teacher
- [ ] Test performance for threshold selection
- [ ] Test edges to train importance model
- [ ] Val/test edges in LP sparsification graph
- [ ] Best-seed shopping
- [ ] Reporting only successful runs
- [ ] Larger tuning budget for CAILP than baselines
- [ ] Feature normalization using test statistics
- [ ] Counterfactual labels from test set

## Implementation hooks

- `src/utils/splits.py::LeakageGuard`
- Scripts call `lock_test()` during sparsify and `unlock_test_for_eval()` only for metrics
- Unit test: `tests/test_core.py::test_forbidden_test_label_access`

## Pilot log

Recorded automatically in result JSON under `leakage` keys.
