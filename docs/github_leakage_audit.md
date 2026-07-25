# GitHub leakage audit (Stage A)

## Splits

- Serialized masks: `data/splits/github_seed0.pt` … `github_seed9.pt`
- Generation: `scripts/build_splits.py` (class-stratified 60/20/20)
- Seed 0 verified: disjoint train/val/test; union covers all 37,700 nodes; fractions 0.6/0.2/0.2
- Class counts seed 0: train (0:16777, 1:5843), val (0:5592, 1:1948), test (0:5592, 1:1948)

## Shared masks

`scripts/run_full_grid.py::load_data` loads the same `{dataset}_seed{seed}.pt` for every method.

## Teacher / fitting

- Task loss and group impact use train∪val / val only (`exact_teacher.py`)
- No GitHub-specific hyperparameter selection from test
- `GRID_CFG["github"]` frozen a priori (teacher_n=40, epochs as documented)
- All budgets for a seed share one scorer / one split

## Test access

Test Macro-F1 computed only in `eval_importance` after scoring/pruning.
