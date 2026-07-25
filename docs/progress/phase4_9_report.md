# Phase 4–9 Progress

**Date:** 2026-07-24

## Implemented
- Original ILP-GCN reproduction (`src/sparsification/original_ilp.py`, `scripts/reproduce_original_ilp.py`)
- Black Hole Track B (`src/sparsification/node_sampling.py`)
- Classical + DropEdge/DSpar/NeuralSparse/PTDNet baselines
- Core CAILP-Social + six-component CF teacher + A31 single-objective teacher
- Constrained pruning, leakage guard, unit tests (14 passed)
- Leakage audit: `docs/leakage_audit.md`

## Pilot (LastFM, seed 0) — core multi-objective CAILP
| Removal | Mode | Test Macro-F1 |
|---|---|---|
| 0.3 | budget | 0.833 |
| 0.3 | constrained | 0.833 |
| 0.5 | budget | 0.829 |
| 0.5 | constrained | 0.834 |
| 0.7 | budget | 0.796 |
| 0.7 | constrained | 0.792 |

Raw: `results/raw/cailp_lastfm_seed0.json`

## Notes
- MPS disabled (PyG GAT scatter unsupported); CPU used
- Structural neighborhood overlaps use `lightweight=True` in pilot training path for scalability
- Full CN/Jaccard features remain available for classical baselines and non-lightweight mode
