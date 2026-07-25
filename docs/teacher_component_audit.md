# Teacher component audit

**Implementation:** `src/counterfactual/exact_teacher.py`  
**Normalization:** `src/counterfactual/sampling.py::normalize_scores` (min–max)  
**Grid call:** `ExactCounterfactualTeacher()` / `SingleObjectiveTeacher()` with default `CFCoefficients` (not validation-tuned).

## Categories

| Criterion | Category |
|---|---|
| Task loss | Deletion-induced |
| Connectivity change | Deletion-induced |
| Representation shift | Deletion-induced |
| Group impact | Deletion-induced |
| Community proxy | Structural proxy |
| Degree-based spectral proxy | Structural proxy |

## β weights (fixed defaults, same across datasets)

| k | β_k |
|---|---|
| task | 1.0 |
| comm | 0.5 |
| conn | 1.0 |
| spec | 0.3 |
| repr | 0.5 |
| group | 0.5 |

## Community proxy (exact)

Function: `community_effect` in `src/counterfactual/exact_teacher.py`.

1. Louvain partition `C = community.best_partition(G)` computed **once** on full `G` and reused.
2. Not recomputed after each deletion.
3. `same = 1[C(u)=C(v)]`.
4. `expected = d_u d_v / (2m)`, `a_uv = 1`.
5. `local = (a_uv - expected)` if same else `|a_uv - expected|`.
6. `inter = 0` if same else `1`.
7. Return `max(0, local) + inter` (equal weight 1.0 between local term and inter flag).
8. Fallback if Louvain import fails: absolute clustering changes at endpoints after deletion.
9. Sign: higher = more harmful.
10. Normalization: min–max over teacher sample.
11. `q0 = modularity(C,G)` is computed but **unused** in the returned score.

## Degree-based spectral proxy

For all reported datasets (`|V|>400`): `Δ_spec = 1/max(d_u,1) + 1/max(d_v,1)`.  
Eigenspectrum branch unused in LastFM/Facebook.

## Aggregation

`y_cf = normalize( Σ_k β_k · normalize(Δ_k) )`  
Task-only: `y = normalize(Δ_task)`.

See also `paper/tables/teacher_component_definitions.csv`.
