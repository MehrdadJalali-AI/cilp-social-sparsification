# LastFM Ablations A1–A32 (seeds 0–4)

Budgets evaluated: 0.3 / 0.5 / 0.7. Adversarial variants A26–A28 skipped (module rejected). Fusion architecture held at **concat** for RQ11 A31 contrast.

## Macro-F1 (mean±std over seeds)

| Ablation | F1@30% | F1@50% | F1@70% |
|---|---|---|---|
| A1_full_core | 0.756±0.048 | 0.717±0.060 | 0.698±0.063 |
| A2_original_ilp | 0.765±0.035 | 0.731±0.022 | 0.682±0.035 |
| A8_concat | 0.709±0.054 | 0.687±0.080 | 0.624±0.063 |
| A9_gated | 0.721±0.053 | 0.691±0.073 | 0.655±0.064 |
| A10_cross_attn | 0.740±0.050 | 0.725±0.061 | 0.675±0.059 |
| A19_no_mass | 0.737±0.024 | 0.729±0.046 | 0.630±0.077 |
| A20_analytical_mass | 0.690±0.060 | 0.654±0.051 | 0.593±0.035 |
| A23_degree_prior | 0.738±0.045 | 0.686±0.089 | 0.621±0.116 |
| A24_pagerank_prior | 0.759±0.030 | 0.756±0.053 | 0.650±0.067 |
| A25_no_adversarial | 0.690±0.091 | 0.670±0.096 | 0.605±0.082 |
| A31_single_obj_concat | 0.687±0.047 | 0.701±0.095 | 0.630±0.075 |
| A31_single_obj_gated | 0.665±0.055 | 0.623±0.074 | 0.571±0.065 |
| A32_black_hole_mass | 0.723±0.044 | 0.744±0.060 | 0.659±0.068 |

## RQ11 fusion-fixed (concat)

Paired seed-wise Δ Macro-F1 (multi − A31):

| Contrast | @30% | @50% | @70% |
|---|---|---|---|
| A1_full_core − A31_concat | +0.069 (5/5) | +0.016 (4/5) | +0.068 (4/5) |
| A8_concat − A31_concat | +0.022 (4/5) | −0.015 (3/5) | −0.006 (2/5) |

**Interpretation:** With fusion held fixed, six-component vs fidelity-only is **not uniformly decisive** on the ablation grid. Prefer the full multi-seed budget curve (0.1–0.9) for the primary RQ11 call; treat ablation as sensitivity evidence.

## Mass / Black Hole (A32)

A32 @50% (0.744) is competitive with A24 pagerank (0.756) and above A20 analytical (0.654). Not a clear rejection, but also not required for the core claim pending full multi-dataset comparison.
