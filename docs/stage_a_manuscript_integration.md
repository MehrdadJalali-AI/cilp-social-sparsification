# Stage A → manuscript integration

**Date:** 2026-07-25  
**Scope:** Integrate frozen GitHub Stage A results into Journal of Big Data Methodology draft.  
**Stage B:** Not started.

## Source of truth

- Authoritative store: `results/processed/authoritative_results.csv` (1,890 rows)
- Freeze: `docs/stage_a_freeze_manifest.md` + `results/frozen/stage_a_github_20260725T124409Z/`
- Decision: `results/processed/multi_dim_decision.json` (LastFM / Facebook / GitHub all Supported)
- Figures: regenerated three-dataset panels in `paper/figures/fig{2,3,4,5,7}_*`
- Tables: `paper/tables/table4_structure.tex`, `s14_full.tex`, `s16_18_paired.tex`, `s21_paired_multidim.tex`, `s22_structure.tex`, `s23_table.tex`, `s19_summary.tex`

## Manuscript updates

| Section | Change |
|---|---|
| Abstract | Three datasets; remove GitHub-incomplete claim; restrained claims |
| Introduction | Three-dataset evaluation; matched Task-only as central comparison |
| Experimental setup | GitHub as completed dataset; teacher_n=40; epochs 15/15/35/15 |
| Results | GitHub predictive block; teacher table; structure; runtime |
| Discussion | Three-dataset support; small GitHub AUC Δ; teacher bottleneck |
| Conclusions | Three datasets complete; future work = adaptive sampling etc. |
| Declarations | Data availability cites 1,890 rows + freeze manifest |
| SI S14–S28 | GitHub rows, decision 4/4, no failures, freeze/repro notes |

## Decision-rule conclusion (unchanged rule)

Multi-criteria supervision satisfied the predefined support criterion on **all three datasets** (4/4 metrics each vs Task-only).  
This support is relative to the matched Task-only teacher and does **not** imply superiority over all established sparsifiers.

## Suggested Git tag

`stage-a-github-frozen`
