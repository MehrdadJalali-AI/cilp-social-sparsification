# Manuscript change log — Stage A GitHub integration

**Timestamp (UTC):** 2026-07-25  
**Trigger:** Frozen Stage A GitHub completion (Supported, 4/4)

## Files updated

- `paper/main.tex` — abstract; data availability
- `paper/sections/introduction.tex`
- `paper/sections/experimental_setup.tex`
- `paper/sections/results.tex`
- `paper/sections/discussion.tex`
- `paper/sections/conclusions.tex`
- `paper/supplementary.tex` — S6, S15, S21, S24, S26, S28
- `paper/tables/s19_summary.tex` — GitHub surrogate means
- `paper/tables/s15_per_seed.csv` — refreshed from authoritative CSV (1,890 rows)
- `docs/stage_a_freeze_manifest.md` — created
- `docs/stage_a_manuscript_integration.md` — created
- `docs/figure_data_provenance.md` — verified three-dataset figures
- `docs/manuscript_change_log.md` — this file

## Explicitly unchanged

- Stage A numeric results and decision rule
- LastFM / Facebook authoritative values
- No Stage B / adaptive teacher sampling
- No parameter retuning
- No experiment reruns

## Language removed

- “GitHub experiments were incomplete…”
- “planned evaluation”
- “third dataset remains necessary”
- Completing GitHub as remaining journal requirement

---

## 2026-07-25 — Related-work positioning

Positioned the paper at the intersection of link prediction, inverse link prediction, and graph sparsification.

- Rewrote `paper/sections/related_work.tex` with subsections for classical LP, inverse LP, structure/spectral sparsification, learned GNN sparsifiers, utility-aware evaluation, and counterfactual explanation.
- Lightly aligned Introduction framing.
- Added bibliography entries: Lindner et al. (ASONAM 2015), Hamann et al. (SNAM 2016), Chen et al. (PVLDB 2023), Feng (DAC 2016), Li et al. SGCN (PAKDD 2020) + journal extension (2022), Sanz-Cruzado et al. (WWW Companion 2018), Mara et al. EvalNE (SoftwareX 2022), Li et al. utility-based link recommendation (Management Science 2017).

---

## 2026-07-25 — Must-fix consistency pass

- Unified train/development/test terminology (S3–S8, data availability, results).
- Fixed Fig.~5 caption to match heatmap (methods×metrics, shared colour bar).
- Defined $E_r$, $G_r$; removed undefined $\rho$ (methods + Fig.~1).
- Documented stratified teacher sampling (degree×cosine quartiles); expanded S12 baselines; clarified Wilcoxon vs $t$-CI; separated surrogate MLP (60) vs CILP scorer epochs.
- Renamed community criterion to community-boundary proxy with `inter`-term explanation.
- Editorial: feasible (not practical); amortization across budgets; Fig.~6 priors subset; abstract four-metric wording.

- Replaced “dense” with edge-rich / substantial edge volume (abstract + intro + density note in setup).
- Renamed middle split as development set; disclosed train∪dev teacher targets and Facebook teacher-size reuse; no claim of full validation isolation.
- Softened runtime claim: CILP training = teacher + scorer; not separately logged.
- Abstract: AUC Δ +0.0265 / +0.0142 / +0.0033; single non-superiority sentence.
- Holm families (i)/(ii) distinguished; avoided “preregistered.”
- GitHub: CILP > Task-only but slightly below ILP-GCN/PTDNet stated directly.
- Fig. 3 → paired forest plot; Table 3 bold/underline; structural 30/50/70 table from frozen CSV.
- Surrogate GitHub interpretation (rank vs MAE); S13 CAILP→CILP; figure captions clarified.
- Roadmap for remaining experiments: `docs/presubmission_experiment_roadmap.md`.
