# Phases 11–17 Status

**Date:** 2026-07-24

## Completed in this session
- Full software architecture + configs + Docker/Makefile/README
- Dataset download/audit/splits for Facebook, LastFM, GitHub
- Unit tests: **14 passed**
- Leakage audit + protocol freeze
- LastFM pilot: CAILP multi-obj, A31, classical baselines, original ILP, Track B (BH/degree/PR), adversarial variant
- Tables/figures pipeline; LaTeX draft with provisional pilot table
- Facebook lite run started in background

## Honest scientific status
- Core ILP extension **runs** and beats random on LastFM pilot.
- RQ11 **mixed** on one seed — multi-objective not yet cleared as headline novelty.
- Optional adversarial/mass modules **not** accepted into final method.
- Abstract / final novelty paragraph still deferred pending multi-seed × multi-dataset evidence.

## Remaining for a submission-ready paper
1. Seeds 1–9 on LastFM + Facebook + GitHub across budgets 0.1–0.9
2. NeuralSparse/PTDNet full baselines; edge-betweenness where feasible
3. Full A1–A32 ablations + sensitivity
4. Robustness, LP track leakage-safe runs, Track C hybrids
5. Multi-seed stats with Holm correction / effect sizes
6. Draft abstract only after (1)–(5)

## Commands to continue
```bash
make baselines
python scripts/run_ablation.py --datasets lastfm facebook --seed 0
python scripts/train_cailp.py --dataset github --config configs/experiments/pilot.yaml --seed 0
python scripts/evaluate_all.py && python scripts/make_tables.py && python scripts/make_figures.py
```
