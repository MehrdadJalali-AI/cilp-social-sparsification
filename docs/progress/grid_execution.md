# Multi-seed Grid Execution Log

**Started:** 2026-07-24

## Running
- LastFM seeds 0–4 and 5–9 (parallel): all core methods, budgets 0.1–0.9
- Facebook diagnostic: teacher_n ∈ {20,40,80,120} (val-ranked, test reported for all)
- Adversarial: **excluded** from core (pilot F1 0.666 vs 0.829)

## Protocol locks
- Fusion for RQ11 (multi vs A31): **concat** (held fixed)
- Equal edge budgets; identical splits/seeds
- No abstract / novelty / contributions until analysis complete

## Next
- Facebook + GitHub full grids after LastFM progresses
- Ablations A1–A32 on LastFM + Facebook
- `scripts/analyze_grid.py` for stats, AUC, Pareto, RQ11 decision
