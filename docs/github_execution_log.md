# GitHub execution log (Stage A)

## Timeline

| Time (local) | Event |
|---|---|
| 2026-07-25 ~07:28 | Baseline diagnostic seed0 @0.5 started (random, dspar, ER, ILP, NeuralSparse, PTDNet) |
| ~07:30 | Baselines completed (~123 s wall-clock for six methods @ one budget) |
| ~07:31 | CILP Task-only + multi diagnostic launched (durable background) |
| ~07:40 | Task-only seed0@0.5 complete (train_seconds=518.0, F1=0.8017) |
| ~08:00 | CILP multi seed0@0.5 complete (train_seconds=590.9, F1=0.8142) |
| ~08:20 | Diagnostic JSON archived to `results/raw/grid/diagnostic_stageA/`; authoritative CSV/JSON backed up |
| ~08:20 | Full GitHub grid launched: seeds 0–9 × all CORE_METHODS × budgets 0.1–0.9 |
| ~14:32 | Full grid completed (exit 0); elapsed ≈ 6.19 h; 80/80 method files |
| ~14:35 | Authoritative merge → 1890 rows; LF/FB SHA unchanged; validator OK |
| ~14:40 | Tables/figures regenerated with GitHub panels; decision: Supported |

## Device

- `get_device()` → **cpu**
- CUDA unavailable; MPS available but unused by protocol

## Notes

- Incomplete diagnostic JSON (1 budget row) moved to `results/raw/grid/diagnostic_stageA/` before full-grid launch so resume logic re-runs seed 0 for all nine budgets.
- Protocol: no teacher_n/epoch/budget reductions.
- Estimated full-grid wall-clock: ~4–7 hours (see `docs/github_runtime_estimate.md`); actual ≈ 6.19 h.
- No method failures; no error JSON files.
