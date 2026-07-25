# Hyperparameter audit

Source of truth for the reported multi-seed comparison: `scripts/run_full_grid.py` (`GRID_CFG` + `score_cailp`).

Complete machine-readable table: `paper/tables/hyperparameters_complete.csv`.

## Highlights

- Encoder: GCN, 2 layers, hidden 64, dropout 0.4, fusion=concat
- Optimizer: Adam lr=1e-3; weight decay not set (library default 0.0)
- Loss weights: λCE=0.5, λNLL=1.0, λrank=0.2; margin=0.1
- Teacher n: LastFM 50; Facebook 120 (selected by validation Macro-F1 @50%); GitHub 40
- β fixed: (1.0, 0.5, 1.0, 0.3, 0.5, 0.5)
- Pruning: unconstrained exact budget; constraints off
- Early stopping / scheduler / grad clipping: not configured (fixed epochs)

Pilot YAML configs differ and were **not** used for the multi-seed tables.
