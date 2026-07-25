# Phase 2–3 Progress

**Date:** 2026-07-24

## Phase 2
- Mathematical formulation: `docs/mathematical_formulation.md`
- Initial research brief: `docs/initial_research_brief.md`
- Full repository skeleton, configs, scripts, package layout

## Phase 3
- Datasets downloaded via SNAP/MUSAE mirrors (PyG `graphmining.ai` URLs return 404)
- Features reconstructed as multi-hot from MUSAE index lists
- Audit written: `docs/dataset_audit.md`
- Splits: 10 seeds × 3 datasets in `data/splits/`

| Dataset | Nodes | Undirected edges | Classes | Homophily | Feat dim |
|---|---|---|---|---|---|
| Facebook | 22470 | 170823 | 4 | 0.885 | 4714 |
| LastFM | 7624 | 27806 | 18 | 0.874 | 7842 |
| GitHub | 37700 | 289003 | 2 | 0.845 | 4005 |
