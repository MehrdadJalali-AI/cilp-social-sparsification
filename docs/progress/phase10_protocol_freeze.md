# Experimental Protocol Freeze

**Frozen:** 2026-07-24 (after pilot success + unit tests + leakage audit)

## Locked
1. Datasets: Facebook Page-Page, LastFM Asia, GitHub Social (SNAP/MUSAE sources; multi-hot features)
2. Splits: stratified 60/20/20, seeds `{0…9}`, files in `data/splits/`
3. Track A equal edge-removal budgets: `{0.1…0.9}` (pilot used `{0.3,0.5,0.7}`)
4. Primary metric: Macro-F1 (node classification, GCN downstream default)
5. Leakage: `LeakageGuard`; test labels only after sparsifier frozen
6. Teacher: six-component aggregate (core) vs fidelity-only (A31)
7. Device: CUDA if available else CPU (not MPS)

## Not yet claimed
- Abstract / novelty paragraph / final contributions (await full Phase 11–15)
- Optional Module A/B inclusion (decision rules apply)

## Pilot gate
- [x] Unit tests pass (14)
- [x] Leakage audit document exists
- [x] LastFM pilot completes with machine-readable JSON
