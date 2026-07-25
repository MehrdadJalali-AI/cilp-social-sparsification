# Result consistency audit

**Date:** 2026-07-25  
**Authoritative store:** `results/processed/authoritative_results.csv` (1260 rows)  
**Validator:** `scripts/validate_result_consistency.py` (exit 0)

## Findings

| Check | Result |
|---|---|
| Duplicate unique keys (dataset×method×seed×budget) | None in authoritative store |
| `dspar` vs `effective_resistance` | **Identical** Macro-F1 for all 90 LastFM + 90 Facebook cells |
| Alias handling | Canonical method `resistance_style_proxy`; prefer `dspar` source file; drop ER duplicate (180 rows removed) |
| Alias inflation | **Prevented** — n=10 seeds, not n=20/12 |
| LastFM seeds | 0–9 for all core methods |
| Facebook seeds | 0–9 for all core methods (including CAILP multi & A31) |
| GitHub | **0 rows** |
| Conflicting values for same key | None |
| Missing budgets 0.1–0.9 | None for core methods |
| Teacher-size diagnostic files | Excluded from authoritative store (`*teacher*.json`) |

## Facebook A31 / CAILP reconciliation

Previous manuscript drafts mixed CAILP n=4 with A31 n=6.  
**Current authoritative:** both methods have seeds **0–9 (n=10)**. Paired contrasts use the common set 0–9.

## Method name map

| Raw JSON key | Authoritative method |
|---|---|
| cailp_multi | cailp_social_multi |
| cailp_a31 | cailp_a31 |
| original_ilp | original_ilp_gcn |
| ptdnet | ptdnet |
| neuralsparse | neuralsparse |
| random | random |
| dspar / effective_resistance | resistance_style_proxy (deduped) |
