# GitHub result merge audit (Stage A)

## Pre-merge backup

- `results/processed/authoritative_results.csv.bak_pre_github_20260725`
- `results/processed/authoritative_results.json.bak_pre_github_20260725`

## LastFM / Facebook integrity

| Check | Result |
|---|---|
| Pre-merge LF+FB row count | 1260 |
| Post-merge LF+FB row count | 1260 |
| SHA256 of (dataset,method,seed,rate,test_macro_f1) | **identical** (`92d89e3539aefc8b…`) |

## GitHub rows added

| Item | Count |
|---|---|
| Raw method×seed JSON files | 80 (10 seeds × 8 runner keys) |
| Authoritative GitHub rows after dedupe | **630** (= 10 × 7 methods × 9 budgets) |
| Resistance proxy dedupe | `dspar` preferred; `effective_resistance` identical and collapsed |

## Conflicts / duplicates / missing

- Duplicate unique keys: **none**
- Conflicting values: **none**
- Missing budgets: **none**
- Invalid seeds: **none**
- Incomplete method records in authoritative table: **none**

## Validator

```text
OK: 1890 authoritative rows, no duplicate keys, budgets complete
```

(`scripts/validate_result_consistency.py` exit 0)

## Totals

| Store | Rows |
|---|---|
| Before merge | 1260 |
| GitHub added | 630 |
| After merge | **1890** |
