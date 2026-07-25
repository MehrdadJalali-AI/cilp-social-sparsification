# Method naming audit (CILP terminology pass)

## Official names (reader-facing)

| Role | Name |
|------|------|
| Method | **CILP** (Counterfactual Inverse Link Prediction) — expand once |
| Main teacher | multi-criteria counterfactual teacher |
| Matched ablation | task-only counterfactual teacher (short: **Task-only**) |
| Parent method | **Original ILP-GCN** |
| Structural proxy | **Resistance proxy** |

## Removed from reader-facing PDFs

CAILP-Social, CAILP, Counterfactual-Aware, Attention-Based Inverse Link Prediction, A31, RQ11, multi-obj., fidelity-only, concat multi, DSpar/ER, resistance-style proxy, CAILP multi, Project Team, authoritative store

## Allowed locations for repository IDs

- `paper/tables/internal_id_mapping.csv` / `.tex` (SI S29 only)
- code, raw JSON keys, scripts, logs, docs

## Validation

```bash
python scripts/validate_manuscript_terminology.py
```

Scans `paper/*.tex`, `paper/sections/*.tex`, `paper/figures/*.tikz`, `paper/tables/*.tex` (excluding the mapping table).
Does **not** scan `_archive_pre_natcomm/` (historical drafts).
