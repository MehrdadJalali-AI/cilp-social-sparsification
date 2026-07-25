# Stage A freeze manifest — GitHub Developers completed

**Status:** FROZEN / IMMUTABLE  
**Timestamp (UTC):** 20260725T124409Z  
**Public repository:** https://github.com/MehrdadJalali-AI/cilp-social-sparsification  
**Git commit (initial public release):** `f2c6bc9` on `main`  
**Suggested Git tag:** `stage-a-github-frozen` (created locally; push with the branch)  
**Checksums file:** `CHECKSUMS.sha256`  
**Local research archive (not in Git):** `results/frozen/stage_a_github_20260725T124409Z/`  

---

## Scope of Stage A

Stage A completed the GitHub Developers evaluation under the **identical frozen protocol** used for LastFM Asia and Facebook Page-Page. Stage B (adaptive teacher sampling) was **not** started.

| Item | Value |
|---|---|
| Datasets in authoritative store | LastFM, Facebook, GitHub |
| Authoritative rows (total) | **1,890** |
| LastFM rows | 630 |
| Facebook rows | 630 |
| GitHub rows | **630** (newly added) |
| Seeds | 0–9 (all datasets) |
| Methods (7) | CILP, Task-only, ILP-GCN, PTDNet, NeuralSparse, Random, Resistance proxy |
| Budgets | 0.1–0.9 |
| GitHub wall-clock (full grid orchestration) | ≈ 6.19 h (22,282,607 ms) |
| Method failures (Stage A GitHub) | **0** |
| LastFM / Facebook values | **Unchanged** (key-field equality vs pre-GitHub backup: True, 1,260 rows) |
| Decision rule | Unchanged (predefined four-metric Holm family) |
| GitHub decision | **Supported (4/4)** |

---

## Checksums (authoritative core files)

| Path | SHA-256 | Bytes |
|---|---|---:|
| `results/processed/authoritative_results.csv` | `a84cd74f21c7fb419480a1abc7fdce2709c2aa8ed891677459bf6fc9c138a9d5` | 603615 |
| `results/processed/authoritative_results.json` | `9be42ef09500798239f594bf977fc746907b2737aadf29708e57e46cba8106a3` | 1552079 |
| `results/processed/multi_dim_decision.json` | `f047c5bb21c11012ba9e60ca4ad50c90233f858d4e364bb518561478dd808d0c` | 795 |
| `results/processed/authoritative_results.csv.bak_pre_github_20260725` | `4be730200e49059652239928da60e7582c24ae68287e2055e2682d593e96b297` | 395894 |

Exact SHA-256 digests for all 154 archived files are recorded in `MANIFEST.json`. Do not alter frozen archive contents after checksum recording.

---

## Archived contents

1. **Authoritative result store** — CSV + JSON (1,890 rows) and pre-GitHub backup  
2. **GitHub raw per-seed outputs** — `results/raw/grid/github_seed{0-9}_*.json` (80 files)  
3. **Diagnostic Stage A outputs** — `results/raw/grid/diagnostic_stageA/`  
4. **Regenerated tables** — `paper/tables/*`  
5. **Regenerated figures** — `paper/figures/fig{2,3,4,5,7}_*`  
6. **Validator / statistical outputs** — decision JSON, paired structural CSV, audits  
7. **GitHub audit reports** — `docs/github_*.md`  
8. **Exact configuration sources** — `scripts/run_full_grid.py`, `src/counterfactual/exact_teacher.py`  
9. **GitHub splits** — `data/splits/github_seed{0-9}.pt`  
10. **Processed GitHub graph** — `data/processed/github.pt`

---

## Validation status

- Unique key `(dataset, method, seed, edge_removal_rate)` complete for all three datasets  
- No duplicate keys  
- Budgets 0.1–0.9 present for every method × seed  
- LastFM and Facebook key-field SHA unchanged vs pre-GitHub backup  
- Multi-dimensional decision: LastFM / Facebook / GitHub all **Supported**

---

## GitHub CILP − Task-only (frozen; n=10)

| Metric | Δ | Holm p | W/T/L |
|---|---:|---:|---|
| Sparsity–Macro-F1 AUC | +0.0033 | 0.0078 | 10/0/0 |
| Giant-component ratio @50% | +0.228 | 0.0078 | 10/0/0 |
| Bridge retention @50% | +0.320 | 0.0078 | 10/0/0 |
| Minority-degree retention @50% | +0.114 | 0.0078 | 9/0/1 |

**Decision:** Supported (4/4).

---

## Immutability rule

After this manifest is written, Stage A result files in the archive must not be modified. Manuscript integration may cite and display these values but must not recompute or retune Stage A experiments.
