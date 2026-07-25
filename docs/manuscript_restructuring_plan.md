# Manuscript Restructuring Plan — CAILP-Social → Nature Communications style

**Reference model (architecture only):** Wang et al., *Nat Commun* (2026) — DGAT (`s41467-026-73114-z.pdf`), 18 pages.  
**Current manuscript:** `paper/main.pdf` (~32 pages), project-report style.  
**Date:** 2026-07-24  

**Constraint:** Use DGAT only for architecture, flow, figure integration, caption style, and visual hierarchy. Do not copy wording, science, or graphics.

---

## 1. Current problem

| Issue | Evidence in current MS |
|---|---|
| Project-report tone | TOC; “working draft”; “evidence freeze”; “queued”; “automated decision False”; reporting checklists |
| Wrong information order | Method + Experimental setup + Related Work *before* Results; Nature Comm puts **Results first**, Methods last |
| Excessive subsectioning | 10 section files; Related Work alone has 8 subsections; Appendix has 25+ micro-sections |
| Repetition | CF-GNNExplainer contrast repeated (Intro/RW/Method/Appendix); LastFM numbers in Intro preview + Results + Discussion + Appendix; unsafe-claims lists duplicated |
| Figures not NatComm-like | Separate LastFM/Facebook PNGs; no framework schematic; no unified multi-panel story; appendix dumps |
| Abstract pollution | Long “working summary” in PDF instead of deferred placeholder |
| RQ list as standalone | Numbered RQ1–RQ11 block in Introduction |
| Main text too long | Algorithms, A1–A32 catalogue, glossary, hyperparams, failure logs belong in SI |

DGAT pattern to emulate:
1. Short abstract → narrative Introduction (gap → prior ML → “Here, we present…”)  
2. **Results** opens with **framework overview + Fig. 1**  
3. Successive Results blocks each introduce a figure/table, then numbers, then one interpretive sentence  
4. Compact Discussion (supported / not supported / limitations)  
5. Detailed Methods after Discussion  
6. Data/Code/Authors/Competing → References → SI pointer  

---

## 2. Section-by-section comparison

| DGAT | Current CAILP | Action |
|---|---|---|
| Title + authors + abstract | Title + deferred abstract + long working summary | Keep deferred abstract; **remove working summary from PDF** |
| Introduction (motivation → gap → prior methods woven in → “Here we present”) | Intro + huge Related Work + Problem + Method before Results | **Merge** RW into short Intro “Related work and positioning”; move Problem/Method detail to Methods / Results overview |
| Results: Overview + Fig.1 | Method section earlier; no schematic Fig.1 | New **Results 4.1** + TikZ Fig.1 (train vs sparsify panels) |
| Results: benchmarking panels | Split LastFM/Facebook; project status language | Unified **Results 4.2–4.4** with Figs 2–3; neutral “incomplete seed coverage” only where needed |
| Results: ablation / interpretability | Ablations + RQ11 decision status prose | **Results 4.5–4.8**; move decision-rule internals to Methods/SI |
| Discussion | Discussion + Limitations section | Merge limitations into Discussion; ethics stay brief or SI |
| Methods | Methods + Problem + Experimental setup overlap | Single Methods with target subsections |
| Data/Code/Authors/Competing | Mixed into Limitations/Appendix | Standard end matter |
| SI | Appendix inside main PDF | Separate `supplementary.tex` |

---

## 3. Proposed section mapping

### Main article (`paper/main.tex`)

1. Title  
2. Abstract — LaTeX comment / one-line deferred placeholder only  
3. Introduction  
   - Motivation (social graph scale; sparsification need)  
   - Gap (explainers ≠ budgeted sparsifiers; joint evaluation fragmented)  
   - Relation to ILP-GCN (prior team work)  
   - Compact “Related work and positioning” (table optional → SI if large)  
   - One CF-GNNExplainer contrast paragraph (once)  
   - Concise framework statement + evidence-dependent objective  
   - **No** numbered contribution list; **no** RQ laundry list  
4. Results  
   - 4.1 Overview of CAILP-Social (Fig. 1)  
   - 4.2 Dataset and protocol overview (Table 1)  
   - 4.3 Main multi-seed sparsification results (Fig. 2, Table 2)  
   - 4.4 Comparison vs ILP / learned sparsifiers (Fig. 3)  
   - 4.5 RQ11 multi vs A31, fusion fixed (Fig. 4, Table 3)  
   - 4.6 Structural / community / group (Fig. 5, Table 4)  
   - 4.7 Surrogate quality and uncertainty (compact; detail → SI)  
   - 4.8 Ablations (Fig. 6)  
   - 4.9 Runtime / memory (Fig. 7, Table 5)  
   - 4.10 Rejected optional hypotheses (adversarial; teacher-size failure)  
5. Discussion  
6. Methods (as specified by user)  
7–11. Data / Code / Authors / Competing / References  
12. Supplementary Information pointer  

### Supplementary (`paper/supplementary.tex`)

- Extended equations and algorithms  
- Full budget×method tables  
- Complete A1–A32 catalogue and tables  
- Hyperparameters  
- Closest-methods map (full table)  
- Unsafe-claims checklist (internal scientific positioning)  
- RQ11 decision-rule pseudocode  
- Glossary, threats to validity, extended stats (Holm tables)  
- Protocol notes (no “queued/freeze” in main text)

### Moved out of manuscript PDF

| Content | Destination |
|---|---|
| Evidence freeze / queue / automated False | `docs/progress/` |
| Reporting checklist / versioning | `docs/progress/` |
| TOC | delete |
| Duplicate CF contrast / unsafe claims | keep once in Intro; full list in SI |
| Preview of interim findings in Intro | delete (belongs in Results only) |

---

## 4. Retain / merge / move / delete

### Retain (main text)
- Honest competitiveness vs PTDNet and original ILP  
- Multi > A31 on LastFM (with stats); Facebook directional / incomplete  
- Adversarial rejection with numbers  
- Facebook teacher \(n{=}40\) negative + \(n{=}120\) validation selection  
- Leakage controls; equal budgets; fusion-fixed RQ11  
- ILP as extension of Bangian Tabrizi et al. 2025  
- No first-ever / SOTA claims  

### Merge
- Related Work § → Intro “Related work and positioning”  
- Problem definition § → Methods “Problem formulation” + Results overview  
- Experimental setup § → Results 4.2 (brief) + Methods  
- Limitations § → Discussion  

### Move to SI
- A1–A32 full catalogue and full ablation numeric table  
- Extended algorithms, loss expansions, modularity formulae  
- Full closest-methods table; glossary; hyperparams  
- Per-budget raw means for all methods  
- Paired-test full matrices  

### Delete from article PDF
- `\tableofcontents`  
- Working-summary abstract block  
- Manuscript organization subsection  
- Standalone RQ1–RQ11 list  
- “Outlook: rewrite abstract after…” project language  
- Closing protocol reminder in appendix  

---

## 5. Figure and table map

| ID | Content | Source / generation | Placement |
|---|---|---|---|
| **Fig. 1** | Framework schematic (a train, b sparsify) | New TikZ → `figures/fig1_framework.pdf` | After Results 4.1 opening |
| **Fig. 2** | Multi-dataset sparsity–Macro-F1 ± CI | Python from grid JSON → `fig2_sparsity_curves.pdf` | Results 4.3 |
| **Fig. 3** | Core method comparison (AUC / key budgets) | Python → `fig3_method_comparison.pdf` | Results 4.4 |
| **Fig. 4** | RQ11 multi vs A31 | Python → `fig4_rq11.pdf` | Results 4.5 |
| **Fig. 5** | Structural / group metrics | Python → `fig5_structure.pdf` | Results 4.6 |
| **Fig. 6** | Ablations (LastFM) | Python → `fig6_ablations.pdf` | Results 4.8 |
| **Fig. 7** | Runtime / memory | Python from JSON if present; else SI note | Results 4.9 |

| Table | Content |
|---|---|
| Table 1 | Dataset characteristics |
| Table 2 | Main predictive (selected budgets + AUC) |
| Table 3 | RQ11 compact |
| Table 4 | Structural/group at 50% |
| Table 5 | Runtime/memory summary |

Old standalone PNGs (`sparsity_f1_lastfm.png`, etc.) superseded by Fig. 2–5 scripts; keep files for provenance.

---

## 6. Evidence gaps (do not invent)

- GitHub multi-seed grid incomplete  
- Facebook not all seeds 0–9 complete  
- RQ11 headline retention not satisfied  
- Runtime/memory may be incomplete for all methods  
- DSpar ≡ ER proxy in current code  
- Final abstract / novelty / contributions still deferred  

Mark missing panels with LaTeX comments `% TODO: unavailable until ...`, not visible placeholder prose.

---

## 7. Implementation order

1. This plan file (done)  
2. Figure generation scripts + TikZ Fig. 1  
3. Refactor section files + `main.tex` + `supplementary.tex`  
4. Compile both PDFs  
5. `docs/manuscript_change_log.md`  
