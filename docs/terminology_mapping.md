# Terminology mapping

| Old term | New term | Reader-facing? | Internal compatibility note |
|---|---|---|---|
| CAILP-Social | CILP | Yes → replace | Repo paths unchanged |
| CAILP | CILP | Yes → replace | JSON keys `cailp_multi` remain |
| Counterfactual-Aware Inverse Link Prediction | Counterfactual Inverse Link Prediction | Yes → replace | Acronym CILP |
| Counterfactual Attention-Based Inverse Link Prediction | Counterfactual Inverse Link Prediction | Yes → replace | Attention not core |
| CAILP multi / CAILP-Social (multi-obj.) | CILP | Yes → replace | — |
| multi-objective / multi-obj. | multi-criteria | Yes → replace | Weighted-sum scalarization, not Pareto |
| A31 / Fidelity-only A31 | Task-only | Yes → replace | Repo key `cailp_a31` |
| fidelity-only | task-only | Yes → replace | — |
| RQ11 | (removed; descriptive prose) | Yes → remove | Rule text only in SI Methods |
| A1–A32 | scientific names; Repo ID column only | IDs only in mapping table | Ablation JSON filenames |
| resistance-style proxy / DSpar/ER | Resistance proxy | Yes → replace | Canonical `resistance_style_proxy` |
| original ILP / ILP baseline / Original ILP-GCN | ILP-GCN (after first intro: “the original ILP-GCN method”) | Yes → replace | — |
| Project Team | (omit if authors unset) | Yes → remove | Comment in LaTeX |
| authoritative store | (scientific wording) | Yes → rephrase | Keep in docs/scripts |
| knowledge-preserving (title) | utility-preserving | Yes | — |
