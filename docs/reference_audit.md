# Reference audit

**Date:** 2026-07-24  
**Rule:** Do not invent bibliographic metadata. Remove or demote unverified entries.

## Verified (kept / corrected for main text)

| Key | Action | Notes |
|---|---|---|
| bangian2025ilp | Keep | DOI 10.1186/s40537-025-01220-8 |
| jalali2025blackhole | Keep | JCIM 2025 DOI 10.1021/acs.jcim.5c01518 |
| lucic2022cfgnn | Keep | AISTATS 2022 |
| pradoromero2022gretel | Keep | CIKM 2022 DOI 10.1145/3511808.3557608 |
| funke2022zorro | **Correct** | TKDE 2023; authors Funke, Khosla, Rathee, Anand; DOI 10.1109/TKDE.2022.3201170 |
| mastropietro2022edgeshaper | Keep | J Cheminform 2022 |
| zheng2020neuralsparse | Keep | ICML 2020 |
| luo2021ptdnet | Keep / prefer TNNLS if confirmed; else arXiv 2011.07057 |
| rong2020dropedge | Keep | ICLR 2020 |
| spielman2011graph | Keep | SIAM J Comput 2011 |
| liben2007link | Keep | JASIST 2007 |
| martinez2016survey | Keep | ACM Comput Surv 2016 |
| rozemberczki2021multi | Keep | J Complex Networks / MUSAE |
| kipf2017semi | Keep | ICLR 2017 |
| agarwal2023evaluating | **Correct** | Scientific Data 10:144 (2023); Agarwal, Queen, Lakkaraju, Zitnik; DOI 10.1038/s41597-023-01974-x |
| holm1979simple | Keep | Scand J Stat 1979 |
| wilcoxon1945individual | Keep | Biometrics Bull 1945 |
| cohen1988statistical | Keep | Book; Lawrence Erlbaum |

## Corrected

- `funke2022zorro`: add Rathee; venue IEEE TKDE 2023; DOI.
- `agarwal2023evaluating`: full authors; Scientific Data; volume/article.

## Removed from citations / bibliography (unverified or placeholder)

| Key | Reason |
|---|---|
| rlp2021reverse | Anonymous / venue TBD — **removed from text** |
| ilpesg2023 | Anonymous / venue TBD |
| baydeepmil2023 | Anonymous / venue TBD |
| li2024fsgnn | Incomplete / placeholder |
| chen2021sgcn | Mis-keyed / not used; SGC is Wu et al. if needed later |
| hoge2023uncertainty | Placeholder survey |
| effrosynidis2022social | Survey/venue placeholder |
| ahmed2020network | Unverified composite citation — **removed from text** |
| casiraghi2024semantic | Unverified |
| primavera2024finding | Unverified placeholder |
| huang2024c2explainer | Insufficient verification for current draft — **removed from text** |
| liu2023dspar | Not used as independent baseline name; resistance cited via Spielman |

## Unresolved (not cited in revised main text)

Optional SI-only literature may be re-added only after DOI/venue verification:
GCFExplainer, MEG, Wellawatte, Bajaj, Yuan survey, Faber, etc.

## Text substitutions after removals

- Social LP/reduction: cite Liben-Nowell & Kleinberg; Martínez et al. survey only.
- Resistance sparsification: cite Spielman & Srivastava only; method label “resistance-style proxy”.
- Explanation evaluation fragility: Agarwal et al. Scientific Data 2023.
