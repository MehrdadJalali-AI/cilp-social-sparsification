from __future__ import annotations

from typing import Dict, Iterable, List, Sequence

import numpy as np
from scipy.stats import friedmanchisquare, ttest_rel, wilcoxon
from statsmodels.stats.multitest import multipletests


def summarize(values: Sequence[float]) -> Dict[str, float]:
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return {"mean": float("nan"), "std": float("nan"), "ci95_low": float("nan"), "ci95_high": float("nan")}
    mean = float(arr.mean())
    std = float(arr.std(ddof=1)) if arr.size > 1 else 0.0
    se = std / max(np.sqrt(arr.size), 1.0)
    return {
        "mean": mean,
        "std": std,
        "ci95_low": mean - 1.96 * se,
        "ci95_high": mean + 1.96 * se,
        "n": float(arr.size),
    }


def paired_compare(a: Sequence[float], b: Sequence[float]) -> Dict[str, float]:
    a_arr, b_arr = np.asarray(a, float), np.asarray(b, float)
    out: Dict[str, float] = {}
    if len(a_arr) < 2:
        return {"paired_t_p": float("nan"), "wilcoxon_p": float("nan"), "effect_size_cohens_d": float("nan")}
    t_stat, t_p = ttest_rel(a_arr, b_arr)
    out["paired_t_p"] = float(t_p)
    try:
        w_stat, w_p = wilcoxon(a_arr, b_arr)
        out["wilcoxon_p"] = float(w_p)
    except Exception:
        out["wilcoxon_p"] = float("nan")
    diff = a_arr - b_arr
    out["effect_size_cohens_d"] = float(diff.mean() / (diff.std(ddof=1) + 1e-8))
    return out


def holm_correct(pvalues: Sequence[float]) -> List[float]:
    reject, p_adj, _, _ = multipletests(pvalues, method="holm")
    return [float(p) for p in p_adj]


def friedman_test(method_arrays: List[Sequence[float]]) -> Dict[str, float]:
    if len(method_arrays) < 3:
        return {"friedman_p": float("nan")}
    stat, p = friedmanchisquare(*[np.asarray(a) for a in method_arrays])
    return {"friedman_stat": float(stat), "friedman_p": float(p)}
