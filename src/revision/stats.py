"""Paired statistics exactly as pre-declared in docs/revision/PREREGISTRATION.md."""
from __future__ import annotations

from typing import Dict, List, Sequence

import numpy as np
from scipy import stats

SESOI_AUC = 0.008
WILCOXON_FLOOR_N10 = 2 / 2**10  # smallest exact two-sided p for n = 10


def sparsity_auc(rates: Sequence[float], f1: Sequence[float]) -> float:
    """Trapezoidal area of Macro-F1 over removal rates 0.1–0.9 (ORIGINAL definition)."""
    o = np.argsort(rates)
    return float(np.trapz(np.asarray(f1)[o], np.asarray(rates)[o]))


def paired(a: Sequence[float], b: Sequence[float]) -> Dict[str, float]:
    """a − b over seed-aligned pairs."""
    d = np.asarray(a, float) - np.asarray(b, float)
    n = len(d)
    mean, sd = float(d.mean()), float(d.std(ddof=1)) if n > 1 else float("nan")
    half = stats.t.ppf(0.975, n - 1) * sd / np.sqrt(n) if n > 1 else float("nan")
    nz = d[d != 0]
    p = float(stats.wilcoxon(nz, zero_method="wilcox", correction=False, method="exact").pvalue) if len(nz) > 0 else 1.0
    return {"n": n, "mean_diff": mean, "ci_low": mean - half, "ci_high": mean + half,
            "d_z": mean / sd if sd and sd > 0 else float("nan"),
            "wins": int((d > 0).sum()), "ties": int((d == 0).sum()), "losses": int((d < 0).sum()), "p_raw": p}


def holm(pvals: List[float]) -> List[float]:
    m = len(pvals)
    order = np.argsort(pvals)
    adj = np.empty(m)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (m - rank) * pvals[i]))
        adj[i] = running
    return adj.tolist()
