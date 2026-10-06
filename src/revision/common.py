"""Shared constants, provenance, timers, and memory sampling for the R1 revision.

Protocol tags
-------------
ORIGINAL : submitted protocol (teacher Δ_task on train∪validation; Δ_group on validation;
           Facebook teacher size chosen on validation; unseeded Louvain in the community proxy).
REVISED  : teacher targets from training labels only; validation used only for model
           and teacher-size selection; test untouched until configurations are locked.
"""
from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import threading
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Dict, Iterator, List, Optional

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
REVISED_DIR = ROOT / "results" / "revised"
CACHE_DIR = ROOT / "results" / "cache" / "teacher_deltas"
LEDGER = REVISED_DIR / "ledger.jsonl"

BUDGETS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
DATASETS = ["lastfm", "facebook", "github"]
TRANSFER_DATASETS = ["pubmed"]
SEEDS = list(range(10))
N_GRID = [20, 40, 80, 120, 240]
HOLDOUT_PROBES = 200

# Epoch budgets are unchanged from the ORIGINAL protocol (scripts/run_full_grid.py::GRID_CFG).
EPOCHS = {
    "lastfm": {"teacher_epochs": 25, "train_epochs": 25, "down_epochs": 50, "ilp_epochs": 25},
    "facebook": {"teacher_epochs": 25, "train_epochs": 25, "down_epochs": 40, "ilp_epochs": 20},
    "github": {"teacher_epochs": 15, "train_epochs": 15, "down_epochs": 35, "ilp_epochs": 15},
    # Non-social transfer graph added in R1; epoch budget fixed a priori to Facebook's (closest
    # node count among the original datasets), not tuned.
    "pubmed": {"teacher_epochs": 25, "train_epochs": 25, "down_epochs": 40, "ilp_epochs": 20},
}
ORIGINAL_TEACHER_N = {"lastfm": 50, "facebook": 120, "github": 40}

COMPONENTS = ["task", "comm", "conn", "spec", "repr", "group"]
BETA_DEFAULT = {"task": 1.0, "comm": 0.5, "conn": 1.0, "spec": 0.3, "repr": 0.5, "group": 0.5}


def _vec(**kw: float) -> Dict[str, float]:
    return {k: float(kw.get(k, 0.0)) for k in COMPONENTS}


def teacher_variants(n_dirichlet: int = 20, dirichlet_seed: int = 20261004) -> Dict[str, Dict[str, float]]:
    """Pre-declared teacher variants (Step 2). Weights are NOT renormalised after zeroing:
    the final min–max normalisation of the composite makes the overall scale irrelevant."""
    v: Dict[str, Dict[str, float]] = {}
    v["cilp_full"] = dict(BETA_DEFAULT)
    v["task_only"] = _vec(task=1.0)
    for k in COMPONENTS:  # leave-one-component-out
        b = dict(BETA_DEFAULT)
        b[k] = 0.0
        v[f"loo_{k}"] = b
    v["grp_deletion_only"] = {k: (BETA_DEFAULT[k] if k in ("task", "conn", "repr", "group") else 0.0) for k in COMPONENTS}
    v["grp_task_plus_proxies"] = {k: (BETA_DEFAULT[k] if k in ("task", "comm", "spec") else 0.0) for k in COMPONENTS}
    v["grp_proxies_only"] = {k: (BETA_DEFAULT[k] if k in ("comm", "spec") else 0.0) for k in COMPONENTS}
    v["w_balanced"] = _vec(task=1, comm=1, conn=1, spec=1, repr=1, group=1)
    v["w_task_focused"] = _vec(task=1.0, comm=0.2, conn=0.2, spec=0.2, repr=0.2, group=0.2)
    v["w_structure_focused"] = _vec(task=0.2, comm=1.0, conn=1.0, spec=1.0, repr=0.2, group=0.2)
    v["w_reduced_proxy"] = {**BETA_DEFAULT, "comm": 0.1, "spec": 0.1}
    rng = np.random.default_rng(dirichlet_seed)
    for i in range(n_dirichlet):
        w = rng.dirichlet(np.ones(len(COMPONENTS)))
        v[f"dir_{i:02d}"] = {k: float(x) for k, x in zip(COMPONENTS, w)}
    return v


# ---------------------------------------------------------------- provenance
def git_commit() -> str:
    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=True)
        dirty = subprocess.run(["git", "status", "--porcelain", "src", "scripts"], cwd=ROOT,
                               capture_output=True, text=True).stdout.strip()
        return out.stdout.strip() + ("-dirty" if dirty else "")
    except Exception:
        return "unknown"


def hardware() -> Dict[str, object]:
    import torch

    def sysctl(key: str) -> str:
        try:
            return subprocess.run(["sysctl", "-n", key], capture_output=True, text=True).stdout.strip()
        except Exception:
            return ""

    return {
        "cpu": sysctl("machdep.cpu.brand_string") or platform.processor(),
        "logical_cores": os.cpu_count(),
        "ram_gb": round(int(sysctl("hw.memsize") or 0) / 2**30, 1),
        "os": f"{platform.system()} {platform.mac_ver()[0] or platform.release()}",
        "python": platform.python_version(),
        "torch": torch.__version__,
        "torch_threads": torch.get_num_threads(),
        "device": "cpu",
    }


def sha_of(obj: object) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, default=str).encode()).hexdigest()[:16]


def append_ledger(entry: Dict[str, object]) -> None:
    REVISED_DIR.mkdir(parents=True, exist_ok=True)
    entry = {"logged_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"), **entry}
    with LEDGER.open("a") as f:
        f.write(json.dumps(entry, default=str) + "\n")


# ---------------------------------------------------------------- timing / memory
class PeakRSS:
    """Samples process RSS in a background thread.

    Peak memory is reported as the maximum resident set size observed during the block
    (``peak_rss_mb``) and as the increment over RSS at block entry (``delta_rss_mb``).
    CPU-only runs: no device-allocator statistics are available.
    """

    def __init__(self, interval: float = 0.005):
        import psutil

        self.proc = psutil.Process()
        self.interval = interval
        self.peak = 0
        self.start = 0
        self._stop = threading.Event()

    def _run(self) -> None:
        while not self._stop.is_set():
            rss = self.proc.memory_info().rss
            if rss > self.peak:
                self.peak = rss
            time.sleep(self.interval)

    def __enter__(self) -> "PeakRSS":
        self.start = self.proc.memory_info().rss
        self.peak = self.start
        self._t = threading.Thread(target=self._run, daemon=True)
        self._t.start()
        return self

    def __exit__(self, *exc) -> None:
        self._stop.set()
        self._t.join()
        rss = self.proc.memory_info().rss
        self.peak = max(self.peak, rss)

    @property
    def peak_mb(self) -> float:
        return self.peak / 2**20

    @property
    def delta_mb(self) -> float:
        return (self.peak - self.start) / 2**20


@contextmanager
def timed(store: Dict[str, float], key: str, memory: bool = False) -> Iterator[None]:
    if memory:
        with PeakRSS() as m:
            t0 = time.perf_counter()
            yield
            store[key] = time.perf_counter() - t0
        store[f"{key}_peak_rss_mb"] = m.peak_mb
        store[f"{key}_delta_rss_mb"] = m.delta_mb
    else:
        t0 = time.perf_counter()
        yield
        store[key] = time.perf_counter() - t0


def gcn_messages(num_undirected_edges: int, num_nodes: int, self_loops: bool = True) -> int:
    """Directed messages per GCN layer per forward pass: both directions of every retained
    undirected edge plus one self-loop per node (GCNConv adds self-loops by default)."""
    return 2 * int(num_undirected_edges) + (int(num_nodes) if self_loops else 0)


def set_seed_all(seed: int) -> None:
    import random

    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def blind_test(data):
    """Selection-phase copy of ``data``: the test mask is emptied so that no test prediction
    or test metric is ever computed before configurations are locked."""
    import torch

    d = data.clone()
    d.test_mask = torch.zeros_like(data.test_mask)
    return d
