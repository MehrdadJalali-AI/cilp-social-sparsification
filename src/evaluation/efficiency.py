from __future__ import annotations

import time
from contextlib import contextmanager
from typing import Dict, Iterator

import torch


@contextmanager
def timer() -> Iterator[Dict[str, float]]:
    out: Dict[str, float] = {}
    t0 = time.perf_counter()
    yield out
    out["seconds"] = time.perf_counter() - t0


def peak_memory_mb(device: torch.device) -> float:
    if device.type == "cuda" and torch.cuda.is_available():
        return torch.cuda.max_memory_allocated(device) / (1024**2)
    return float("nan")


def count_parameters(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
