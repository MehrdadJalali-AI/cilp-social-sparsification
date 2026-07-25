from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor


class MCDropoutUncertainty:
    """Optional MC-dropout uncertainty over an importance decoder."""

    def __init__(self, model: nn.Module, n_samples: int = 20):
        self.model = model
        self.n_samples = n_samples

    @torch.no_grad()
    def predict(self, z: Tensor) -> tuple[Tensor, Tensor]:
        self.model.train()  # keep dropout on
        mus = []
        for _ in range(self.n_samples):
            mu, _ = self.model(z)
            mus.append(mu)
        stack = torch.stack(mus, dim=0)
        mean = stack.mean(0)
        var = stack.var(0, unbiased=False) + 1e-6
        return mean, torch.log(var)
