from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class RepresentationDiscriminator(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, h: Tensor) -> Tensor:
        # Pool to graph-level if node matrix given
        if h.dim() == 2:
            h = h.mean(dim=0, keepdim=True)
        return self.net(h)


def adversarial_bce_loss(d_real: Tensor, d_fake: Tensor) -> Tensor:
    real = F.binary_cross_entropy_with_logits(d_real, torch.ones_like(d_real))
    fake = F.binary_cross_entropy_with_logits(d_fake, torch.zeros_like(d_fake))
    return real + fake


def gradient_penalty(discriminator: nn.Module, real: Tensor, fake: Tensor) -> Tensor:
    """WGAN-GP style penalty on interpolated node/graph embeddings."""
    if real.dim() == 1:
        real = real.unsqueeze(0)
        fake = fake.unsqueeze(0)
    alpha = torch.rand(real.size(0), 1, device=real.device)
    if real.dim() == 3:
        alpha = alpha.unsqueeze(-1)
    # For [N, D] node matrices, interpolate in embedding space after mean pool
    if real.size(0) != fake.size(0):
        real_p = real.mean(0, keepdim=True)
        fake_p = fake.mean(0, keepdim=True)
        alpha = torch.rand(1, 1, device=real.device)
        interp = (alpha * real_p + (1 - alpha) * fake_p).requires_grad_(True)
    else:
        interp = (alpha * real + (1 - alpha) * fake).requires_grad_(True)
    d_interp = discriminator(interp)
    grads = torch.autograd.grad(
        outputs=d_interp,
        inputs=interp,
        grad_outputs=torch.ones_like(d_interp),
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    grads = grads.view(grads.size(0), -1)
    return ((grads.norm(2, dim=1) - 1) ** 2).mean()
