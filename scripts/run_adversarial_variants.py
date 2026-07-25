#!/usr/bin/env python3
"""Optional adversarial variants (Module B)."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.models.adversarial_mask_generator import AdversarialMaskGenerator
from src.models.cailp import CAILPConfig, CAILPSocial
from src.models.representation_discriminator import (
    RepresentationDiscriminator,
    adversarial_bce_loss,
    gradient_penalty,
)
from src.sparsification.constrained_pruning import budget_prune_unconstrained
from src.tasks.node_classification import train_node_classifier
from src.utils.graph import subgraph_from_undirected_edges
from src.utils.io import get_device, save_json, set_seed, setup_logging


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", default=["lastfm"])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--removal-rate", type=float, default=0.5)
    parser.add_argument("--epochs", type=int, default=30)
    args = parser.parse_args()
    setup_logging()
    set_seed(args.seed)
    device = get_device()
    results = []

    for ds in args.datasets:
        data = torch.load(ROOT / "data" / "processed" / f"{ds}.pt", weights_only=False)
        split = torch.load(ROOT / "data" / "splits" / f"{ds}_seed{args.seed}.pt", weights_only=False)
        data.train_mask, data.val_mask, data.test_mask = split["train_mask"], split["val_mask"], split["test_mask"]
        data = data.to(device)
        n_cls = int(data.y.max().item()) + 1
        model = CAILPSocial(data.x.size(1), n_cls, CAILPConfig()).to(device)
        _ = model(data)
        disc = RepresentationDiscriminator(64).to(device)
        gen = AdversarialMaskGenerator(64).to(device)
        opt_m = torch.optim.Adam(model.parameters(), lr=1e-3)
        opt_d = torch.optim.Adam(disc.parameters(), lr=1e-3)
        opt_g = torch.optim.Adam(gen.parameters(), lr=1e-3)

        unstable = False
        for epoch in range(args.epochs):
            model.train()
            out = model(data)
            # Representation adversarial
            with torch.no_grad():
                h_orig = out["h"].detach()
            # Fake: dropout-masked embeddings as sparse proxy
            h_fake = F.dropout(h_orig, p=0.3, training=True)
            d_real = disc(h_orig)
            d_fake = disc(h_fake)
            loss_d = adversarial_bce_loss(d_real, d_fake) + 10 * gradient_penalty(disc, h_orig, h_fake)
            opt_d.zero_grad()
            loss_d.backward()
            opt_d.step()

            out = model(data)
            loss_task = F.cross_entropy(out["logits"][data.train_mask], data.y[data.train_mask])
            d_fake2 = disc(out["h"])
            loss_adv = F.binary_cross_entropy_with_logits(d_fake2, torch.ones_like(d_fake2))
            # Mask generator on fused edge scores via mu as z proxy
            masks = gen(out["mu"].unsqueeze(-1).repeat(1, 64), tau=1.0, hard=False)
            loss_sparse = masks.mean()
            loss_g = loss_task + 0.1 * loss_adv + 0.1 * loss_sparse
            opt_m.zero_grad()
            opt_g.zero_grad()
            loss_g.backward()
            opt_m.step()
            opt_g.step()
            if torch.isnan(loss_g):
                unstable = True
                break
            if epoch % 10 == 0:
                print(ds, epoch, float(loss_g), float(loss_d))

        model.eval()
        with torch.no_grad():
            out = model(data)
            mu = out["mu"].cpu()
            und = out["undirected_edges"].cpu()
        keep = budget_prune_unconstrained(und, mu, args.removal_rate)
        data_cpu = data.clone().cpu()
        sparse = subgraph_from_undirected_edges(data_cpu, keep)
        sparse.train_mask, sparse.val_mask, sparse.test_mask = (
            data_cpu.train_mask,
            data_cpu.val_mask,
            data_cpu.test_mask,
        )
        _, metrics = train_node_classifier(sparse, kind="gcn", epochs=60, device=device)
        results.append(
            {
                "dataset": ds,
                "method": "cailp_adversarial",
                "removal_rate": args.removal_rate,
                "unstable": unstable,
                **{k: metrics[k] for k in metrics if k.startswith("test_")},
            }
        )

    out = ROOT / "results" / "raw" / f"adversarial_seed{args.seed}.json"
    save_json(results, out)
    print("Wrote", out)


if __name__ == "__main__":
    main()
