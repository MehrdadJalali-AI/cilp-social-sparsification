#!/usr/bin/env python3
"""Build data/processed/pubmed.pt (R1 transfer dataset) and its stratified 60/20/20 splits for seeds 0-9.

Downloads the Planetoid PubMed files via PyTorch Geometric (Sen et al. 2008; Yang et al. 2016), applies the
same preprocessing as the social graphs (symmetrize, remove self-loops/duplicates), and writes our own
stratified splits (the public Planetoid split is not used). Verify against CHECKSUMS.sha256.
"""
import sys
from pathlib import Path

import torch
from torch_geometric.data import Data
from torch_geometric.datasets import Planetoid

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.data.datasets import preprocess_graph, save_processed  # noqa: E402
from src.utils.splits import stratified_node_splits  # noqa: E402

raw = Planetoid(str(ROOT / "data" / "raw" / "pubmed_planetoid"), "PubMed")[0]
d0 = Data(x=raw.x, edge_index=raw.edge_index, y=raw.y)
d0.name = "pubmed"
d0.citation = "Sen et al. 2008; Yang et al. 2016 (Planetoid PubMed)"
data, audit = preprocess_graph(d0)
print(save_processed(data, "pubmed", processed_dir=ROOT / "data" / "processed"), audit)
for seed in range(10):
    tr, va, te = stratified_node_splits(data.y, seed=seed)
    torch.save({"train_mask": tr, "val_mask": va, "test_mask": te, "seed": seed},
               ROOT / "data" / "splits" / f"pubmed_seed{seed}.pt")
