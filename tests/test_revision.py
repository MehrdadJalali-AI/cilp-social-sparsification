"""Equivalence checks for the R1 revision pipeline (run on the real LastFM graph, seed 0)."""
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
pytestmark = pytest.mark.skipif(not (ROOT / "data/processed/lastfm.pt").exists(), reason="processed LastFM graph absent")


def _setup():
    from scripts.run_full_grid import load_data, pretrain_encoder
    from src.utils.graph import undirected_edge_list
    from src.utils.io import set_seed

    set_seed(0)
    data = load_data("lastfm", 0)
    enc, clf = pretrain_encoder(data, torch.device("cpu"), 5)
    return data, enc, clf, undirected_edge_list(data.edge_index)


def test_deltas_match_original_teacher():
    import networkx as nx

    from src.counterfactual.exact_teacher import ExactCounterfactualTeacher, community_effect
    from src.revision.teacher import compute_deltas, graph_context

    data, enc, clf, und = _setup()
    ctx = graph_context(data, louvain_seed=0)
    bridge_ids = [i for i in range(und.size(1)) if frozenset((int(und[0, i]), int(und[1, i]))) in ctx.bridges][:3]
    ids = bridge_ids + [5, 77, 1234, 20000]
    raw, _ = compute_deltas(data, enc, clf, und, ids, ctx)
    _, eff = ExactCounterfactualTeacher().score_edges(data, torch.tensor(ids), und, enc, clf, torch.device("cpu"))
    np.testing.assert_allclose(raw["task_orig"], eff.task, rtol=0, atol=1e-6)
    np.testing.assert_allclose(raw["repr"], eff.repr, rtol=1e-5, atol=1e-7)
    np.testing.assert_allclose(raw["group_orig"], eff.group, atol=1e-12)
    np.testing.assert_allclose(raw["conn"], eff.conn, atol=1e-12)
    np.testing.assert_allclose(raw["spec"], eff.spec, atol=1e-12)
    expected_comm = [community_effect(ctx.G, int(und[0, i]), int(und[1, i]), partition=ctx.partition) for i in ids]
    np.testing.assert_allclose(raw["comm"], expected_comm, atol=1e-12)
    assert len(bridge_ids) == 3 and all(raw["conn"][:3] > 0)


def test_cached_views_match_original_scorer():
    from src.models.cailp import CAILPConfig, CAILPSocial
    from src.revision.scorer import CachedViewCAILP

    data, *_ = _setup()
    cfg = CAILPConfig(encoder_type="gcn", fusion="concat", hidden_dim=64, num_layers=2, dropout=0.4)
    outs = []
    for cls in (CAILPSocial, CachedViewCAILP):
        torch.manual_seed(1)
        model = cls(data.x.size(1), 18, cfg)
        _ = model(data)
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        torch.manual_seed(2)
        for _ in range(2):
            model.train()
            opt.zero_grad()
            out = model(data)
            (out["mu"].mean() + F.cross_entropy(out["logits"][data.train_mask], data.y[data.train_mask])).backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            outs.append(model(data)["mu"])
    torch.testing.assert_close(outs[0], outs[1], rtol=0, atol=1e-6)


def test_composite_reproduces_original_aggregation():
    from src.counterfactual.sampling import normalize_scores
    from src.revision.common import BETA_DEFAULT
    from src.revision.teacher import composite_target

    rng = np.random.default_rng(0)
    raw = {k: rng.random(30) for k in ["task_orig", "task_rev", "comm", "conn", "spec", "repr", "group_orig", "group_rev"]}
    raw["group_orig"][:] = 0.0
    b = BETA_DEFAULT
    expected = normalize_scores(sum(b[k] * normalize_scores(raw[{"task": "task_orig", "group": "group_orig"}.get(k, k)]) for k in b))
    np.testing.assert_allclose(composite_target(raw, b, "ORIGINAL"), expected, atol=1e-12)
    np.testing.assert_allclose(composite_target(raw, {"task": 1.0}, "ORIGINAL"), normalize_scores(raw["task_orig"]))


def test_sparse_q_matches_dense_on_facebook():
    from scripts.run_full_grid import load_data
    from src.models.cailp import CAILPConfig
    from src.revision.scorer import CachedViewCAILP

    if not (ROOT / "data/processed/facebook.pt").exists():
        pytest.skip("facebook graph absent")
    data = load_data("facebook", 0)
    cfg = CAILPConfig(encoder_type="gcn", fusion="concat", hidden_dim=64, num_layers=2, dropout=0.4)
    outs = []
    for sparse in (False, True):
        torch.manual_seed(1)
        model = CachedViewCAILP(data.x.size(1), 4, cfg)
        model.sparse_q = sparse
        _ = model(data)
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        torch.manual_seed(2)
        for _ in range(3):
            model.train()
            opt.zero_grad()
            out = model(data)
            (out["mu"].mean() + F.cross_entropy(out["logits"][data.train_mask], data.y[data.train_mask])).backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            outs.append(model(data)["mu"])
    torch.testing.assert_close(outs[0], outs[1], rtol=0, atol=1e-4)
