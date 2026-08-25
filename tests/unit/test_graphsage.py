"""Unit tests for GraphSAGE model architecture, strict label isolation, and BBKNN sweep."""

import numpy as np
import torch
from torch_geometric.data import Data

from scgraph_bench.models.graphsage import (
    GraphSAGEClassifier,
    GraphSAGEConfig,
    GraphSAGENet,
)


def test_graphsage_network_forward_pass_shapes():
    """Verify that GraphSAGENet produces correct logit tensor shapes [N, num_classes]."""
    x = torch.randn(100, 50)
    edge_index = torch.randint(0, 100, (2, 300), dtype=torch.long)
    net = GraphSAGENet(in_features=50, hidden_dim=64, num_classes=12, dropout=0.1, aggr="mean")

    out = net(x, edge_index)
    assert out.shape == (100, 12)
    assert not torch.isnan(out).any()


def test_graphsage_classifier_fit_predict_strict_label_isolation():
    """Verify that GraphSAGEClassifier fits and predicts with strict label isolation on train_mask."""
    n_tr, n_va, n_te = 60, 20, 20
    n_tot = n_tr + n_va + n_te
    n_classes = 4

    x = torch.randn(n_tot, 50)
    y_train = torch.randint(0, n_classes, (n_tr,))
    y_val = np.random.randint(0, n_classes, n_va)

    train_mask = torch.zeros(n_tot, dtype=torch.bool)
    val_mask = torch.zeros(n_tot, dtype=torch.bool)
    test_mask = torch.zeros(n_tot, dtype=torch.bool)

    train_mask[:n_tr] = True
    val_mask[n_tr : n_tr + n_va] = True
    test_mask[n_tr + n_va :] = True

    # Build strict inductive edges: train-train and train-to-val/test
    src_tr = torch.randint(0, n_tr, (100,))
    dst_tr = torch.randint(0, n_tr, (100,))
    src_val = torch.randint(0, n_tr, (50,))
    dst_val = torch.randint(n_tr, n_tr + n_va, (50,))
    src_te = torch.randint(0, n_tr, (50,))
    dst_te = torch.randint(n_tr + n_va, n_tot, (50,))

    edge_index = torch.stack(
        [
            torch.cat([src_tr, src_val, src_te]),
            torch.cat([dst_tr, dst_val, dst_te]),
        ]
    )

    y_full = torch.full((n_tot,), fill_value=-1, dtype=torch.long)
    y_full[train_mask] = y_train

    data = Data(
        x=x,
        edge_index=edge_index,
        y=y_full,
        train_mask=train_mask,
        val_mask=val_mask,
        test_mask=test_mask,
    )

    cfg = GraphSAGEConfig(
        in_features=50,
        hidden_dim=32,
        num_classes=n_classes,
        max_epochs=10,
        patience=5,
        seed=42,
        device="cpu",
    )
    clf = GraphSAGEClassifier(cfg)
    clf.fit(pyg_data=data, val_labels=y_val)

    assert clf.best_epoch_ is not None and clf.best_epoch_ >= 1
    assert clf.best_val_macro_f1_ is not None

    tr_preds, va_preds, te_preds = clf.predict_all(data)
    assert len(tr_preds) == n_tr
    assert len(va_preds) == n_va
    assert len(te_preds) == n_te
    assert np.all(tr_preds >= 0) and np.all(tr_preds < n_classes)


def test_graphsage_history_and_embed_all():
    """Verify per-epoch history capture and hidden-layer embedding extraction."""
    n_tr, n_va, n_te = 60, 20, 20
    n_tot = n_tr + n_va + n_te
    n_classes = 4
    hidden_dim = 32

    x = torch.randn(n_tot, 50)
    y_train = torch.randint(0, n_classes, (n_tr,))
    y_val = np.random.randint(0, n_classes, n_va)

    train_mask = torch.zeros(n_tot, dtype=torch.bool)
    val_mask = torch.zeros(n_tot, dtype=torch.bool)
    test_mask = torch.zeros(n_tot, dtype=torch.bool)
    train_mask[:n_tr] = True
    val_mask[n_tr : n_tr + n_va] = True
    test_mask[n_tr + n_va :] = True

    src = torch.randint(0, n_tr, (200,))
    dst = torch.randint(0, n_tot, (200,))
    edge_index = torch.stack([src, dst])

    y_full = torch.full((n_tot,), fill_value=-1, dtype=torch.long)
    y_full[train_mask] = y_train

    data = Data(
        x=x,
        edge_index=edge_index,
        y=y_full,
        train_mask=train_mask,
        val_mask=val_mask,
        test_mask=test_mask,
    )

    cfg = GraphSAGEConfig(
        in_features=50,
        hidden_dim=hidden_dim,
        num_classes=n_classes,
        max_epochs=8,
        patience=8,
        seed=42,
        device="cpu",
    )
    clf = GraphSAGEClassifier(cfg)
    clf.fit(pyg_data=data, val_labels=y_val)

    assert len(clf.history_) >= 1
    assert len(clf.history_) <= cfg.max_epochs
    assert all({"epoch", "train_loss", "val_loss", "val_macro_f1"} <= set(h) for h in clf.history_)

    tr_emb, va_emb, te_emb = clf.embed_all(data)
    assert tr_emb.shape == (n_tr, hidden_dim)
    assert va_emb.shape == (n_va, hidden_dim)
    assert te_emb.shape == (n_te, hidden_dim)
    assert np.isfinite(te_emb).all()


def test_run_graphsage_bbknn_sweep_smoke():
    """Smoke test: verify run_graphsage_bbknn_sweep executes on BBKNN graph on CPU."""
    from scripts.run_graphsage_bbknn_sweep import run_graphsage_bbknn_sweep

    from scgraph_bench.utils.paths import ArtifactPaths

    paths = ArtifactPaths.default()
    prep_dir = (
        paths.artifacts_dir
        / "preprocessed"
        / "stephenson_2021_healthy_pbmc"
        / "site_stratified_seed42"
    )
    g_dir = (
        paths.artifacts_dir
        / "graphs"
        / "stephenson_2021_healthy_pbmc"
        / "site_stratified_seed42"
        / "bbknn_kperbatch2_donors12"
    )

    if (prep_dir / "feature_manifest.json").is_file() and (g_dir / "graph_manifest.json").is_file():
        results = run_graphsage_bbknn_sweep(
            dataset_name="stephenson_2021_healthy_pbmc",
            split_id="site_stratified_seed42",
            graph_name="bbknn_kperbatch2_donors12",
            seeds=[42],
            device="cpu",
            max_epochs=1,
            patience=1,
        )
        assert len(results) == 1
        assert results[0]["graph_name"] == "bbknn_kperbatch2_donors12"
        assert results[0]["seed"] == 42
        assert "test_macro_f1" in results[0]
        assert "matched_graph_lift" in results[0]
