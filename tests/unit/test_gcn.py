"""Unit tests for GCN model architecture, strict label isolation, and device placement."""

import numpy as np
import torch
from torch_geometric.data import Data

from scgraph_bench.models.gcn import GCNClassifier, GCNConfig, GCNNet


def test_gcn_network_forward_pass_shapes():
    """Verify that GCNNet produces correct logit tensor shapes [N, num_classes]."""
    x = torch.randn(100, 50)
    edge_index = torch.randint(0, 100, (2, 300), dtype=torch.long)
    net = GCNNet(in_features=50, hidden_dim=64, num_classes=12, dropout=0.1)

    out = net(x, edge_index)
    assert out.shape == (100, 12)
    assert not torch.isnan(out).any()


def test_gcn_classifier_fit_predict_strict_label_isolation():
    """Verify that GCNClassifier fits and predicts with strict label isolation on train_mask."""
    n_tr, n_va, n_te = 60, 20, 20
    n_tot = n_tr + n_va + n_te
    n_classes = 4

    x = torch.randn(n_tot, 50)
    y_train = torch.randint(0, n_classes, (n_tr,))
    y_val = np.random.randint(0, n_classes, n_va)
    np.random.randint(0, n_classes, n_te)

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

    cfg = GCNConfig(
        in_features=50,
        hidden_dim=32,
        num_classes=n_classes,
        max_epochs=10,
        patience=5,
        seed=42,
        device="cpu",
    )
    clf = GCNClassifier(cfg)
    clf.fit(pyg_data=data, val_labels=y_val)

    assert clf.best_epoch_ is not None and clf.best_epoch_ >= 1
    assert clf.best_val_macro_f1_ is not None

    tr_preds, va_preds, te_preds = clf.predict_all(data)
    assert len(tr_preds) == n_tr
    assert len(va_preds) == n_va
    assert len(te_preds) == n_te
    assert np.all(tr_preds >= 0) and np.all(tr_preds < n_classes)


def test_gcn_history_and_embed_all():
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

    cfg = GCNConfig(
        in_features=50,
        hidden_dim=hidden_dim,
        num_classes=n_classes,
        max_epochs=8,
        patience=8,
        seed=42,
        device="cpu",
    )
    clf = GCNClassifier(cfg)
    clf.fit(pyg_data=data, val_labels=y_val)

    assert len(clf.history_) >= 1
    assert len(clf.history_) <= cfg.max_epochs
    epochs = [h["epoch"] for h in clf.history_]
    assert epochs == sorted(epochs)
    assert all({"epoch", "train_loss", "val_loss", "val_macro_f1"} <= set(h) for h in clf.history_)
    best_hist_epoch = max(clf.history_, key=lambda h: h["val_macro_f1"])["epoch"]
    assert clf.best_epoch_ == best_hist_epoch

    tr_emb, va_emb, te_emb = clf.embed_all(data)
    assert tr_emb.shape == (n_tr, hidden_dim)
    assert va_emb.shape == (n_va, hidden_dim)
    assert te_emb.shape == (n_te, hidden_dim)
    assert np.isfinite(tr_emb).all()


def test_run_gcn_graph_sweep_smoke():
    """Smoke test: verify run_gcn_graph_sweep executes on precomputed artifacts on CPU."""
    from scripts.run_gcn_graph_sweep import run_gcn_graph_sweep

    from scgraph_bench.utils.paths import ArtifactPaths

    paths = ArtifactPaths.default()
    prep_dir = (
        paths.artifacts_dir
        / "preprocessed"
        / "stephenson_2021_healthy_pbmc"
        / "site_stratified_seed42"
    )

    if (prep_dir / "feature_manifest.json").is_file():
        results = run_gcn_graph_sweep(
            dataset_name="stephenson_2021_healthy_pbmc",
            split_id="site_stratified_seed42",
            graphs=["pca_knn_k24_unweighted"],
            seeds=[42],
            device="cpu",
            max_epochs=1,
            patience=1,
        )
        assert len(results) == 1
        assert results[0]["graph_name"] == "pca_knn_k24_unweighted"
        assert results[0]["seed"] == 42
        assert "test_macro_f1" in results[0]
        assert "matched_graph_lift" in results[0]


def test_run_gcn_graph_sweep_skip_lift():
    """Verify run_gcn_graph_sweep with skip_lift=True completes without requiring matched MLP baselines."""
    from scripts.run_gcn_graph_sweep import run_gcn_graph_sweep

    from scgraph_bench.utils.paths import ArtifactPaths

    paths = ArtifactPaths.default()
    prep_dir = (
        paths.artifacts_dir
        / "preprocessed"
        / "stephenson_2021_healthy_pbmc"
        / "site_stratified_seed42"
    )

    if (prep_dir / "feature_manifest.json").is_file():
        results = run_gcn_graph_sweep(
            dataset_name="stephenson_2021_healthy_pbmc",
            split_id="site_stratified_seed42",
            graphs=["pca_knn_k24_unweighted"],
            seeds=[999],  # non-existent seed baseline
            device="cpu",
            max_epochs=1,
            patience=1,
            skip_lift=True,
        )
        assert len(results) == 1
        assert results[0]["seed"] == 999
        assert results[0]["matched_graph_lift"] == "N/A"


def test_compute_offline_graph_lift():
    """Verify compute_offline_graph_lift scans and aggregates result directories."""
    from scripts.compute_graph_lift_offline import compute_offline_graph_lift

    df = compute_offline_graph_lift(
        dataset_name="stephenson_2021_healthy_pbmc",
        split_id="site_stratified_seed42",
    )
    assert df is not None
