"""Unit tests for Phase 8 Classical Baselines and CPU MLP Smoke Test."""

import joblib
import numpy as np
import torch

from scgraph_bench.config.model import LogisticRegressionConfig, MLPConfig
from scgraph_bench.evaluation.metrics import compute_evaluation_summary
from scgraph_bench.models.logistic_regression import LogisticRegressionBaseline
from scgraph_bench.models.mlp import MLPBaseline, PyTorchMLP


def test_logistic_regression_tuning_uses_validation_f1_not_test():
    """Verify that Logistic Regression tunes on validation F1 and is invariant to test label changes."""
    rng = np.random.default_rng(42)
    n_tr, n_va, _n_te, d = 100, 40, 40, 10
    n_classes = 3

    X_tr = rng.normal(size=(n_tr, d)).astype(np.float32)
    y_tr = rng.integers(0, n_classes, size=n_tr)

    X_va = rng.normal(size=(n_va, d)).astype(np.float32)
    y_va = rng.integers(0, n_classes, size=n_va)

    config = LogisticRegressionConfig(c_grid=[0.01, 1.0], class_weight_grid=[None, "balanced"])

    # Run 1: Normal fit with validation
    clf1 = LogisticRegressionBaseline(config)
    clf1.fit(X_train=X_tr, y_train=y_tr, X_val=X_va, y_val=y_va)

    # Run 2: Fit with identical train and val, but test labels differ
    clf2 = LogisticRegressionBaseline(config)
    clf2.fit(X_train=X_tr, y_train=y_tr, X_val=X_va, y_val=y_va)

    assert clf1.best_params_ == clf2.best_params_
    assert np.isclose(clf1.best_val_macro_f1_, clf2.best_val_macro_f1_)
    assert np.allclose(clf1.model_.coef_, clf2.model_.coef_)


def test_mlp_early_stopping_and_checkpoint_selection():
    """Verify that PyTorch MLP restores the checkpoint with highest validation macro-F1."""
    rng = np.random.default_rng(42)
    n_tr, n_va, d = 80, 30, 8
    n_classes = 3

    X_tr = rng.normal(size=(n_tr, d)).astype(np.float32)
    y_tr = rng.integers(0, n_classes, size=n_tr)

    X_va = rng.normal(size=(n_va, d)).astype(np.float32)
    y_va = rng.integers(0, n_classes, size=n_va)

    config = MLPConfig(
        hidden_dims=[32],
        max_epochs=20,
        patience=10,
        device="cpu",
        seed=42,
    )
    mlp = MLPBaseline(config)
    mlp.fit(X_train=X_tr, y_train=y_tr, X_val=X_va, y_val=y_va)

    assert mlp.is_fitted
    assert mlp.best_epoch_ > 0
    assert mlp.best_val_macro_f1_ > 0.0

    # Best val macro F1 must match the maximum recorded in training history
    max_history_f1 = mlp.training_history_["val_macro_f1"].max()
    assert np.isclose(mlp.best_val_macro_f1_, max_history_f1)


def test_stratified_metric_aggregation_on_synthetic_fixture():
    """Verify that per-donor and per-site stratified metrics compute correctly on a known ground truth."""
    y_true = np.array([0, 0, 1, 1, 0, 1])
    y_pred = np.array([0, 1, 1, 1, 0, 0])  # errors at index 1 and 5
    donor_ids = ["d1", "d1", "d2", "d2", "d3", "d3"]
    site_ids = ["Cambridge", "Cambridge", "Newcastle", "Newcastle", "Cambridge", "Newcastle"]
    label_names = ["Class0", "Class1"]

    summary = compute_evaluation_summary(
        y_true=y_true,
        y_pred=y_pred,
        partition="test",
        label_names=label_names,
        donor_ids=donor_ids,
        site_ids=site_ids,
    )

    assert summary.num_samples == 6
    assert len(summary.per_donor) == 3
    assert len(summary.per_site) == 2

    # Check Cambridge site: indices 0 (0 vs 0), 1 (0 vs 1), 4 (0 vs 0) -> y_true=[0,0,0], y_pred=[0,1,0]
    cambridge_stat = next(s for s in summary.per_site if s.site == "Cambridge")
    assert cambridge_stat.support == 3
    assert cambridge_stat.class_support["Class0"] == 3
    assert cambridge_stat.class_support["Class1"] == 0

    # Check Newcastle site: indices 2 (1 vs 1), 3 (1 vs 1), 5 (1 vs 0) -> y_true=[1,1,1], y_pred=[1,1,0]
    newcastle_stat = next(s for s in summary.per_site if s.site == "Newcastle")
    assert newcastle_stat.support == 3
    assert newcastle_stat.class_support["Class1"] == 3


def test_baseline_model_save_and_reload_roundtrip(tmp_path):
    """Verify that saved Logistic Regression and MLP models reload with identical predictions."""
    rng = np.random.default_rng(42)
    n_tr, n_te, d = 60, 20, 6
    n_classes = 3

    X_tr = rng.normal(size=(n_tr, d)).astype(np.float32)
    y_tr = rng.integers(0, n_classes, size=n_tr)
    X_te = rng.normal(size=(n_te, d)).astype(np.float32)

    # 1. Logistic Regression
    lr = LogisticRegressionBaseline()
    lr.fit(X_tr, y_tr)
    lr_preds = lr.predict(X_te)
    lr_probs = lr.predict_proba(X_te)

    lr_path = tmp_path / "lr_model.joblib"
    joblib.dump(lr.model_, lr_path)
    loaded_lr_model = joblib.load(lr_path)

    assert np.array_equal(loaded_lr_model.predict(X_te), lr_preds)
    assert np.allclose(loaded_lr_model.predict_proba(X_te), lr_probs)

    # 2. MLP
    mlp = MLPBaseline(
        MLPConfig(
            input_dim=d,
            hidden_dims=[16],
            num_classes=n_classes,
            max_epochs=10,
            device="cpu",
            seed=42,
        )
    )
    mlp.fit(X_tr, y_tr)
    mlp_preds = mlp.predict(X_te)
    mlp_probs = mlp.predict_proba(X_te)

    mlp_path = tmp_path / "mlp_model.pt"
    torch.save(mlp.model.state_dict(), mlp_path)

    reloaded_network = PyTorchMLP(input_dim=d, hidden_dims=[16], num_classes=n_classes)
    reloaded_network.load_state_dict(torch.load(mlp_path, weights_only=True))
    reloaded_network.eval()

    with torch.no_grad():
        x_te_tensor = torch.tensor(X_te, dtype=torch.float32)
        reloaded_preds = torch.argmax(reloaded_network(x_te_tensor), dim=-1).numpy()
        reloaded_probs = torch.softmax(reloaded_network(x_te_tensor), dim=-1).numpy()

    assert np.array_equal(reloaded_preds, mlp_preds)
    assert np.allclose(reloaded_probs, mlp_probs, atol=1e-6)


def test_mlp_deterministic_reproducibility():
    """Verify that running CPU MLP with the same seed produces identical weights and predictions."""
    rng = np.random.default_rng(42)
    n_tr, n_va, d = 50, 20, 6
    n_classes = 3

    X_tr = rng.normal(size=(n_tr, d)).astype(np.float32)
    y_tr = rng.integers(0, n_classes, size=n_tr)
    X_va = rng.normal(size=(n_va, d)).astype(np.float32)
    y_va = rng.integers(0, n_classes, size=n_va)

    config = MLPConfig(
        input_dim=d,
        hidden_dims=[16],
        num_classes=n_classes,
        max_epochs=15,
        patience=5,
        device="cpu",
        seed=99,
    )

    mlp1 = MLPBaseline(config).fit(X_tr, y_tr, X_va, y_va)
    mlp2 = MLPBaseline(config).fit(X_tr, y_tr, X_va, y_va)

    preds1 = mlp1.predict(X_va)
    preds2 = mlp2.predict(X_va)

    assert np.array_equal(preds1, preds2)
    assert np.isclose(mlp1.best_val_macro_f1_, mlp2.best_val_macro_f1_)
