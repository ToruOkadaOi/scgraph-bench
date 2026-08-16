"""Unit tests for Phase 10 Tidy Result Aggregation, Matched Graph Lift, and MLflow Tracking."""

import numpy as np
import pytest

from scgraph_bench.evaluation.metrics import compute_evaluation_summary
from scgraph_bench.tracking.graph_lift import compute_matched_graph_lift
from scgraph_bench.tracking.mlflow_tracker import LocalMLflowTracker
from scgraph_bench.tracking.schema import (
    MetricRecord,
    RunManifest,
    TidyResultsCollection,
)


def test_tidy_metric_record_and_dataframe_conversion():
    """Verify that MetricRecord and TidyResultsCollection correctly export to pandas DataFrame."""
    collection = TidyResultsCollection()

    record = MetricRecord(
        run_id="test_run_1",
        dataset_name="synthetic",
        split_id="split_1",
        seed=42,
        graph_name="pca_knn_k20",
        model_name="mlp",
        metric_name="macro_f1",
        metric_value=0.8850,
        partition="test",
        feature_manifest_hash="feat_hash_1",
        label_mapping_hash="lbl_hash_1",
        model_config_hash="cfg_hash_1",
        runtime_seconds=12.5,
    )
    collection.add_record(record)

    df = collection.to_dataframe()
    assert len(df) == 1
    assert df["run_id"].iloc[0] == "test_run_1"
    assert np.isclose(df["metric_value"].iloc[0], 0.8850)
    assert df["model_name"].iloc[0] == "mlp"


def test_matched_graph_lift_invariants_fail_on_mismatch():
    """Verify that compute_matched_graph_lift raises ValueError on mismatched seed, split, or feature hashes."""
    gnn_manifest = RunManifest(
        run_id="gnn_run_1",
        model_name="gcn",
        model_config_hash="cfg_gnn",
        dataset_name="synthetic",
        split_id="split_42",
        feature_manifest_hash="feat_hash_A",
        label_mapping_hash="lbl_hash_A",
        seed=42,
    )

    mlp_manifest_bad_seed = RunManifest(
        run_id="mlp_run_1",
        model_name="mlp",
        model_config_hash="cfg_mlp",
        dataset_name="synthetic",
        split_id="split_42",
        feature_manifest_hash="feat_hash_A",
        label_mapping_hash="lbl_hash_A",
        seed=43,  # Mismatched seed!
    )

    dummy_summary = compute_evaluation_summary(
        y_true=np.array([0, 1]),
        y_pred=np.array([0, 1]),
        partition="test",
        label_names=["c0", "c1"],
    )

    with pytest.raises(ValueError, match="Seed mismatch"):
        compute_matched_graph_lift(
            gnn_summary=dummy_summary,
            mlp_summary=dummy_summary,
            gnn_manifest=gnn_manifest,
            mlp_manifest=mlp_manifest_bad_seed,
            graph_name="pca_knn_k20",
        )

    mlp_manifest_bad_prep = RunManifest(
        run_id="mlp_run_2",
        model_name="mlp",
        model_config_hash="cfg_mlp",
        dataset_name="synthetic",
        split_id="split_42",
        feature_manifest_hash="feat_hash_A",
        preprocessing_config_hash="prep_hash_DIFFERENT",
        label_mapping_hash="lbl_hash_A",
        seed=42,
    )
    with pytest.raises(ValueError, match="Preprocessing config hash mismatch"):
        compute_matched_graph_lift(
            gnn_summary=dummy_summary,
            mlp_summary=dummy_summary,
            gnn_manifest=gnn_manifest,
            mlp_manifest=mlp_manifest_bad_prep,
            graph_name="pca_knn_k20",
        )


def test_matched_graph_lift_calculation_accuracy():
    """Verify exact numerical calculation of overall, per-site, and per-donor graph lift."""
    gnn_manifest = RunManifest(
        run_id="gnn_seed42",
        model_name="gcn",
        model_config_hash="cfg_gnn",
        dataset_name="synthetic",
        split_id="split_42",
        feature_manifest_hash="feat_hash_A",
        label_mapping_hash="lbl_hash_A",
        seed=42,
    )

    mlp_manifest = RunManifest(
        run_id="mlp_seed42",
        model_name="mlp",
        model_config_hash="cfg_mlp",
        dataset_name="synthetic",
        split_id="split_42",
        feature_manifest_hash="feat_hash_A",
        label_mapping_hash="lbl_hash_A",
        seed=42,
    )

    # GNN achieves 100% (macro-F1 = 1.0)
    gnn_summary = compute_evaluation_summary(
        y_true=np.array([0, 0, 1, 1]),
        y_pred=np.array([0, 0, 1, 1]),
        partition="test",
        label_names=["c0", "c1"],
        donor_ids=["d1", "d1", "d2", "d2"],
        site_ids=["Cambridge", "Cambridge", "Newcastle", "Newcastle"],
    )

    # MLP makes 1 error on Cambridge donor d1 (macro-F1 < 1.0)
    mlp_summary = compute_evaluation_summary(
        y_true=np.array([0, 0, 1, 1]),
        y_pred=np.array([0, 1, 1, 1]),
        partition="test",
        label_names=["c0", "c1"],
        donor_ids=["d1", "d1", "d2", "d2"],
        site_ids=["Cambridge", "Cambridge", "Newcastle", "Newcastle"],
    )

    lift_rec = compute_matched_graph_lift(
        gnn_summary=gnn_summary,
        mlp_summary=mlp_summary,
        gnn_manifest=gnn_manifest,
        mlp_manifest=mlp_manifest,
        graph_name="pca_knn_k20",
    )

    assert lift_rec.is_valid_match
    assert lift_rec.overall_graph_lift > 0.0
    assert np.isclose(lift_rec.overall_graph_lift, gnn_summary.macro_f1 - mlp_summary.macro_f1)
    assert lift_rec.cambridge_lift is not None and lift_rec.cambridge_lift > 0.0
    assert np.isclose(lift_rec.newcastle_lift, 0.0)  # Both got 100% on Newcastle


def test_local_mlflow_tracker_logs_directory_structure(tmp_path):
    """Verify that LocalMLflowTracker writes parameters, metrics, tags, and artifacts."""
    tracker = LocalMLflowTracker(experiment_name="test_experiment", tracking_uri=tmp_path)

    manifest = RunManifest(
        run_id="test_run_mlflow",
        model_name="mlp",
        model_config_hash="cfg_hash",
        dataset_name="synthetic",
        split_id="split_1",
        feature_manifest_hash="feat_hash",
        label_mapping_hash="lbl_hash",
        seed=42,
        parameter_count=1234,
    )

    summary = compute_evaluation_summary(
        y_true=np.array([0, 1]),
        y_pred=np.array([0, 1]),
        partition="test",
        label_names=["c0", "c1"],
    )

    run_dir = tracker.log_run(
        manifest=manifest,
        evaluation_summaries={"test": summary},
    )

    assert run_dir.is_dir()
    assert (run_dir / "params" / "model_name").read_text() == "mlp"
    assert (run_dir / "params" / "seed").read_text() == "42"
    assert (run_dir / "metrics" / "test_macro_f1").is_file()
    assert (run_dir / "artifacts" / "run_manifest.json").is_file()
