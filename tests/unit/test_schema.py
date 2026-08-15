"""Unit tests for tidy metric schema, failure tracking, and graph lift."""

from scgraph_bench.tracking.schema import (
    FailureMetadata,
    GraphLiftRecord,
    LabelSupportTracking,
    MetricRecord,
    RunStatus,
    TidyResultsCollection,
)


def test_metric_record_success():
    """Verify standard success metric record creation."""
    rec = MetricRecord(
        dataset="kang_pbmc",
        split_id="split_01",
        seed=42,
        graph_name="pca_knn_k20",
        graph_settings={"k": 20, "metric": "euclidean"},
        model="mlp",
        metric="macro_f1",
        value=0.885,
        partition="test",
        status=RunStatus.SUCCESS,
        config_hash="abc123hash",
        artifact_hash="xyz789hash",
        runtime_seconds=12.4,
    )
    assert rec.status == RunStatus.SUCCESS
    assert rec.value == 0.885
    assert rec.failure_metadata is None


def test_metric_record_failure_tracking():
    """Verify metric record with failure status and error metadata."""
    fail_meta = FailureMetadata(
        error_type="ValueError",
        error_message="Missing classes in validation partition",
        traceback_summary="Traceback: line 42 in split.py",
        failed_phase="splitting",
    )
    rec = MetricRecord(
        dataset="kang_pbmc",
        split_id="split_02",
        seed=42,
        graph_name="bbknn",
        model="mlp",
        metric="macro_f1",
        value=None,
        status=RunStatus.FAILED,
        failure_metadata=fail_meta,
        config_hash="abc123hash",
    )
    assert rec.status == RunStatus.FAILED
    assert rec.value is None
    assert rec.failure_metadata is not None
    assert rec.failure_metadata.error_type == "ValueError"


def test_label_support_tracking():
    """Verify distinct tracking of unsupported, low-support, and excluded labels."""
    tracker = LabelSupportTracking(
        unsupported_labels=["Rare_Plasma_Cell"],
        low_support_labels=["Dendritic_Cell"],
        excluded_labels=["Doublets_Unassigned"],
        evaluated_labels=["CD4_T", "CD8_T", "B_cell", "NK_cell", "Monocyte"],
    )
    assert "Rare_Plasma_Cell" in tracker.unsupported_labels
    assert "Dendritic_Cell" in tracker.low_support_labels
    assert "Doublets_Unassigned" in tracker.excluded_labels
    assert len(tracker.evaluated_labels) == 5


def test_graph_lift_record():
    """Verify graph lift record schema."""
    lift = GraphLiftRecord(
        dataset="kang_pbmc",
        split_id="split_01",
        seed=42,
        graph_name="pca_knn_k20",
        gnm_model="GCN",
        matched_mlp_model="MLP",
        gnn_macro_f1=0.890,
        matched_mlp_macro_f1=0.870,
        graph_lift=0.020,
        config_hash="hash_123",
        is_valid_match=True,
    )
    assert abs(lift.graph_lift - (lift.gnn_macro_f1 - lift.matched_mlp_macro_f1)) < 1e-6
    assert lift.is_valid_match is True


def test_tidy_results_dataframe_export():
    """Verify collection converts to a valid pandas DataFrame including error columns."""
    collection = TidyResultsCollection()

    # Add successful record
    collection.add_record(
        MetricRecord(
            dataset="kang_pbmc",
            split_id="split_01",
            seed=42,
            graph_name="pca_knn",
            model="logistic_regression",
            metric="macro_f1",
            value=0.85,
            status=RunStatus.SUCCESS,
            config_hash="h1",
        )
    )

    # Add failed record
    collection.add_record(
        MetricRecord(
            dataset="kang_pbmc",
            split_id="split_01",
            seed=42,
            graph_name="bbknn",
            model="mlp",
            metric="macro_f1",
            value=None,
            status=RunStatus.FAILED,
            failure_metadata=FailureMetadata(
                error_type="RuntimeError",
                error_message="CUDA out of memory simulation",
            ),
            config_hash="h2",
        )
    )

    df = collection.to_dataframe()
    assert len(df) == 2
    assert "error_type" in df.columns
    assert "error_message" in df.columns
    assert df.loc[df["status"] == "failed", "error_type"].iloc[0] == "RuntimeError"
