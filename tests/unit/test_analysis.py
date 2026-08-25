"""Unit tests for the analysis subpackage: run flattening and confidence calibration."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from scgraph_bench.analysis.calibration import (
    confidence_margin,
    expected_calibration_error,
    max_probability_confidence,
    multiclass_brier_score,
    prediction_entropy,
    reliability_diagram_data,
    summarize_confidence,
)
from scgraph_bench.analysis.flatten import (
    compute_matched_per_class_deltas,
    describe_run,
    discover_run_records,
    flatten_per_class,
    flatten_per_donor,
    load_run_record,
)
from scgraph_bench.analysis.schema import CalibrationSummary
from scgraph_bench.evaluation.schema import (
    EvaluationSummary,
    PerClassMetric,
    StratifiedDonorMetric,
)
from scgraph_bench.tracking.schema import RunManifest


def _per_class(f1_values: list[float], offset: int = 0) -> list[PerClassMetric]:
    return [
        PerClassMetric(
            class_index=i,
            class_name=f"class_{i + offset}",
            precision=0.9,
            recall=0.9,
            f1=f1,
            support=100,
        )
        for i, f1 in enumerate(f1_values)
    ]


def _summary(
    partition: str = "test",
    macro_f1: float = 0.85,
    donors: list[StratifiedDonorMetric] | None = None,
) -> EvaluationSummary:
    return EvaluationSummary(
        partition=partition,
        num_samples=300,
        macro_f1=macro_f1,
        weighted_f1=macro_f1,
        balanced_accuracy=0.84,
        overall_accuracy=0.86,
        macro_precision=0.88,
        macro_recall=0.83,
        per_class=_per_class([0.8, 0.9, 0.85]),
        per_donor=donors or [],
        per_site=[],
        confusion_matrix=[[100, 0, 0], [0, 100, 0], [0, 10, 90]],
    )


def _manifest(run_id: str, model_name: str, seed: int = 7) -> RunManifest:
    return RunManifest(
        run_id=run_id,
        model_name=model_name,
        model_config_hash="a" * 64,
        dataset_name="synthetic_pbmc",
        split_id="split_x",
        feature_manifest_hash="b" * 64,
        label_mapping_hash="c" * 64,
        seed=seed,
    )


def _write_run(
    run_dir,
    run_id: str,
    model_name: str,
    seed: int,
    summaries: dict[str, EvaluationSummary],
) -> None:
    run_dir.mkdir(parents=True)
    (run_dir / "run_manifest.json").write_text(
        _manifest(run_id, model_name, seed).model_dump_json(), encoding="utf-8"
    )
    (run_dir / "metrics_summary.json").write_text(
        json.dumps({k: v.model_dump(mode="json") for k, v in summaries.items()}), encoding="utf-8"
    )


class TestDescribeRun:
    def test_gcn_run_pattern(self):
        assert describe_run("gcn_pca_knn_k20_unweighted_seed42", "gcn") == (
            "gcn",
            "pca_knn_k20_unweighted",
        )

    def test_graphsage_run_pattern(self):
        assert describe_run("graphsage_bbknn_kperbatch2_donors12_seed17", "graphsage") == (
            "graphsage",
            "bbknn_kperbatch2_donors12",
        )

    def test_non_graph_model_maps_to_none_graph(self):
        assert describe_run("mlp_seed42", "mlp") == ("mlp", "none")
        assert describe_run("logistic_regression", "logistic_regression") == (
            "logistic_regression",
            "none",
        )


class TestFlatten:
    def _make_results_root(self, tmp_path):
        res = tmp_path / "results"
        gnn_donors = [
            StratifiedDonorMetric(
                donor_id="D1",
                site="cambridge",
                observed_class_macro_f1=0.91,
                global_label_macro_f1=0.80,
                balanced_accuracy=0.88,
                support=150,
            )
        ]
        mlp_donors = [
            StratifiedDonorMetric(
                donor_id="D1",
                site="cambridge",
                observed_class_macro_f1=0.89,
                global_label_macro_f1=0.82,
                balanced_accuracy=0.87,
                support=150,
            )
        ]
        _write_run(
            res / "gcn_k20_seed7",
            "gcn_k20_seed7",
            "gcn",
            7,
            {"test": _summary(donors=gnn_donors)},
        )
        _write_run(
            res / "mlp_seed7",
            "mlp_seed7",
            "mlp",
            7,
            {"test": _summary(macro_f1=0.90, donors=mlp_donors)},
        )
        (res / "incomplete").mkdir()
        return res

    def test_discover_skips_incomplete_dirs(self, tmp_path):
        res = self._make_results_root(tmp_path / "root")
        records = discover_run_records(res)
        assert len(records) == 2
        assert {r.run_id for r in records} == {"gcn_k20_seed7", "mlp_seed7"}

    def test_missing_root_returns_empty(self, tmp_path):
        assert discover_run_records(tmp_path / "nope") == []

    def test_load_run_record_none_on_missing_files(self, tmp_path):
        d = tmp_path / "empty_run"
        d.mkdir()
        assert load_run_record(d) is None

    def test_flatten_per_class_rows_and_columns(self, tmp_path):
        res = self._make_results_root(tmp_path)
        df = flatten_per_class(discover_run_records(res))
        assert len(df) == 6
        assert {
            "run_id",
            "model_name",
            "graph_name",
            "seed",
            "partition",
            "class_index",
            "class_name",
            "f1",
            "support",
        }.issubset(df.columns)
        gcn_rows = df[df["model_name"] == "gcn"]
        assert set(gcn_rows["graph_name"]) == {"k20"}
        assert set(df[df["model_name"] == "mlp"]["graph_name"]) == {"none"}

    def test_flatten_per_donor_values(self, tmp_path):
        res = self._make_results_root(tmp_path)
        df = flatten_per_donor(discover_run_records(res))
        assert len(df) == 2
        row = df[df["model_name"] == "gcn"].iloc[0]
        assert row["donor_id"] == "D1"
        assert row["site"] == "cambridge"
        assert row["observed_class_macro_f1"] == pytest.approx(0.91)

    def test_partition_filter_excludes_other_partitions(self, tmp_path):
        res = tmp_path / "results"
        _write_run(
            res / "gcn_k20_seed7",
            "gcn_k20_seed7",
            "gcn",
            7,
            {"train": _summary(partition="train"), "test": _summary()},
        )
        df_test = flatten_per_class(discover_run_records(res), partitions=("test",))
        df_all = flatten_per_class(discover_run_records(res), partitions=("train", "test"))
        assert set(df_test["partition"]) == {"test"}
        assert set(df_all["partition"]) == {"train", "test"}


class TestCalibration:
    def test_perfectly_confident_correct_predictions_have_zero_ece(self):
        y = np.array([0, 1, 2])
        probs = np.eye(3)
        assert expected_calibration_error(y, probs) == pytest.approx(0.0)

    def test_confident_wrong_predictions_have_high_ece(self):
        y = np.array([0, 0])
        probs = np.array([[0.05, 0.95], [0.05, 0.95]])
        assert expected_calibration_error(y, probs) == pytest.approx(0.95)

    def test_uniform_predictions_ece_matches_accuracy_gap(self):
        y = np.array([0, 1, 0, 1])
        probs = np.full((4, 3), 1 / 3)
        ece = expected_calibration_error(y, probs, n_bins=15)
        assert ece == pytest.approx(abs(0.5 - 1 / 3))

    def test_brier_score_known_case(self):
        y = np.array([0])
        probs = np.array([[0.7, 0.2, 0.1]])
        expected = (0.7 - 1.0) ** 2 + 0.2**2 + 0.1**2
        assert multiclass_brier_score(y, probs) == pytest.approx(expected)

    def test_entropy_of_uniform_is_log_classes(self):
        probs = np.full((2, 4), 0.25)
        entropies = prediction_entropy(probs)
        assert np.allclose(entropies, np.log(4))

    def test_margin_and_max_probability(self):
        probs = np.array([[0.6, 0.3, 0.1]])
        assert max_probability_confidence(probs)[0] == pytest.approx(0.6)
        assert confidence_margin(probs)[0] == pytest.approx(0.3)

    def test_single_class_margin_is_one(self):
        probs = np.array([[1.0]])
        assert confidence_margin(probs)[0] == pytest.approx(1.0)

    def test_invalid_probabilities_raise(self):
        with pytest.raises(ValueError):
            expected_calibration_error(np.array([0]), np.array([[1.2, -0.2]]))

    def test_shape_mismatch_raises(self):
        with pytest.raises(ValueError):
            summarize_confidence(np.array([0, 1]), np.ones((3, 2)))

    def test_summarize_confidence_schema(self):
        rng = np.random.default_rng(42)
        logits = rng.normal(size=(200, 5))
        probs = np.exp(logits) / np.exp(logits).sum(axis=1, keepdims=True)
        y = rng.integers(0, 5, size=200)
        summary = summarize_confidence(y, probs, run_id="r1", n_bins=10)
        assert isinstance(summary, CalibrationSummary)
        assert summary.n_samples == 200
        assert len(summary.bins) == 10
        assert 0.0 <= summary.ece <= 1.0
        assert summary.brier_score >= 0.0

    def test_reliability_bins_counts_sum_to_n(self):
        rng = np.random.default_rng(7)
        probs = rng.dirichlet(np.ones(4), size=50)
        y = rng.integers(0, 4, size=50)
        bins = reliability_diagram_data(y, probs, n_bins=5)
        assert sum(b.count for b in bins) == 50
        assert all(b.bin_upper > b.bin_lower for b in bins)


class TestMatchedDeltaJoin:
    def test_delta_join_logic(self, tmp_path):
        res = tmp_path / "results"

        def write_with_f1(run_id, model, seed, f1s):
            summary = _summary()
            per_class = _per_class(f1s)
            summary = summary.model_copy(update={"per_class": per_class})
            _write_run(res / run_id, run_id, model, seed, {"test": summary})

        write_with_f1("mlp_seed7", "mlp", 7, [0.9, 0.8, 0.7])
        write_with_f1("gcn_k20_seed7", "gcn", 7, [0.8, 0.9, 0.6])

        deltas = compute_matched_per_class_deltas(flatten_per_class(discover_run_records(res)))
        assert deltas is not None
        by_class = deltas.set_index("class_index")["class_delta_f1"]
        assert by_class[0] == pytest.approx(-0.1)
        assert by_class[1] == pytest.approx(0.1)
        assert by_class[2] == pytest.approx(-0.1)

    def test_delta_join_returns_none_without_mlp(self, tmp_path):
        res = tmp_path / "results"
        _write_run(res / "gcn_k20_seed7", "gcn_k20_seed7", "gcn", 7, {"test": _summary()})
        df = flatten_per_class(discover_run_records(res))
        assert compute_matched_per_class_deltas(df) is None

    def test_delta_join_returns_none_with_only_mlp(self, tmp_path):
        res = tmp_path / "results"
        _write_run(res / "mlp_seed7", "mlp_seed7", "mlp", 7, {"test": _summary()})
        df = flatten_per_class(discover_run_records(res))
        assert compute_matched_per_class_deltas(df) is None


class TestEmbeddingQuality:
    def _clusters(self, n_per=60, dims=16, sep=14.0):
        rng = np.random.default_rng(3)
        centers = np.array([[sep * i] + [0.0] * (dims - 1) for i in range(3)])
        embs = np.vstack([c + rng.normal(scale=0.5, size=(n_per, dims)) for c in centers])
        y = np.repeat(np.arange(3), n_per)
        return embs, y

    def test_separable_embeddings_score_high(self):
        from scgraph_bench.analysis.embedding_quality import compute_embedding_quality

        emb, y = self._clusters()
        report = compute_embedding_quality(emb, y, representation_name="clean")
        assert report.silhouette_euclidean > 0.8
        assert report.knn_accuracy > 0.95
        assert report.centroid_separation > 2.0

    def test_overlapping_embeddings_score_low(self):
        from scgraph_bench.analysis.embedding_quality import compute_embedding_quality

        rng = np.random.default_rng(4)
        emb = rng.normal(size=(180, 8))
        y = rng.integers(0, 3, size=180)
        report = compute_embedding_quality(emb, y, representation_name="noise")
        assert report.silhouette_euclidean < 0.1
        assert report.knn_accuracy < 0.75

    def test_reference_probe_uses_train_split(self):
        from scgraph_bench.analysis.embedding_quality import compute_embedding_quality

        emb_tr, y_tr = self._clusters()
        emb_te, y_te = self._clusters(n_per=30)
        report = compute_embedding_quality(
            emb_te,
            y_te,
            representation_name="probe",
            reference_emb=emb_tr,
            reference_y_train=y_tr,
        )
        assert report.knn_accuracy is not None and report.knn_accuracy > 0.95

    def test_length_mismatch_raises(self):
        from scgraph_bench.analysis.embedding_quality import compute_embedding_quality

        emb, y = self._clusters()
        with pytest.raises(ValueError):
            compute_embedding_quality(emb, y[:-1], representation_name="bad")

    def test_compare_representations_table(self):
        from scgraph_bench.analysis.embedding_quality import (
            compare_representations,
            compute_embedding_quality,
        )

        emb, y = self._clusters()
        r1 = compute_embedding_quality(emb, y, representation_name="a")
        r2 = compute_embedding_quality(emb, y, representation_name="b")
        df = compare_representations([r1, r2])
        assert list(df["representation_name"]) == ["a", "b"]
        assert {"silhouette_euclidean", "knn_accuracy"}.issubset(df.columns)


class TestTrainingDynamicsLoading:
    def test_load_histories_and_summary(self, tmp_path):
        import pandas as pd
        from scripts.analyze_training_dynamics import load_histories, summarize_dynamics

        res_root = tmp_path / "results"
        epochs = np.arange(1, 11)
        hist = pd.DataFrame(
            {
                "epoch": epochs,
                "train_loss": np.linspace(2.0, 0.2, 10),
                "val_loss": np.linspace(1.8, 0.3, 10),
                "val_macro_f1": np.linspace(0.3, 0.9, 10),
            }
        )
        manifest = RunManifest(
            run_id="gcn_k20_seed7",
            model_name="gcn",
            model_config_hash="a" * 64,
            dataset_name="synthetic",
            split_id="s",
            feature_manifest_hash="b" * 64,
            label_mapping_hash="c" * 64,
            seed=7,
        )
        d = res_root / "gcn_k20_seed7"
        d.mkdir(parents=True)
        hist.to_csv(d / "training_history.csv", index=False)
        (d / "run_manifest.json").write_text(manifest.model_dump_json(), encoding="utf-8")
        (d / "metrics_summary.json").write_text("{}", encoding="utf-8")

        df = load_histories(res_root)
        assert len(df) == 10
        assert set(df["model_name"]) == {"gcn"}
        summary = summarize_dynamics(df)
        row = summary.iloc[0]
        assert row["best_val_f1_mean"] == pytest.approx(0.9)
        assert row["final_train_loss"] == pytest.approx(0.2)

    def test_load_histories_empty_root(self, tmp_path):
        from scripts.analyze_training_dynamics import load_histories

        assert load_histories(tmp_path / "missing").empty


def test_calibration_dataframe_roundtrip(tmp_path):
    rng = np.random.default_rng(0)
    probs = rng.dirichlet(np.ones(3), size=40)
    y = rng.integers(0, 3, size=40)
    summary = summarize_confidence(y, probs, run_id="rt")
    df = pd.DataFrame([b.model_dump() for b in summary.bins])
    csv_path = tmp_path / "bins.csv"
    df.to_csv(csv_path, index=False)
    reloaded = pd.read_csv(csv_path)
    assert len(reloaded) == len(summary.bins)
