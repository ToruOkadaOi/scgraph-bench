"""Central results aggregation engine compiling tidy benchmark tables and matched graph lifts."""

from __future__ import annotations

import json
from pathlib import Path

from scgraph_bench.evaluation.schema import EvaluationSummary
from scgraph_bench.tracking.graph_lift import compute_matched_graph_lift
from scgraph_bench.tracking.schema import (
    MetricRecord,
    RunManifest,
    TidyResultsCollection,
)
from scgraph_bench.utils.logging import get_logger
from scgraph_bench.utils.paths import ArtifactPaths

logger = get_logger("tracking.aggregator")


class ResultsAggregator:
    """Discovers, validates, and aggregates benchmark run artifacts across splits and models."""

    def __init__(self, artifacts_dir: Path | str | None = None) -> None:
        paths = ArtifactPaths.default()
        self.results_base_dir = (
            Path(artifacts_dir) if artifacts_dir else paths.artifacts_dir / "results"
        )

    def aggregate_all(
        self,
        dataset_name: str = "stephenson_2021_healthy_pbmc",
        split_id: str = "site_stratified_seed42",
    ) -> TidyResultsCollection:
        """Scan results directory, validate cryptographic provenance, and compile tidy collection.

        Args:
            dataset_name: Dataset identifier.
            split_id: Split identifier.

        Returns:
            TidyResultsCollection containing all valid metric records and matched graph lifts.
        """
        split_results_dir = self.results_base_dir / dataset_name / split_id
        if not split_results_dir.is_dir():
            logger.warning("No results directory found at %s", split_results_dir)
            return TidyResultsCollection()

        collection = TidyResultsCollection()
        run_manifests: dict[str, RunManifest] = {}
        eval_summaries: dict[str, dict[str, EvaluationSummary]] = {}

        # 1. Discover all run directories
        for run_dir in sorted(split_results_dir.iterdir()):
            if not run_dir.is_dir():
                continue

            manifest_path = run_dir / "run_manifest.json"
            metrics_path = run_dir / "metrics_summary.json"

            if not manifest_path.is_file() or not metrics_path.is_file():
                continue

            try:
                manifest_dict = json.loads(manifest_path.read_text(encoding="utf-8"))
                manifest = RunManifest.model_validate(manifest_dict)
                run_manifests[manifest.run_id] = manifest

                metrics_dict = json.loads(metrics_path.read_text(encoding="utf-8"))
                summaries: dict[str, EvaluationSummary] = {}
                for part, part_data in metrics_dict.items():
                    summaries[part] = EvaluationSummary.model_validate(part_data)
                eval_summaries[manifest.run_id] = summaries

                # Extract tidy metric records
                self._extract_metric_records(manifest, summaries, collection)
            except Exception as e:
                logger.error("Failed to parse run artifacts at %s: %s", run_dir, e)

        # 2. Compute matched graph lifts across GNN and MLP pairs
        self._compute_all_matched_lifts(run_manifests, eval_summaries, collection)

        return collection

    def _extract_metric_records(
        self,
        manifest: RunManifest,
        summaries: dict[str, EvaluationSummary],
        collection: TidyResultsCollection,
    ) -> None:
        """Extract atomic MetricRecord objects from an evaluation summary."""
        for part, summ in summaries.items():
            # Global metrics
            for m_name, m_val in [
                ("macro_f1", summ.macro_f1),
                ("weighted_f1", summ.weighted_f1),
                ("balanced_accuracy", summ.balanced_accuracy),
                ("overall_accuracy", summ.overall_accuracy),
                ("macro_precision", summ.macro_precision),
                ("macro_recall", summ.macro_recall),
            ]:
                collection.add_record(
                    MetricRecord(
                        run_id=manifest.run_id,
                        dataset_name=manifest.dataset_name,
                        dataset_version=manifest.dataset_version,
                        split_id=manifest.split_id,
                        split_hash=manifest.split_hash,
                        seed=manifest.seed,
                        graph_name=manifest.graph_artifact_hash or "none",
                        model_name=manifest.model_name,
                        metric_name=m_name,
                        metric_value=m_val,
                        partition=part,
                        feature_manifest_hash=manifest.feature_manifest_hash,
                        preprocessing_config_hash=manifest.preprocessing_config_hash,
                        graph_artifact_hash=manifest.graph_artifact_hash,
                        label_mapping_hash=manifest.label_mapping_hash,
                        model_config_hash=manifest.model_config_hash,
                        runtime_seconds=manifest.training_time_seconds,
                    )
                )

            # Stratified site metrics
            for s in summ.per_site:
                for m_name, m_val in [
                    ("site_observed_macro_f1", s.observed_class_macro_f1),
                    ("site_global_macro_f1", s.global_label_macro_f1),
                    ("site_balanced_accuracy", s.balanced_accuracy),
                    ("site_overall_accuracy", s.overall_accuracy),
                ]:
                    collection.add_record(
                        MetricRecord(
                            run_id=manifest.run_id,
                            dataset_name=manifest.dataset_name,
                            dataset_version=manifest.dataset_version,
                            split_id=manifest.split_id,
                            split_hash=manifest.split_hash,
                            seed=manifest.seed,
                            graph_name=manifest.graph_artifact_hash or "none",
                            model_name=manifest.model_name,
                            metric_name=m_name,
                            metric_value=m_val,
                            partition=part,
                            site=s.site,
                            observed_support=s.support,
                            feature_manifest_hash=manifest.feature_manifest_hash,
                            preprocessing_config_hash=manifest.preprocessing_config_hash,
                            graph_artifact_hash=manifest.graph_artifact_hash,
                            label_mapping_hash=manifest.label_mapping_hash,
                            model_config_hash=manifest.model_config_hash,
                            runtime_seconds=manifest.training_time_seconds,
                        )
                    )

            # Stratified donor metrics
            for d in summ.per_donor:
                for m_name, m_val in [
                    ("donor_observed_macro_f1", d.observed_class_macro_f1),
                    ("donor_global_macro_f1", d.global_label_macro_f1),
                    ("donor_balanced_accuracy", d.balanced_accuracy),
                ]:
                    collection.add_record(
                        MetricRecord(
                            run_id=manifest.run_id,
                            dataset_name=manifest.dataset_name,
                            dataset_version=manifest.dataset_version,
                            split_id=manifest.split_id,
                            split_hash=manifest.split_hash,
                            seed=manifest.seed,
                            graph_name=manifest.graph_artifact_hash or "none",
                            model_name=manifest.model_name,
                            metric_name=m_name,
                            metric_value=m_val,
                            partition=part,
                            donor_id=d.donor_id,
                            site=d.site,
                            observed_support=d.support,
                            feature_manifest_hash=manifest.feature_manifest_hash,
                            preprocessing_config_hash=manifest.preprocessing_config_hash,
                            graph_artifact_hash=manifest.graph_artifact_hash,
                            label_mapping_hash=manifest.label_mapping_hash,
                            model_config_hash=manifest.model_config_hash,
                            runtime_seconds=manifest.training_time_seconds,
                        )
                    )

            # Per-class metrics
            for pc in summ.per_class:
                collection.add_record(
                    MetricRecord(
                        run_id=manifest.run_id,
                        dataset_name=manifest.dataset_name,
                        dataset_version=manifest.dataset_version,
                        split_id=manifest.split_id,
                        split_hash=manifest.split_hash,
                        seed=manifest.seed,
                        graph_name=manifest.graph_artifact_hash or "none",
                        model_name=manifest.model_name,
                        metric_name="class_f1",
                        metric_value=pc.f1,
                        partition=part,
                        class_label=pc.class_name,
                        observed_support=pc.support,
                        feature_manifest_hash=manifest.feature_manifest_hash,
                        preprocessing_config_hash=manifest.preprocessing_config_hash,
                        graph_artifact_hash=manifest.graph_artifact_hash,
                        label_mapping_hash=manifest.label_mapping_hash,
                        model_config_hash=manifest.model_config_hash,
                        runtime_seconds=manifest.training_time_seconds,
                    )
                )

    def _compute_all_matched_lifts(
        self,
        run_manifests: dict[str, RunManifest],
        eval_summaries: dict[str, dict[str, EvaluationSummary]],
        collection: TidyResultsCollection,
    ) -> None:
        """Find matching GNN and MLP pairs and calculate graph lift."""
        mlp_runs = {
            m.seed: (m, eval_summaries[m.run_id]["test"])
            for m in run_manifests.values()
            if m.model_name == "mlp" and "test" in eval_summaries.get(m.run_id, {})
        }

        gnn_runs = [
            (m, eval_summaries[m.run_id]["test"])
            for m in run_manifests.values()
            if m.model_name not in ("mlp", "logistic_regression")
            and "test" in eval_summaries.get(m.run_id, {})
        ]

        for gnn_m, gnn_test_sum in gnn_runs:
            if gnn_m.seed in mlp_runs:
                mlp_m, mlp_test_sum = mlp_runs[gnn_m.seed]
                try:
                    lift_record = compute_matched_graph_lift(
                        gnn_summary=gnn_test_sum,
                        mlp_summary=mlp_test_sum,
                        gnn_manifest=gnn_m,
                        mlp_manifest=mlp_m,
                        graph_name=gnn_m.graph_artifact_hash or "unknown_graph",
                    )
                    collection.add_lift(lift_record)
                except Exception as e:
                    logger.warning("Could not compute matched lift for %s: %s", gnn_m.run_id, e)
