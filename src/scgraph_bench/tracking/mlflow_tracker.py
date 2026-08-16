"""MLflow-compatible local experiment tracking interface."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from scgraph_bench.evaluation.schema import EvaluationSummary
from scgraph_bench.tracking.schema import RunManifest
from scgraph_bench.utils.logging import get_logger
from scgraph_bench.utils.paths import ArtifactPaths

logger = get_logger("tracking.mlflow")


class LocalMLflowTracker:
    """Local MLflow-compatible experiment tracker recording parameters, metrics, and artifacts."""

    def __init__(
        self,
        experiment_name: str = "scgraph-bench",
        tracking_uri: str | Path | None = None,
    ) -> None:
        paths = ArtifactPaths.default()
        self.experiment_name = experiment_name
        self.tracking_dir = Path(tracking_uri) if tracking_uri else paths.artifacts_dir / "mlruns"
        self.tracking_dir.mkdir(parents=True, exist_ok=True)

    def log_run(
        self,
        manifest: RunManifest,
        evaluation_summaries: dict[str, EvaluationSummary],
        artifacts: dict[str, Path | str] | None = None,
    ) -> Path:
        """Log a complete experimental run adhering to MLflow directory structure.

        Directory structure:
        <tracking_dir>/<experiment_name>/<run_id>/
            ├── params/
            ├── metrics/
            ├── tags/
            └── artifacts/

        Args:
            manifest: Cryptographic run manifest with provenance.
            evaluation_summaries: Partition-keyed dictionary of EvaluationSummary objects.
            artifacts: Optional dictionary of artifact names and filepaths to copy/record.

        Returns:
            Path to recorded run directory.
        """
        run_dir = self.tracking_dir / self.experiment_name / manifest.run_id
        params_dir = run_dir / "params"
        metrics_dir = run_dir / "metrics"
        tags_dir = run_dir / "tags"
        art_dir = run_dir / "artifacts"

        for dir_path in (params_dir, metrics_dir, tags_dir, art_dir):
            dir_path.mkdir(parents=True, exist_ok=True)

        # 1. Log parameters
        params: dict[str, Any] = {
            "model_name": manifest.model_name,
            "dataset_name": manifest.dataset_name,
            "split_id": manifest.split_id,
            "seed": str(manifest.seed),
            "model_config_hash": manifest.model_config_hash,
            "feature_manifest_hash": manifest.feature_manifest_hash,
            "label_mapping_hash": manifest.label_mapping_hash,
            "graph_artifact_hash": manifest.graph_artifact_hash or "none",
        }
        if manifest.parameter_count is not None:
            params["parameter_count"] = str(manifest.parameter_count)
        if manifest.selected_params is not None:
            for k, v in manifest.selected_params.items():
                params[f"param_{k}"] = str(v)

        for k, v in params.items():
            (params_dir / k).write_text(str(v), encoding="utf-8")

        # 2. Log metrics
        for part, summ in evaluation_summaries.items():
            prefix = f"{part}_"
            metric_kvs = {
                f"{prefix}macro_f1": summ.macro_f1,
                f"{prefix}weighted_f1": summ.weighted_f1,
                f"{prefix}balanced_accuracy": summ.balanced_accuracy,
                f"{prefix}overall_accuracy": summ.overall_accuracy,
                f"{prefix}macro_precision": summ.macro_precision,
                f"{prefix}macro_recall": summ.macro_recall,
            }
            for s in summ.per_site:
                metric_kvs[f"{prefix}site_{s.site}_observed_macro_f1"] = s.observed_class_macro_f1
                metric_kvs[f"{prefix}site_{s.site}_global_macro_f1"] = s.global_label_macro_f1
            for donor_metric in summ.per_donor:
                metric_kvs[f"{prefix}donor_{donor_metric.donor_id}_observed_macro_f1"] = (
                    donor_metric.observed_class_macro_f1
                )

            for mk, mv in metric_kvs.items():
                (metrics_dir / mk).write_text(f"{mv:.6f}\n", encoding="utf-8")

        # 3. Log tags
        tags = {
            "status": manifest.status.value,
            "created_at_utc": manifest.created_at_utc,
            "dataset_version": manifest.dataset_version,
        }
        for k, v in tags.items():
            (tags_dir / k).write_text(str(v), encoding="utf-8")

        # 4. Save manifest and artifacts
        (art_dir / "run_manifest.json").write_text(
            manifest.model_dump_json(indent=2), encoding="utf-8"
        )
        if artifacts:
            for art_name, art_path in artifacts.items():
                p = Path(art_path)
                if p.is_file():
                    target_file = art_dir / art_name
                    target_file.write_bytes(p.read_bytes())

        logger.info("Logged run '%s' to MLflow local directory: %s", manifest.run_id, run_dir)
        return run_dir
