"""CLI runner for Logistic Regression baseline."""

from __future__ import annotations

import argparse
import json

import joblib
import numpy as np
import pandas as pd
from rich.console import Console
from rich.table import Table

from scgraph_bench.config.model import LogisticRegressionConfig
from scgraph_bench.evaluation.metrics import (
    compute_evaluation_summary,
    confusion_matrix_to_dataframe,
)
from scgraph_bench.models.logistic_regression import LogisticRegressionBaseline
from scgraph_bench.preprocessing.schema import PreprocessedBundle
from scgraph_bench.utils.paths import ArtifactPaths

console = Console()


def train_logistic_regression_cli(
    dataset_name: str = "stephenson_2021_healthy_pbmc",
    split_id: str = "site_stratified_seed42",
) -> None:
    paths = ArtifactPaths.default()
    prep_dir = paths.artifacts_dir / "preprocessed" / dataset_name / split_id

    if not (prep_dir / "feature_manifest.json").is_file():
        raise FileNotFoundError(f"Feature bundle missing at {prep_dir}")

    console.print(f"[blue]Loading preprocessed features from:[/blue] {prep_dir}")
    prep_bundle = PreprocessedBundle.load(prep_dir)

    # 1. Load metadata for stratified evaluation
    cell_meta_file = prep_dir / "cell_metadata.json"
    if cell_meta_file.is_file():
        cell_meta = json.loads(cell_meta_file.read_text(encoding="utf-8"))
        train_donors = cell_meta["train_donors"]
        val_donors = cell_meta["val_donors"]
        test_donors = cell_meta["test_donors"]
        train_sites = cell_meta["train_sites"]
        val_sites = cell_meta["val_sites"]
        test_sites = cell_meta["test_sites"]
    else:
        train_donors, val_donors, test_donors = None, None, None
        train_sites, val_sites, test_sites = None, None, None

    inv_label_map = {v: k for k, v in prep_bundle.label_to_id.items()}
    label_names = [inv_label_map[i] for i in range(len(prep_bundle.label_to_id))]

    # 2. Fit Logistic Regression with validation-based tuning
    config = LogisticRegressionConfig()
    clf = LogisticRegressionBaseline(config)
    console.print(
        "[bold green]Fitting Logistic Regression baseline (tuning C & class_weight on validation macro-F1)...[/bold green]"
    )
    clf.fit(
        X_train=prep_bundle.X_pca_train,
        y_train=prep_bundle.train_labels,
        X_val=prep_bundle.X_pca_val,
        y_val=prep_bundle.val_labels,
    )

    # 3. Predict on all partitions
    train_preds = clf.predict(prep_bundle.X_pca_train)
    val_preds = clf.predict(prep_bundle.X_pca_val)
    test_preds = clf.predict(prep_bundle.X_pca_test)

    train_probs = clf.predict_proba(prep_bundle.X_pca_train)
    val_probs = clf.predict_proba(prep_bundle.X_pca_val)
    test_probs = clf.predict_proba(prep_bundle.X_pca_test)

    # 4. Compute comprehensive evaluation summaries
    train_summary = compute_evaluation_summary(
        prep_bundle.train_labels,
        train_preds,
        partition="train",
        label_names=label_names,
        donor_ids=train_donors,
        site_ids=train_sites,
    )
    val_summary = compute_evaluation_summary(
        prep_bundle.val_labels,
        val_preds,
        partition="val",
        label_names=label_names,
        donor_ids=val_donors,
        site_ids=val_sites,
    )
    test_summary = compute_evaluation_summary(
        prep_bundle.test_labels,
        test_preds,
        partition="test",
        label_names=label_names,
        donor_ids=test_donors,
        site_ids=test_sites,
    )

    # 5. Persist artifacts
    out_dir = paths.artifacts_dir / "results" / dataset_name / split_id / "logistic_regression"
    out_dir.mkdir(parents=True, exist_ok=True)

    joblib.dump(clf.model_, out_dir / "model.joblib")
    np.save(out_dir / "train_preds.npy", train_preds)
    np.save(out_dir / "val_preds.npy", val_preds)
    np.save(out_dir / "test_preds.npy", test_preds)
    np.save(out_dir / "train_probs.npy", train_probs)
    np.save(out_dir / "val_probs.npy", val_probs)
    np.save(out_dir / "test_probs.npy", test_probs)

    clf.selection_table_.to_csv(out_dir / "validation_selection_table.csv", index=False)

    # Confusion matrices
    cm_test_df = confusion_matrix_to_dataframe(test_summary.confusion_matrix, label_names)
    cm_test_df.to_csv(out_dir / "confusion_matrix_test.csv")

    metrics_payload = {
        "train": train_summary.model_dump(mode="json"),
        "val": val_summary.model_dump(mode="json"),
        "test": test_summary.model_dump(mode="json"),
    }
    (out_dir / "metrics_summary.json").write_text(
        json.dumps(metrics_payload, indent=2), encoding="utf-8"
    )

    # Run Manifest
    run_manifest = {
        "run_id": f"logistic_regression_{split_id}",
        "status": "success",
        "model_name": "logistic_regression",
        "model_config_hash": config.compute_hash(),
        "dataset_name": dataset_name,
        "dataset_version": "2025-11-08",
        "split_id": split_id,
        "split_hash": prep_bundle.manifest.split_config_hash,
        "feature_manifest_hash": prep_bundle.manifest.compute_manifest_hash(),
        "preprocessing_config_hash": prep_bundle.manifest.preprocessing_config_hash,
        "graph_artifact_hash": None,
        "label_mapping_hash": prep_bundle.manifest.label_mapping_hash,
        "seed": config.random_state,
        "selected_params": clf.best_params_,
        "training_time_seconds": clf.training_time_seconds_,
    }
    (out_dir / "run_manifest.json").write_text(json.dumps(run_manifest, indent=2), encoding="utf-8")

    # 6. Build tidy metrics dataframe
    tidy_rows = []
    for part, summ in [("train", train_summary), ("val", val_summary), ("test", test_summary)]:
        for metric_name, val in [
            ("macro_f1", summ.macro_f1),
            ("weighted_f1", summ.weighted_f1),
            ("balanced_accuracy", summ.balanced_accuracy),
            ("overall_accuracy", summ.overall_accuracy),
            ("macro_precision", summ.macro_precision),
            ("macro_recall", summ.macro_recall),
        ]:
            tidy_rows.append(
                {
                    "dataset_name": dataset_name,
                    "split_id": split_id,
                    "model_name": "logistic_regression",
                    "partition": part,
                    "metric_name": metric_name,
                    "metric_value": val,
                    "donor_id": None,
                    "site": None,
                    "class_label": None,
                }
            )
        for s in summ.per_site:
            tidy_rows.append(
                {
                    "dataset_name": dataset_name,
                    "split_id": split_id,
                    "model_name": "logistic_regression",
                    "partition": part,
                    "metric_name": "site_observed_macro_f1",
                    "metric_value": s.observed_class_macro_f1,
                    "donor_id": None,
                    "site": s.site,
                    "class_label": None,
                }
            )
            tidy_rows.append(
                {
                    "dataset_name": dataset_name,
                    "split_id": split_id,
                    "model_name": "logistic_regression",
                    "partition": part,
                    "metric_name": "site_global_macro_f1",
                    "metric_value": s.global_label_macro_f1,
                    "donor_id": None,
                    "site": s.site,
                    "class_label": None,
                }
            )
        for d in summ.per_donor:
            tidy_rows.append(
                {
                    "dataset_name": dataset_name,
                    "split_id": split_id,
                    "model_name": "logistic_regression",
                    "partition": part,
                    "metric_name": "donor_observed_macro_f1",
                    "metric_value": d.observed_class_macro_f1,
                    "donor_id": d.donor_id,
                    "site": d.site,
                    "class_label": None,
                }
            )
            tidy_rows.append(
                {
                    "dataset_name": dataset_name,
                    "split_id": split_id,
                    "model_name": "logistic_regression",
                    "partition": part,
                    "metric_name": "donor_global_macro_f1",
                    "metric_value": d.global_label_macro_f1,
                    "donor_id": d.donor_id,
                    "site": d.site,
                    "class_label": None,
                }
            )
        for pc in summ.per_class:
            tidy_rows.append(
                {
                    "dataset_name": dataset_name,
                    "split_id": split_id,
                    "model_name": "logistic_regression",
                    "partition": part,
                    "metric_name": "class_f1",
                    "metric_value": pc.f1,
                    "donor_id": None,
                    "site": None,
                    "class_label": pc.class_name,
                }
            )

    pd.DataFrame(tidy_rows).to_csv(out_dir / "tidy_metrics.csv", index=False)

    # 7. Print Rich Summary Tables
    summary_table = Table(title="Logistic Regression Baseline: Partition Performance Summary")
    summary_table.add_column("Partition", style="cyan")
    summary_table.add_column("Cell Count", style="green")
    summary_table.add_column("Macro-F1", style="magenta")
    summary_table.add_column("Balanced Accuracy", style="yellow")
    summary_table.add_column("Overall Accuracy", style="blue")

    summary_table.add_row(
        "Train (Fitted)",
        f"{train_summary.num_samples:,}",
        f"{train_summary.macro_f1:.4f}",
        f"{train_summary.balanced_accuracy:.4f}",
        f"{train_summary.overall_accuracy:.4f}",
    )
    summary_table.add_row(
        "Validation (Tuned)",
        f"{val_summary.num_samples:,}",
        f"{val_summary.macro_f1:.4f}",
        f"{val_summary.balanced_accuracy:.4f}",
        f"{val_summary.overall_accuracy:.4f}",
    )
    summary_table.add_row(
        "Test (Held-Out Donors)",
        f"{test_summary.num_samples:,}",
        f"{test_summary.macro_f1:.4f}",
        f"{test_summary.balanced_accuracy:.4f}",
        f"{test_summary.overall_accuracy:.4f}",
    )
    console.print(summary_table)

    site_table = Table(title="Logistic Regression: Test Partition Per-Site Performance")
    site_table.add_column("Site", style="cyan")
    site_table.add_column("Cell Count", style="green")
    site_table.add_column("Observed Macro-F1", style="magenta")
    site_table.add_column("Global Macro-F1", style="blue")
    site_table.add_column("Balanced Accuracy", style="yellow")
    site_table.add_column("Overall Accuracy", style="cyan")

    for s in test_summary.per_site:
        site_table.add_row(
            s.site,
            f"{s.support:,}",
            f"{s.observed_class_macro_f1:.4f}",
            f"{s.global_label_macro_f1:.4f}",
            f"{s.balanced_accuracy:.4f}",
            f"{s.overall_accuracy:.4f}",
        )
    console.print(site_table)

    console.print(
        f"[bold green]Logistic Regression Baseline Completed Successfully! Artifacts saved in {out_dir}[/bold green]\n"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Logistic Regression baseline.")
    parser.add_argument("--dataset", type=str, default="stephenson_2021_healthy_pbmc")
    parser.add_argument("--split", type=str, default="site_stratified_seed42")
    args = parser.parse_args()

    train_logistic_regression_cli(
        dataset_name=args.dataset,
        split_id=args.split,
    )
