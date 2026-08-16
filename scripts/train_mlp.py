"""CLI runner for PyTorch MLP baseline on CPU."""

from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd
import torch
from rich.console import Console
from rich.table import Table

from scgraph_bench.config.model import MLPConfig
from scgraph_bench.data.loaders import StephensonHealthyPBMCLoader
from scgraph_bench.evaluation.metrics import (
    compute_evaluation_summary,
    confusion_matrix_to_dataframe,
)
from scgraph_bench.models.mlp import MLPBaseline
from scgraph_bench.preprocessing.schema import PreprocessedBundle
from scgraph_bench.utils.paths import ArtifactPaths

console = Console()


def train_mlp_cli(
    dataset_name: str = "stephenson_2021_healthy_pbmc",
    split_id: str = "site_stratified_seed42",
    seed: int = 42,
    max_epochs: int = 500,
    patience: int = 50,
) -> None:
    paths = ArtifactPaths.default()
    prep_dir = paths.artifacts_dir / "preprocessed" / dataset_name / split_id

    if not (prep_dir / "feature_manifest.json").is_file():
        raise FileNotFoundError(f"Feature bundle missing at {prep_dir}")

    console.print(f"[blue]Loading preprocessed features from:[/blue] {prep_dir}")
    prep_bundle = PreprocessedBundle.load(prep_dir)

    # 1. Load metadata for stratified evaluation
    loader = StephensonHealthyPBMCLoader()
    adata = loader.load()
    obs_map = {str(cid): idx for idx, cid in enumerate(adata.obs_names)}

    train_donors = adata.obs.iloc[[obs_map[cid] for cid in prep_bundle.train_cell_ids]][
        "donor_id"
    ].tolist()
    val_donors = adata.obs.iloc[[obs_map[cid] for cid in prep_bundle.val_cell_ids]][
        "donor_id"
    ].tolist()
    test_donors = adata.obs.iloc[[obs_map[cid] for cid in prep_bundle.test_cell_ids]][
        "donor_id"
    ].tolist()

    train_sites = adata.obs.iloc[[obs_map[cid] for cid in prep_bundle.train_cell_ids]][
        "site"
    ].tolist()
    val_sites = adata.obs.iloc[[obs_map[cid] for cid in prep_bundle.val_cell_ids]]["site"].tolist()
    test_sites = adata.obs.iloc[[obs_map[cid] for cid in prep_bundle.test_cell_ids]][
        "site"
    ].tolist()

    inv_label_map = {v: k for k, v in prep_bundle.label_to_id.items()}
    label_names = [inv_label_map[i] for i in range(len(prep_bundle.label_to_id))]

    # 2. Fit PyTorch MLP
    config = MLPConfig(
        seed=seed,
        max_epochs=max_epochs,
        patience=patience,
        device="cpu",
    )
    mlp = MLPBaseline(config)
    console.print(
        f"[bold green]Fitting CPU MLP baseline (Seed={seed}, Max Epochs={max_epochs}, Patience={patience})...[/bold green]"
    )
    mlp.fit(
        X_train=prep_bundle.X_pca_train,
        y_train=prep_bundle.train_labels,
        X_val=prep_bundle.X_pca_val,
        y_val=prep_bundle.val_labels,
    )

    # 3. Predict on all partitions
    train_preds = mlp.predict(prep_bundle.X_pca_train)
    val_preds = mlp.predict(prep_bundle.X_pca_val)
    test_preds = mlp.predict(prep_bundle.X_pca_test)

    train_probs = mlp.predict_proba(prep_bundle.X_pca_train)
    val_probs = mlp.predict_proba(prep_bundle.X_pca_val)
    test_probs = mlp.predict_proba(prep_bundle.X_pca_test)

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
    out_dir = paths.artifacts_dir / "results" / dataset_name / split_id / "mlp"
    out_dir.mkdir(parents=True, exist_ok=True)

    if mlp.model is not None:
        torch.save(mlp.model.state_dict(), out_dir / "model.pt")

    np.save(out_dir / "train_preds.npy", train_preds)
    np.save(out_dir / "val_preds.npy", val_preds)
    np.save(out_dir / "test_preds.npy", test_preds)
    np.save(out_dir / "train_probs.npy", train_probs)
    np.save(out_dir / "val_probs.npy", val_probs)
    np.save(out_dir / "test_probs.npy", test_probs)

    mlp.training_history_.to_csv(out_dir / "training_history.csv", index=False)

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
        "run_id": f"mlp_{split_id}_seed{seed}",
        "status": "success",
        "model_name": "mlp",
        "model_config_hash": config.compute_hash(),
        "dataset_name": dataset_name,
        "dataset_version": "2025-11-08",
        "split_id": split_id,
        "split_hash": prep_bundle.manifest.split_config_hash,
        "feature_manifest_hash": prep_bundle.manifest.compute_manifest_hash(),
        "preprocessing_config_hash": prep_bundle.manifest.preprocessing_config_hash,
        "graph_artifact_hash": None,
        "label_mapping_hash": prep_bundle.manifest.label_mapping_hash,
        "seed": seed,
        "device": mlp.device_info_,
        "parameter_count": mlp.parameter_count_,
        "best_epoch": mlp.best_epoch_,
        "best_val_macro_f1": mlp.best_val_macro_f1_,
        "training_time_seconds": mlp.training_time_seconds_,
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
                    "model_name": "mlp",
                    "seed": seed,
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
                    "model_name": "mlp",
                    "seed": seed,
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
                    "model_name": "mlp",
                    "seed": seed,
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
                    "model_name": "mlp",
                    "seed": seed,
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
                    "model_name": "mlp",
                    "seed": seed,
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
                    "model_name": "mlp",
                    "seed": seed,
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
    summary_table = Table(
        title=f"PyTorch MLP Baseline (Seed={seed}): Partition Performance Summary"
    )
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
        "Validation (Early Stopped)",
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

    site_table = Table(title="PyTorch MLP: Test Partition Per-Site Performance")
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
        f"[bold green]MLP Baseline Completed Successfully! Artifacts saved in {out_dir}[/bold green]\n"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train PyTorch MLP baseline on CPU.")
    parser.add_argument("--dataset", type=str, default="stephenson_2021_healthy_pbmc")
    parser.add_argument("--split", type=str, default="site_stratified_seed42")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--patience", type=int, default=50)
    args = parser.parse_args()

    train_mlp_cli(
        dataset_name=args.dataset,
        split_id=args.split,
        seed=args.seed,
        max_epochs=args.epochs,
        patience=args.patience,
    )
