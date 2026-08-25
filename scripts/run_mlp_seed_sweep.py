"""CLI runner for multi-seed MLP baseline sweep.

Trains MLP baselines across a specified list of random seeds on frozen preprocessed artifacts,
saving canonical results to artifacts/results/<dataset>/<split>/mlp_seed<seed>/
for downstream matched graph lift alignment with GNN runs.
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd
import torch
from rich.console import Console
from rich.table import Table

from scgraph_bench.config.model import MLPConfig
from scgraph_bench.evaluation.metrics import (
    compute_evaluation_summary,
    confusion_matrix_to_dataframe,
)
from scgraph_bench.models.mlp import MLPBaseline
from scgraph_bench.preprocessing.schema import PreprocessedBundle
from scgraph_bench.tracking.mlflow_tracker import LocalMLflowTracker
from scgraph_bench.tracking.schema import RunManifest, RunStatus
from scgraph_bench.utils.paths import ArtifactPaths
from scgraph_bench.utils.seed import set_seed
from scgraph_bench.utils.versioning import get_code_version

console = Console()

DEFAULT_SEEDS = [7, 17, 42, 73, 101]


def run_mlp_seed_sweep(
    dataset_name: str = "stephenson_2021_healthy_pbmc",
    split_id: str = "site_stratified_seed42",
    seeds: list[int] | None = None,
    device: str = "auto",
    max_epochs: int = 500,
    patience: int = 50,
) -> list[dict[str, float | str | int]]:
    """Execute MLP baseline training across multiple seeds on precomputed features."""
    target_seeds = seeds if seeds is not None else DEFAULT_SEEDS

    target_device = (
        device
        if device in ["cuda", "mps", "cpu"]
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )

    paths = ArtifactPaths.default()
    prep_dir = paths.artifacts_dir / "preprocessed" / dataset_name / split_id

    if not (prep_dir / "feature_manifest.json").is_file():
        raise FileNotFoundError(f"Preprocessed feature bundle missing at {prep_dir}")

    console.print(f"[bold cyan]Loading preprocessed features from:[/bold cyan] {prep_dir}")
    prep_bundle = PreprocessedBundle.load(prep_dir)

    inv_label_map = {v: k for k, v in prep_bundle.label_to_id.items()}
    label_names = [inv_label_map[i] for i in range(len(prep_bundle.label_to_id))]

    # Load cell metadata for stratified donor/site metrics
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

    tracker = LocalMLflowTracker(experiment_name=f"scgraph-mlp-{split_id}")
    sweep_results: list[dict[str, float | str | int]] = []

    console.print(
        f"\n[bold green]Starting MLP Baseline Sweep on {target_device}: {len(target_seeds)} seeds ({target_seeds})[/bold green]\n"
    )

    for idx, seed in enumerate(target_seeds, 1):
        set_seed(seed)
        run_tag = f"[{idx}/{len(target_seeds)}] MLP Baseline (Seed {seed})"
        console.print(f"[bold cyan]Training {run_tag}...[/bold cyan]")

        # 1. Configure and fit MLP Baseline
        config = MLPConfig(
            input_dim=prep_bundle.X_pca_train.shape[1],
            hidden_dims=[128, 128],
            num_classes=len(label_names),
            dropout=0.3,
            learning_rate=1e-3,
            weight_decay=1e-4,
            batch_size=256,
            max_epochs=max_epochs,
            patience=patience,
            device=target_device,
            seed=seed,
        )
        mlp = MLPBaseline(config)
        mlp.fit(
            X_train=prep_bundle.X_pca_train,
            y_train=prep_bundle.train_labels,
            X_val=prep_bundle.X_pca_val,
            y_val=prep_bundle.val_labels,
        )

        # 2. Predict on all partitions
        train_preds = mlp.predict(prep_bundle.X_pca_train)
        val_preds = mlp.predict(prep_bundle.X_pca_val)
        test_preds = mlp.predict(prep_bundle.X_pca_test)

        train_probs = mlp.predict_proba(prep_bundle.X_pca_train)
        val_probs = mlp.predict_proba(prep_bundle.X_pca_val)
        test_probs = mlp.predict_proba(prep_bundle.X_pca_test)

        # 3. Compute evaluation summaries
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

        # 4. Target directories: seed-specific and legacy mlp if seed=42
        target_dirs = [
            paths.artifacts_dir / "results" / dataset_name / split_id / f"mlp_seed{seed}",
        ]
        if seed == 42:
            target_dirs.append(paths.artifacts_dir / "results" / dataset_name / split_id / "mlp")

        cm_test_df = confusion_matrix_to_dataframe(test_summary.confusion_matrix, label_names)
        eval_summaries = {
            "train": train_summary,
            "val": val_summary,
            "test": test_summary,
        }

        run_manifest = RunManifest(
            run_id=f"mlp_{split_id}_seed{seed}",
            status=RunStatus.SUCCESS,
            model_name="mlp",
            model_config_hash=config.compute_hash(),
            dataset_name=dataset_name,
            dataset_version="2025-11-08",
            split_id=split_id,
            split_hash=prep_bundle.manifest.split_config_hash,
            feature_manifest_hash=prep_bundle.manifest.compute_manifest_hash(),
            preprocessing_config_hash=prep_bundle.manifest.preprocessing_config_hash,
            graph_artifact_hash=None,
            label_mapping_hash=prep_bundle.manifest.label_mapping_hash,
            seed=seed,
            device=mlp.device_info_,
            code_version=get_code_version(),
            parameter_count=mlp.parameter_count_,
            best_epoch=mlp.best_epoch_,
            best_val_macro_f1=mlp.best_val_macro_f1_,
            training_time_seconds=mlp.training_time_seconds_,
        )

        # Build tidy metrics
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
                        "partition": part,
                        "metric_name": metric_name,
                        "metric_value": val,
                        "num_samples": summ.num_samples,
                        "num_classes": len(label_names),
                        "best_epoch": mlp.best_epoch_,
                        "seed": seed,
                    }
                )
        tidy_df = pd.DataFrame(tidy_rows)

        for out_dir in target_dirs:
            out_dir.mkdir(parents=True, exist_ok=True)
            np.save(out_dir / "test_preds.npy", test_preds)
            np.save(out_dir / "test_probs.npy", test_probs)
            np.save(out_dir / "val_preds.npy", val_preds)
            np.save(out_dir / "val_probs.npy", val_probs)
            np.save(out_dir / "train_preds.npy", train_preds)
            np.save(out_dir / "train_probs.npy", train_probs)

            np.save(out_dir / "embeddings_train.npy", mlp.embed(prep_bundle.X_pca_train))
            np.save(out_dir / "embeddings_val.npy", mlp.embed(prep_bundle.X_pca_val))
            np.save(out_dir / "embeddings_test.npy", mlp.embed(prep_bundle.X_pca_test))
            if not mlp.training_history_.empty:
                mlp.training_history_.to_csv(out_dir / "training_history.csv", index=False)

            cm_test_df.to_csv(out_dir / "confusion_matrix_test.csv")
            (out_dir / "metrics_summary.json").write_text(
                json.dumps(
                    {k: v.model_dump(mode="json") for k, v in eval_summaries.items()}, indent=2
                ),
                encoding="utf-8",
            )
            (out_dir / "run_manifest.json").write_text(
                run_manifest.model_dump_json(indent=2), encoding="utf-8"
            )
            tidy_df.to_csv(out_dir / "tidy_metrics.csv", index=False)

        # Log to MLflow
        primary_dir = target_dirs[0]
        tracker.log_run(
            manifest=run_manifest,
            evaluation_summaries=eval_summaries,
            artifacts={
                "confusion_matrix_test.csv": primary_dir / "confusion_matrix_test.csv",
                "tidy_metrics.csv": primary_dir / "tidy_metrics.csv",
            },
        )

        res_entry = {
            "seed": seed,
            "train_macro_f1": train_summary.macro_f1,
            "val_macro_f1": val_summary.macro_f1,
            "test_macro_f1": test_summary.macro_f1,
            "test_balanced_acc": test_summary.balanced_accuracy,
            "best_epoch": mlp.best_epoch_,
            "runtime_seconds": mlp.training_time_seconds_,
        }
        sweep_results.append(res_entry)

        console.print(
            f"[bold green]✔ {run_tag}[/bold green] -> Test Macro-F1: [bold cyan]{test_summary.macro_f1:.4f}[/bold cyan] | Test BalAcc: [bold blue]{test_summary.balanced_accuracy:.4f}[/bold blue] | Best Epoch: {mlp.best_epoch_} ({mlp.training_time_seconds_:.1f}s)"
        )

    # Summary Table across entire sweep
    console.print("\n")
    table = Table(title=f"MLP Baseline Seed Sweep Summary ({dataset_name} - {split_id})")
    table.add_column("Seed", style="yellow")
    table.add_column("Train Macro-F1", style="green")
    table.add_column("Val Macro-F1", style="magenta")
    table.add_column("Test Macro-F1", style="bold cyan")
    table.add_column("Balanced Acc", style="blue")
    table.add_column("Best Epoch", style="cyan")
    table.add_column("Runtime (s)", style="green")

    for r in sweep_results:
        table.add_row(
            str(r["seed"]),
            f"{float(r['train_macro_f1']):.4f}",
            f"{float(r['val_macro_f1']):.4f}",
            f"[bold]{float(r['test_macro_f1']):.4f}[/bold]",
            f"{float(r['test_balanced_acc']):.4f}",
            str(r["best_epoch"]),
            f"{float(r['runtime_seconds']):.1f}s",
        )
    console.print(table)
    return sweep_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run multi-seed MLP baseline sweep on frozen features."
    )
    parser.add_argument("--dataset", type=str, default="stephenson_2021_healthy_pbmc")
    parser.add_argument("--split", type=str, default="site_stratified_seed42")
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=DEFAULT_SEEDS,
        help="Random seeds to train MLP baselines for (e.g. --seeds 17 73 101).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Target device (auto, cuda, mps, cpu).",
    )
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--patience", type=int, default=50)
    args = parser.parse_args()

    run_mlp_seed_sweep(
        dataset_name=args.dataset,
        split_id=args.split,
        seeds=args.seeds,
        device=args.device,
        max_epochs=args.epochs,
        patience=args.patience,
    )
