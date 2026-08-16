"""CLI runner for 5-seed GraphSAGE benchmark on BBKNN graph.

Evaluates GraphSAGE across 5 seeds (7, 17, 42, 73, 101) on bbknn_kperbatch2_donors12
using immutable precomputed features and frozen split site_stratified_seed42.
Strictly matches against identically-seeded MLP baselines (mlp_seed<seed>).
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import torch
from rich.console import Console
from rich.table import Table

from scgraph_bench.evaluation.metrics import (
    compute_evaluation_summary,
    confusion_matrix_to_dataframe,
)
from scgraph_bench.evaluation.schema import EvaluationSummary
from scgraph_bench.graph.schema import GraphBundle
from scgraph_bench.models.graphsage import GraphSAGEClassifier, GraphSAGEConfig
from scgraph_bench.preprocessing.schema import PreprocessedBundle
from scgraph_bench.tracking.graph_lift import compute_matched_graph_lift
from scgraph_bench.tracking.mlflow_tracker import LocalMLflowTracker
from scgraph_bench.tracking.schema import RunManifest, RunStatus
from scgraph_bench.utils.paths import ArtifactPaths
from scgraph_bench.utils.seed import set_seed

console = Console()

DEFAULT_GRAPH = "bbknn_kperbatch2_donors12"
DEFAULT_SEEDS = [7, 17, 42, 73, 101]


def run_graphsage_bbknn_sweep(
    dataset_name: str = "stephenson_2021_healthy_pbmc",
    split_id: str = "site_stratified_seed42",
    graph_name: str = DEFAULT_GRAPH,
    seeds: list[int] | None = None,
    device: str = "auto",
    max_epochs: int = 500,
    patience: int = 50,
) -> list[dict[str, float | str | int]]:
    """Execute the GraphSAGE benchmark sweep across seeds on the specified BBKNN graph."""
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

    # Full node feature matrix
    X_all = np.vstack([prep_bundle.X_pca_train, prep_bundle.X_pca_val, prep_bundle.X_pca_test])
    y_train = prep_bundle.train_labels
    y_val = prep_bundle.val_labels
    y_test = prep_bundle.test_labels

    # Load Graph Bundle
    g_dir = paths.artifacts_dir / "graphs" / dataset_name / split_id / graph_name
    if not (g_dir / "graph_manifest.json").is_file():
        raise FileNotFoundError(f"Graph artifact missing at {g_dir}")

    console.print(f"[bold cyan]Loading Graph Bundle from:[/bold cyan] {g_dir}")
    graph_bundle = GraphBundle.load(g_dir)
    pyg_data = graph_bundle.to_pyg_data(x=X_all, y_train_only=y_train)

    tracker = LocalMLflowTracker(experiment_name=f"scgraph-graphsage-{split_id}")
    sweep_results: list[dict[str, float | str | int]] = []

    console.print(
        f"\n[bold green]Starting GraphSAGE Benchmark Sweep on {target_device}: Graph={graph_name}, Seeds={target_seeds}[/bold green]\n"
    )

    for s_idx, seed in enumerate(target_seeds, 1):
        set_seed(seed)
        run_tag = f"[{s_idx}/{len(target_seeds)}] GraphSAGE {graph_name} (Seed {seed})"

        # 1. Load exact matching MLP baseline for comparative graph lift
        mlp_res_dir = paths.artifacts_dir / "results" / dataset_name / split_id / f"mlp_seed{seed}"
        if not (mlp_res_dir / "run_manifest.json").is_file() and seed == 42:
            mlp_res_dir = paths.artifacts_dir / "results" / dataset_name / split_id / "mlp"

        if (
            not (mlp_res_dir / "run_manifest.json").is_file()
            or not (mlp_res_dir / "metrics_summary.json").is_file()
        ):
            raise FileNotFoundError(
                f"Strict Baseline Invariant Violation: Matched MLP baseline missing for Seed={seed}! "
                f"Run 'scripts/run_mlp_seed_sweep.py --seeds {seed}' first."
            )

        mlp_manifest = RunManifest.model_validate_json(
            (mlp_res_dir / "run_manifest.json").read_text(encoding="utf-8")
        )
        if mlp_manifest.seed != seed:
            raise ValueError(
                f"Strict Seed Mismatch: Expected MLP baseline seed {seed}, but found seed {mlp_manifest.seed} in {mlp_res_dir}."
            )

        mlp_metrics = json.loads((mlp_res_dir / "metrics_summary.json").read_text(encoding="utf-8"))
        mlp_test_summary = EvaluationSummary.model_validate(mlp_metrics["test"])

        # 2. Fit 2-layer GraphSAGE Classifier (50 -> 128 -> 12, BatchNorm, ReLU, Dropout 0.2, mean aggr)
        cfg = GraphSAGEConfig(
            in_features=prep_bundle.X_pca_train.shape[1],
            hidden_dim=128,
            num_classes=len(label_names),
            dropout=0.2,
            lr=0.001,
            weight_decay=1e-4,
            max_epochs=max_epochs,
            patience=patience,
            seed=seed,
            aggr="mean",
            device=target_device,
        )
        clf = GraphSAGEClassifier(cfg)
        clf.fit(pyg_data=pyg_data, val_labels=y_val)

        # 3. Predict across all partitions
        tr_preds, va_preds, te_preds = clf.predict_all(pyg_data)
        _, _, te_probs = clf.predict_proba_all(pyg_data)

        # 4. Compute summaries
        train_summary = compute_evaluation_summary(
            y_true=y_train,
            y_pred=tr_preds,
            partition="train",
            label_names=label_names,
            donor_ids=train_donors,
            site_ids=train_sites,
        )
        val_summary = compute_evaluation_summary(
            y_true=y_val,
            y_pred=va_preds,
            partition="val",
            label_names=label_names,
            donor_ids=val_donors,
            site_ids=val_sites,
        )
        test_summary = compute_evaluation_summary(
            y_true=y_test,
            y_pred=te_preds,
            partition="test",
            label_names=label_names,
            donor_ids=test_donors,
            site_ids=test_sites,
        )

        # 5. Persist Result Artifacts
        out_dir = (
            paths.artifacts_dir
            / "results"
            / dataset_name
            / split_id
            / f"graphsage_{graph_name}_seed{seed}"
        )
        out_dir.mkdir(parents=True, exist_ok=True)

        np.save(out_dir / "test_preds.npy", te_preds)
        np.save(out_dir / "test_probs.npy", te_probs)
        df_cm = confusion_matrix_to_dataframe(test_summary.confusion_matrix, label_names)
        df_cm.to_csv(out_dir / "confusion_matrix_test.csv")

        eval_summaries = {
            "train": train_summary,
            "val": val_summary,
            "test": test_summary,
        }
        (out_dir / "metrics_summary.json").write_text(
            json.dumps({k: v.model_dump(mode="json") for k, v in eval_summaries.items()}, indent=2),
            encoding="utf-8",
        )

        run_manifest = RunManifest(
            run_id=f"graphsage_{graph_name}_seed{seed}",
            status=RunStatus.SUCCESS,
            model_name="graphsage",
            model_config_hash=cfg.compute_hash(),
            dataset_name=dataset_name,
            dataset_version="2025-11-08",
            split_id=split_id,
            split_hash=prep_bundle.manifest.split_config_hash,
            feature_manifest_hash=prep_bundle.manifest.compute_manifest_hash(),
            preprocessing_config_hash=prep_bundle.manifest.preprocessing_config_hash,
            graph_artifact_hash=graph_name,
            label_mapping_hash=prep_bundle.manifest.label_mapping_hash,
            seed=seed,
            device=clf.device_info_,
            parameter_count=clf.parameter_count_,
            best_epoch=clf.best_epoch_,
            best_val_macro_f1=clf.best_val_macro_f1_,
            training_time_seconds=clf.training_time_seconds_,
        )
        (out_dir / "run_manifest.json").write_text(
            run_manifest.model_dump_json(indent=2), encoding="utf-8"
        )

        # 6. Log to MLflow
        tracker.log_run(
            manifest=run_manifest,
            evaluation_summaries=eval_summaries,
            artifacts={"confusion_matrix_test.csv": out_dir / "confusion_matrix_test.csv"},
        )

        # 7. Compute Matched Graph Lift
        lift_record = compute_matched_graph_lift(
            gnn_summary=test_summary,
            mlp_summary=mlp_test_summary,
            gnn_manifest=run_manifest,
            mlp_manifest=mlp_manifest,
            graph_name=graph_name,
        )

        res_entry: dict[str, float | str | int] = {
            "graph_name": graph_name,
            "seed": seed,
            "test_macro_f1": test_summary.macro_f1,
            "mlp_macro_f1": mlp_test_summary.macro_f1,
            "matched_graph_lift": lift_record.overall_graph_lift,
            "test_balanced_acc": test_summary.balanced_accuracy,
            "best_epoch": clf.best_epoch_,
            "runtime_seconds": clf.training_time_seconds_,
        }
        sweep_results.append(res_entry)

        # One-line summary
        console.print(
            f"[bold green]✔ {run_tag}[/bold green] -> Test Macro-F1: [bold cyan]{test_summary.macro_f1:.4f}[/bold cyan] | MLP Ref: [bold magenta]{mlp_test_summary.macro_f1:.4f}[/bold magenta] | Matched Lift: [bold yellow]{lift_record.overall_graph_lift:+.4f}[/bold yellow] ({clf.training_time_seconds_:.1f}s, epoch {clf.best_epoch_})"
        )

    # Summary Table across entire sweep
    console.print("\n")
    table = Table(title=f"GraphSAGE Benchmark Sweep Summary ({dataset_name} - {graph_name})")
    table.add_column("Graph Variant", style="cyan")
    table.add_column("Seed", style="yellow")
    table.add_column("GraphSAGE Test F1", style="green")
    table.add_column("MLP Test F1", style="magenta")
    table.add_column("Matched Lift (Δ)", style="bold yellow")
    table.add_column("Balanced Acc", style="blue")
    table.add_column("Best Epoch", style="cyan")
    table.add_column("Runtime (s)", style="green")

    for r in sweep_results:
        table.add_row(
            str(r["graph_name"]),
            str(r["seed"]),
            f"{float(r['test_macro_f1']):.4f}",
            f"{float(r['mlp_macro_f1']):.4f}",
            f"[bold]{float(r['matched_graph_lift']):+.4f}[/bold]",
            f"{float(r['test_balanced_acc']):.4f}",
            str(r["best_epoch"]),
            f"{float(r['runtime_seconds']):.1f}s",
        )
    console.print(table)

    # Compute Aggregate Stats (Mean ± Std)
    if len(sweep_results) > 1:
        f1_vals = [float(r["test_macro_f1"]) for r in sweep_results]
        mlp_vals = [float(r["mlp_macro_f1"]) for r in sweep_results]
        lift_vals = [float(r["matched_graph_lift"]) for r in sweep_results]
        balacc_vals = [float(r["test_balanced_acc"]) for r in sweep_results]

        agg_table = Table(
            title=f"GraphSAGE {graph_name} Aggregate Statistics (N={len(sweep_results)})"
        )
        agg_table.add_column("Metric", style="cyan")
        agg_table.add_column("Mean ± Std", style="bold yellow")

        agg_table.add_row(
            "GraphSAGE Test Macro-F1", f"{np.mean(f1_vals):.4f} ± {np.std(f1_vals, ddof=1):.4f}"
        )
        agg_table.add_row(
            "Matched MLP Test Macro-F1", f"{np.mean(mlp_vals):.4f} ± {np.std(mlp_vals, ddof=1):.4f}"
        )
        agg_table.add_row(
            "Matched Graph Lift (Δ)",
            f"[bold]{np.mean(lift_vals):+.4f} ± {np.std(lift_vals, ddof=1):.4f}[/bold]",
        )
        agg_table.add_row(
            "GraphSAGE Balanced Accuracy",
            f"{np.mean(balacc_vals):.4f} ± {np.std(balacc_vals, ddof=1):.4f}",
        )
        console.print("\n")
        console.print(agg_table)

    return sweep_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run 5-seed GraphSAGE sweep on BBKNN graph.")
    parser.add_argument("--dataset", type=str, default="stephenson_2021_healthy_pbmc")
    parser.add_argument("--split", type=str, default="site_stratified_seed42")
    parser.add_argument(
        "--graph",
        type=str,
        default=DEFAULT_GRAPH,
        help="Graph artifact directory name.",
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=DEFAULT_SEEDS,
        help="Random seeds to evaluate (e.g. --seeds 7 17 42 73 101).",
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

    run_graphsage_bbknn_sweep(
        dataset_name=args.dataset,
        split_id=args.split,
        graph_name=args.graph,
        seeds=args.seeds,
        device=args.device,
        max_epochs=args.epochs,
        patience=args.patience,
    )
