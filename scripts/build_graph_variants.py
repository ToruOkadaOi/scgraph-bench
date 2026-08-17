"""CLI script to build and serialize specific graph variants on frozen preprocessed artifacts."""

from __future__ import annotations

import argparse

from rich.console import Console
from rich.table import Table

from scgraph_bench.config.graph import (
    EdgeWeightingMode,
    PCAkNNConfig,
    RewiredControlConfig,
)
from scgraph_bench.graph.pca_knn import PCAkNNGraphBuilder
from scgraph_bench.graph.rewired_control import RewiredControlGraphBuilder
from scgraph_bench.preprocessing.schema import PreprocessedBundle
from scgraph_bench.utils.paths import ArtifactPaths

console = Console()


def build_graph_variants(
    dataset_name: str = "stephenson_2021_healthy_pbmc",
    split_id: str = "site_stratified_seed42",
    graphs: list[str] | None = None,
) -> None:
    """Build and save requested graph variants."""
    paths = ArtifactPaths.default()
    prep_dir = paths.artifacts_dir / "preprocessed" / dataset_name / split_id

    if not (prep_dir / "feature_manifest.json").is_file():
        raise FileNotFoundError(
            f"Preprocessed bundle not found at {prep_dir}. Run scripts/run_preprocessing.py first."
        )

    console.print(f"[bold cyan]Loading preprocessed features from:[/bold cyan] {prep_dir}")
    bundle = PreprocessedBundle.load(prep_dir)
    manifest_hash = bundle.manifest.compute_manifest_hash()

    target_graphs = graphs or [
        "pca_knn_k10_unweighted",
        "pca_knn_k20_weighted",
        "pca_knn_k50_unweighted",
        "pca_knn_k20_unweighted_rewired",
    ]

    out_root = paths.artifacts_dir / "graphs" / dataset_name / split_id
    out_root.mkdir(parents=True, exist_ok=True)

    summary_table = Table(title=f"Built Graph Variants ({dataset_name} - {split_id})")
    summary_table.add_column("Graph Variant", style="cyan")
    summary_table.add_column("Builder Type", style="green")
    summary_table.add_column("Total Nodes", style="yellow")
    summary_table.add_column("Total Edges", style="magenta")
    summary_table.add_column("Train -> Train", style="blue")
    summary_table.add_column("Train -> Val", style="blue")
    summary_table.add_column("Train -> Test", style="blue")

    for g_name in target_graphs:
        console.print(f"\n[bold green]Constructing graph:[/bold green] [yellow]{g_name}[/yellow]")

        if g_name in ["pca_knn_k10_unweighted", "k10"]:
            cfg = PCAkNNConfig(k=10, weighting=EdgeWeightingMode.UNWEIGHTED)
            builder = PCAkNNGraphBuilder(cfg)
            graph_bundle = builder.build(
                X_pca_train=bundle.X_pca_train,
                X_pca_val=bundle.X_pca_val,
                X_pca_test=bundle.X_pca_test,
                train_cell_ids=bundle.train_cell_ids,
                val_cell_ids=bundle.val_cell_ids,
                test_cell_ids=bundle.test_cell_ids,
                feature_manifest_hash=manifest_hash,
                dataset_name=dataset_name,
                split_id=split_id,
            )
            canonical_name = "pca_knn_k10_unweighted"

        elif g_name in ["pca_knn_k20_weighted", "pca_knn_k20_rbf_weighted", "k20_weighted"]:
            cfg = PCAkNNConfig(k=20, weighting=EdgeWeightingMode.RBF_WEIGHTED)
            builder = PCAkNNGraphBuilder(cfg)
            graph_bundle = builder.build(
                X_pca_train=bundle.X_pca_train,
                X_pca_val=bundle.X_pca_val,
                X_pca_test=bundle.X_pca_test,
                train_cell_ids=bundle.train_cell_ids,
                val_cell_ids=bundle.val_cell_ids,
                test_cell_ids=bundle.test_cell_ids,
                feature_manifest_hash=manifest_hash,
                dataset_name=dataset_name,
                split_id=split_id,
            )
            canonical_name = "pca_knn_k20_weighted"

        elif g_name in ["pca_knn_k50_unweighted", "k50"]:
            cfg = PCAkNNConfig(k=50, weighting=EdgeWeightingMode.UNWEIGHTED)
            builder = PCAkNNGraphBuilder(cfg)
            graph_bundle = builder.build(
                X_pca_train=bundle.X_pca_train,
                X_pca_val=bundle.X_pca_val,
                X_pca_test=bundle.X_pca_test,
                train_cell_ids=bundle.train_cell_ids,
                val_cell_ids=bundle.val_cell_ids,
                test_cell_ids=bundle.test_cell_ids,
                feature_manifest_hash=manifest_hash,
                dataset_name=dataset_name,
                split_id=split_id,
            )
            canonical_name = "pca_knn_k50_unweighted"

        elif g_name in [
            "pca_knn_k20_unweighted_rewired",
            "rewired_control_pca_knn_seed42",
            "rewired",
        ]:
            cfg = RewiredControlConfig(seed=42, n_swaps_factor=10.0)
            builder = RewiredControlGraphBuilder(cfg)
            graph_bundle = builder.build(
                X_pca_train=bundle.X_pca_train,
                X_pca_val=bundle.X_pca_val,
                X_pca_test=bundle.X_pca_test,
                train_cell_ids=bundle.train_cell_ids,
                val_cell_ids=bundle.val_cell_ids,
                test_cell_ids=bundle.test_cell_ids,
                feature_manifest_hash=manifest_hash,
                dataset_name=dataset_name,
                split_id=split_id,
            )
            canonical_name = "pca_knn_k20_unweighted_rewired"

        else:
            raise ValueError(f"Unknown graph name requested: {g_name}")

        # Save to canonical directory
        out_dir = out_root / canonical_name
        graph_bundle.save(out_dir)
        console.print(f"[green]Saved GraphBundle to:[/green] {out_dir}")

        # Also create symlink / mirror alias if needed
        if canonical_name == "pca_knn_k20_unweighted_rewired":
            alias_dir = out_root / "rewired_control_pca_knn_seed42"
            if not alias_dir.exists():
                graph_bundle.save(alias_dir)
        elif canonical_name == "pca_knn_k20_weighted":
            alias_dir = out_root / "pca_knn_k20_rbf_weighted"
            if not alias_dir.exists():
                graph_bundle.save(alias_dir)

        summary_table.add_row(
            canonical_name,
            graph_bundle.manifest.builder_type,
            f"{graph_bundle.num_nodes:,}",
            f"{graph_bundle.manifest.num_edges:,}",
            f"{graph_bundle.manifest.num_train_train_edges:,}",
            f"{graph_bundle.manifest.num_train_to_val_edges:,}",
            f"{graph_bundle.manifest.num_train_to_test_edges:,}",
        )

    console.print("\n")
    console.print(summary_table)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build and serialize graph variants.")
    parser.add_argument("--dataset", type=str, default="stephenson_2021_healthy_pbmc")
    parser.add_argument("--split", type=str, default="site_stratified_seed42")
    parser.add_argument(
        "--graphs",
        nargs="+",
        default=[
            "pca_knn_k10_unweighted",
            "pca_knn_k20_weighted",
            "pca_knn_k50_unweighted",
            "pca_knn_k20_unweighted_rewired",
        ],
        help="List of graph variant names to construct.",
    )
    args = parser.parse_args()

    build_graph_variants(
        dataset_name=args.dataset,
        split_id=args.split,
        graphs=args.graphs,
    )
