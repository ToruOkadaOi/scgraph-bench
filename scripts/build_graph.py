"""CLI script to construct and serialize strict-inductive single-cell graphs."""

from __future__ import annotations

import argparse

from rich.console import Console
from rich.table import Table

from scgraph_bench.config.graph import EdgeWeightingMode, PCAkNNConfig
from scgraph_bench.graph.pca_knn import PCAkNNGraphBuilder
from scgraph_bench.preprocessing.schema import PreprocessedBundle
from scgraph_bench.utils.paths import ArtifactPaths

console = Console()


def build_and_save_graph(
    dataset_name: str = "stephenson_2021_healthy_pbmc",
    split_id: str = "site_stratified_seed42",
    k: int = 20,
    metric: str = "euclidean",
    rbf_weighting: bool = False,
    symmetrize: bool = True,
) -> None:
    paths = ArtifactPaths.default()
    prep_dir = paths.artifacts_dir / "preprocessed" / dataset_name / split_id

    if not (prep_dir / "feature_manifest.json").is_file():
        raise FileNotFoundError(
            f"Preprocessed bundle not found at {prep_dir}. Run scripts/run_preprocessing.py first."
        )

    console.print(f"[blue]Loading preprocessed features from:[/blue] {prep_dir}")
    bundle = PreprocessedBundle.load(prep_dir)

    weighting_mode = (
        EdgeWeightingMode.RBF_WEIGHTED if rbf_weighting else EdgeWeightingMode.UNWEIGHTED
    )
    config = PCAkNNConfig(
        k=k,
        metric=metric,
        symmetrize=symmetrize,
        weighting=weighting_mode,
    )

    builder = PCAkNNGraphBuilder(config)
    console.print(
        f"[bold green]Building strict-inductive PCA-kNN graph (k={k}, metric='{metric}', weighting='{weighting_mode.value}')...[/bold green]"
    )

    manifest_hash = bundle.manifest.compute_manifest_hash()
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

    out_dir = (
        paths.artifacts_dir / "graphs" / dataset_name / split_id / graph_bundle.manifest.graph_name
    )
    console.print(f"[blue]Saving GraphBundle to:[/blue] {out_dir}")
    graph_bundle.save(out_dir)

    # Print summary table
    table = Table(
        title=f"Graph Construction & Topology Summary: {graph_bundle.manifest.graph_name}"
    )
    table.add_column("Property", style="cyan")
    table.add_column("Value", style="green")
    table.add_column("Verification / Integrity", style="magenta")

    table.add_row("Graph Algorithm", graph_bundle.manifest.builder_type, "Strict Inductive")
    table.add_row(
        "Neighborhood k", str(graph_bundle.manifest.k), f"Metric: {graph_bundle.manifest.metric}"
    )
    table.add_row(
        "Edge Weighting",
        graph_bundle.manifest.weighting,
        f"sigma_k: {graph_bundle.manifest.sigma_k:.4f}"
        if graph_bundle.manifest.sigma_k
        else "Unweighted",
    )
    table.add_row(
        "Total Nodes",
        f"{graph_bundle.num_nodes:,}",
        f"Train: {graph_bundle.manifest.num_train_nodes:,} | Val: {graph_bundle.manifest.num_val_nodes:,} | Test: {graph_bundle.manifest.num_test_nodes:,}",
    )
    table.add_row(
        "Total Directed Edges",
        f"{graph_bundle.manifest.num_edges:,}",
        f"Edge Hash: {graph_bundle.manifest.edge_index_hash[:16]}...",
    )
    table.add_row(
        "Train -> Train Edges",
        f"{graph_bundle.manifest.num_train_train_edges:,}",
        "Internal reference connectivity",
    )
    table.add_row(
        "Train -> Val Edges",
        f"{graph_bundle.manifest.num_train_to_val_edges:,}",
        "Strict bipartite train -> val",
    )
    table.add_row(
        "Train -> Test Edges",
        f"{graph_bundle.manifest.num_train_to_test_edges:,}",
        "Strict bipartite train -> test",
    )
    table.add_row(
        "Disallowed Edges",
        f"{graph_bundle.manifest.num_disallowed_edges}",
        "0 Forbidden (val->train, test->train, val-val, test-test)",
    )

    console.print(table)
    console.print(
        f"[bold green]Phase 6 Graph Construction Completed Successfully![/bold green] Artifacts in {out_dir}\n"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Construct strict-inductive PCA-kNN graph.")
    parser.add_argument("--dataset", type=str, default="stephenson_2021_healthy_pbmc")
    parser.add_argument("--split", type=str, default="site_stratified_seed42")
    parser.add_argument("--k", type=int, default=20)
    parser.add_argument("--metric", type=str, default="euclidean")
    parser.add_argument("--rbf", action="store_true", help="Enable RBF kernel edge weighting.")
    args = parser.parse_args()

    build_and_save_graph(
        dataset_name=args.dataset,
        split_id=args.split,
        k=args.k,
        metric=args.metric,
        rbf_weighting=args.rbf,
    )
