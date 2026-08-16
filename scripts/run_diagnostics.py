"""CLI script to compute and persist graph diagnostics."""

from __future__ import annotations

import argparse

import numpy as np
from rich.console import Console
from rich.table import Table

from scgraph_bench.data.loaders import StephensonHealthyPBMCLoader
from scgraph_bench.diagnostics.runner import run_graph_diagnostics, save_diagnostics_report
from scgraph_bench.graph.schema import GraphBundle
from scgraph_bench.preprocessing.schema import PreprocessedBundle
from scgraph_bench.utils.paths import ArtifactPaths

console = Console()


def run_diagnostics_cli(
    dataset_name: str = "stephenson_2021_healthy_pbmc",
    split_id: str = "site_stratified_seed42",
    graph_name: str = "pca_knn_k20_unweighted",
) -> None:
    paths = ArtifactPaths.default()
    prep_dir = paths.artifacts_dir / "preprocessed" / dataset_name / split_id
    graph_dir = paths.artifacts_dir / "graphs" / dataset_name / split_id / graph_name

    console.print(f"[blue]Loading GraphBundle from:[/blue] {graph_dir}")
    graph_bundle = GraphBundle.load(graph_dir)

    console.print(f"[blue]Loading PreprocessedBundle from:[/blue] {prep_dir}")
    prep_bundle = PreprocessedBundle.load(prep_dir)

    # 1. Load full metadata from dataset
    loader = StephensonHealthyPBMCLoader()
    adata = loader.load()

    obs_idx_map = {str(cid): idx for idx, cid in enumerate(adata.obs_names)}
    node_indices = [obs_idx_map[cid] for cid in graph_bundle.node_cell_ids]

    donor_ids = adata.obs.iloc[node_indices]["donor_id"].astype(str).tolist()
    site_ids = adata.obs.iloc[node_indices]["site"].astype(str).tolist()

    # Reconcile integer labels in node order
    y_all = np.concatenate(
        [
            prep_bundle.train_labels,
            prep_bundle.val_labels,
            prep_bundle.test_labels,
        ]
    )
    inv_label_map = {v: k for k, v in prep_bundle.label_to_id.items()}
    label_names = [inv_label_map[i] for i in range(len(prep_bundle.label_to_id))]

    # 2. Execute diagnostics
    console.print(f"[bold green]Running graph diagnostics for '{graph_name}'...[/bold green]")
    report = run_graph_diagnostics(
        graph_bundle=graph_bundle,
        y_all=y_all,
        label_names=label_names,
        donor_ids=donor_ids,
        site_ids=site_ids,
    )

    out_dir = paths.artifacts_dir / "diagnostics" / dataset_name / split_id / graph_name
    save_diagnostics_report(report, out_dir)
    console.print(f"[blue]Saved diagnostics report to:[/blue] {out_dir}")

    # 3. Print Topology Table
    topo_table = Table(title=f"Graph Topology Diagnostics: {graph_name}")
    topo_table.add_column("Topology Metric", style="cyan")
    topo_table.add_column("Value", style="green")
    topo_table.add_row("Total Nodes", f"{report.topology.num_nodes:,}")
    topo_table.add_row("Total Edges", f"{report.topology.num_edges:,}")
    topo_table.add_row("Density", f"{report.topology.density:.6e}")
    topo_table.add_row(
        "In-Degree (Mean ± Std)",
        f"{report.topology.in_degree_mean:.2f} ± {report.topology.in_degree_std:.2f}",
    )
    topo_table.add_row(
        "In-Degree (Min / Median / Max)",
        f"{report.topology.in_degree_min} / {report.topology.in_degree_median:.1f} / {report.topology.in_degree_max}",
    )
    topo_table.add_row(
        "Out-Degree (Mean ± Std)",
        f"{report.topology.out_degree_mean:.2f} ± {report.topology.out_degree_std:.2f}",
    )
    topo_table.add_row(
        "Isolated Nodes",
        f"{report.topology.isolated_node_count} ({report.topology.isolated_node_fraction * 100:.2f}%)",
    )
    topo_table.add_row("Connected Components", f"{report.topology.num_connected_components}")
    topo_table.add_row(
        "Largest Component Fraction", f"{report.topology.largest_component_fraction * 100:.2f}%"
    )
    console.print(topo_table)

    # 4. Print Homophily Table
    if report.label_diagnostics is not None:
        ld = report.label_diagnostics
        homo_table = Table(title="Label Homophily & Neighborhood Purity")
        homo_table.add_column("Homophily Metric", style="cyan")
        homo_table.add_column("Score (0 to 1)", style="magenta")
        homo_table.add_row("Overall Edge Homophily", f"{ld.overall_edge_homophily:.4f}")
        homo_table.add_row("Overall Node Homophily", f"{ld.overall_node_homophily:.4f}")
        homo_table.add_row("Train -> Train Edge Homophily", f"{ld.train_train_edge_homophily:.4f}")
        homo_table.add_row("Val -> Train Query Homophily", f"{ld.val_to_train_query_homophily:.4f}")
        homo_table.add_row(
            "Test -> Train Query Homophily", f"{ld.test_to_train_query_homophily:.4f}"
        )
        homo_table.add_row("Macro-Average Class Purity", f"{ld.macro_average_class_purity:.4f}")
        console.print(homo_table)

    # 5. Print Metadata Mixing Table
    if report.metadata_diagnostics is not None:
        md = report.metadata_diagnostics
        mix_table = Table(title="Donor & Site Mixing Diagnostics")
        mix_table.add_column("Mixing Metric", style="cyan")
        mix_table.add_column("Value", style="yellow")
        mix_table.add_row(
            "Train Intra-Donor Edge Fraction", f"{md.train_intra_donor_edge_fraction:.4f}"
        )
        mix_table.add_row(
            "Train Intra-Site Edge Fraction", f"{md.train_intra_site_edge_fraction:.4f}"
        )
        mix_table.add_row(
            "Val -> Train Site Match Fraction", f"{md.val_to_train_site_match_fraction:.4f}"
        )
        mix_table.add_row(
            "Test -> Train Site Match Fraction", f"{md.test_to_train_site_match_fraction:.4f}"
        )
        mix_table.add_row("Mean Train Donor Entropy (bits)", f"{md.mean_train_donor_entropy:.4f}")
        mix_table.add_row("Mean Train Site Entropy (bits)", f"{md.mean_train_site_entropy:.4f}")
        console.print(mix_table)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run graph diagnostics suite.")
    parser.add_argument("--dataset", type=str, default="stephenson_2021_healthy_pbmc")
    parser.add_argument("--split", type=str, default="site_stratified_seed42")
    parser.add_argument("--graph", type=str, default="pca_knn_k20_unweighted")
    args = parser.parse_args()

    run_diagnostics_cli(
        dataset_name=args.dataset,
        split_id=args.split,
        graph_name=args.graph,
    )
