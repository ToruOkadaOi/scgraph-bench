"""CLI runner to construct and run diagnostics across all approved graph variants."""

from __future__ import annotations

import argparse

import numpy as np
from rich.console import Console
from rich.table import Table

from scgraph_bench.config.dataset import DatasetConfig
from scgraph_bench.config.graph import (
    BBKNNConfig,
    MutualkNNConfig,
    PCAkNNConfig,
    RewiredControlConfig,
)
from scgraph_bench.data.registry import get_dataset_loader
from scgraph_bench.diagnostics.runner import (
    run_graph_diagnostics,
    save_diagnostics_report,
)
from scgraph_bench.graph.bbknn import StrictInductiveBBKNNGraphBuilder
from scgraph_bench.graph.mutual_knn import MutualKNNGraphBuilder
from scgraph_bench.graph.pca_knn import PCAkNNGraphBuilder
from scgraph_bench.graph.rewired_control import RewiredControlGraphBuilder
from scgraph_bench.preprocessing.schema import PreprocessedBundle
from scgraph_bench.utils.paths import ArtifactPaths

console = Console()


def build_and_evaluate_graphs_cli(
    dataset_name: str = "stephenson_2021_healthy_pbmc",
    split_id: str = "site_stratified_seed42",
) -> None:
    paths = ArtifactPaths.default()
    prep_dir = paths.artifacts_dir / "preprocessed" / dataset_name / split_id

    if not (prep_dir / "feature_manifest.json").is_file():
        raise FileNotFoundError(f"Feature bundle missing at {prep_dir}")

    console.print(f"[blue]Loading preprocessed features from:[/blue] {prep_dir}")
    prep_bundle = PreprocessedBundle.load(prep_dir)

    # Load donor and site/batch metadata for BBKNN and diagnostics
    ds_config_path = (
        paths.configs_dir
        / "dataset"
        / f"{dataset_name.replace('stephenson_2021_healthy_pbmc', 'stephenson_healthy_pbmc')}.yaml"
    )
    if not ds_config_path.is_file():
        ds_config_path = paths.configs_dir / "dataset" / f"{dataset_name}.yaml"
    ds_config = DatasetConfig.from_yaml(ds_config_path)

    loader = get_dataset_loader(dataset_name)
    adata = loader.load(ds_config, primary_only=True)
    obs_map = {str(cid): idx for idx, cid in enumerate(adata.obs_names)}

    site_col = (
        "site"
        if "site" in adata.obs.columns
        else ("hpv_status" if "hpv_status" in adata.obs.columns else ds_config.batch_key)
    )

    train_donors = adata.obs.iloc[[obs_map[cid] for cid in prep_bundle.train_cell_ids]][
        ds_config.donor_key
    ].tolist()
    val_donors = adata.obs.iloc[[obs_map[cid] for cid in prep_bundle.val_cell_ids]][
        ds_config.donor_key
    ].tolist()
    test_donors = adata.obs.iloc[[obs_map[cid] for cid in prep_bundle.test_cell_ids]][
        ds_config.donor_key
    ].tolist()

    train_sites = adata.obs.iloc[[obs_map[cid] for cid in prep_bundle.train_cell_ids]][
        site_col
    ].tolist()
    val_sites = adata.obs.iloc[[obs_map[cid] for cid in prep_bundle.val_cell_ids]][
        site_col
    ].tolist()
    test_sites = adata.obs.iloc[[obs_map[cid] for cid in prep_bundle.test_cell_ids]][
        site_col
    ].tolist()

    all_donor_ids = train_donors + val_donors + test_donors
    all_site_ids = train_sites + val_sites + test_sites
    all_labels = np.concatenate(
        [prep_bundle.train_labels, prep_bundle.val_labels, prep_bundle.test_labels]
    )

    allowed_metadata = {
        "donor_ids_train": train_donors,
        "donor_ids_val": val_donors,
        "donor_ids_test": test_donors,
        "site_ids_train": train_sites,
        "site_ids_val": val_sites,
        "site_ids_test": test_sites,
    }

    inv_label_map = {v: k for k, v in prep_bundle.label_to_id.items()}
    label_names = [inv_label_map[i] for i in range(len(prep_bundle.label_to_id))]

    builders = [
        ("PCA-kNN (k=20, unweighted)", PCAkNNGraphBuilder(PCAkNNConfig(k=20))),
        ("Density-Matched PCA-kNN (k=24, unweighted)", PCAkNNGraphBuilder(PCAkNNConfig(k=24))),
        ("Mutual PCA-kNN (k=20, unweighted)", MutualKNNGraphBuilder(MutualkNNConfig(k=20))),
        (
            "Strict-Inductive BBKNN (k=2 per donor x 12 donors = 24)",
            StrictInductiveBBKNNGraphBuilder(BBKNNConfig(k_per_batch=2)),
        ),
        (
            "Rewired Control (PCA-kNN degree-matched)",
            RewiredControlGraphBuilder(RewiredControlConfig(seed=42, n_swaps_factor=10.0)),
        ),
    ]

    results_table = Table(title="Graph Variants Diagnostics Summary (Stephenson Healthy PBMC)")
    results_table.add_column("Graph Variant", style="cyan")
    results_table.add_column("Edges", style="green")
    results_table.add_column("Train Homophily", style="magenta")
    results_table.add_column("Val Query Homophily", style="blue")
    results_table.add_column("Test Query Homophily", style="yellow")
    results_table.add_column("Expected Random", style="cyan")
    results_table.add_column("Lift Over Random", style="green")
    results_table.add_column("Macro Class Purity", style="cyan")
    results_table.add_column("Train Intra-Donor %", style="green")

    for desc, builder in builders:
        console.print(f"\n[bold green]Building {desc}...[/bold green]")
        bundle = builder.build(
            X_pca_train=prep_bundle.X_pca_train,
            X_pca_val=prep_bundle.X_pca_val,
            X_pca_test=prep_bundle.X_pca_test,
            train_cell_ids=prep_bundle.train_cell_ids,
            val_cell_ids=prep_bundle.val_cell_ids,
            test_cell_ids=prep_bundle.test_cell_ids,
            feature_manifest_hash=prep_bundle.manifest.compute_manifest_hash(),
            dataset_name=dataset_name,
            split_id=split_id,
            allowed_metadata=allowed_metadata,
        )

        graph_dir = (
            paths.artifacts_dir / "graphs" / dataset_name / split_id / bundle.manifest.graph_name
        )
        bundle.save(graph_dir)
        console.print(f"Graph bundle saved to {graph_dir}")

        # Run Diagnostics
        diag_dir = (
            paths.artifacts_dir
            / "diagnostics"
            / dataset_name
            / split_id
            / bundle.manifest.graph_name
        )
        report = run_graph_diagnostics(
            graph_bundle=bundle,
            y_all=all_labels,
            donor_ids=all_donor_ids,
            site_ids=all_site_ids,
            label_names=label_names,
        )
        save_diagnostics_report(report, diag_dir)

        results_table.add_row(
            bundle.manifest.graph_name,
            f"{bundle.manifest.num_edges:,}",
            f"{report.label_diagnostics.train_train_edge_homophily:.4f}"
            if report.label_diagnostics
            else "N/A",
            f"{report.label_diagnostics.val_to_train_query_homophily:.4f}"
            if report.label_diagnostics
            else "N/A",
            f"{report.label_diagnostics.test_to_train_query_homophily:.4f}"
            if report.label_diagnostics
            else "N/A",
            f"{report.label_diagnostics.expected_random_homophily:.4f}"
            if report.label_diagnostics
            else "N/A",
            f"{report.label_diagnostics.homophily_lift_over_random:+.4f}"
            if report.label_diagnostics
            else "N/A",
            f"{report.label_diagnostics.macro_average_class_purity:.4f}"
            if report.label_diagnostics
            else "N/A",
            f"{report.metadata_diagnostics.train_intra_donor_edge_fraction * 100:.2f}%"
            if report.metadata_diagnostics
            else "N/A",
        )

    console.print("\n")
    console.print(results_table)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build and evaluate all graph variants.")
    parser.add_argument("--dataset", type=str, default="stephenson_2021_healthy_pbmc")
    parser.add_argument("--split", type=str, default="site_stratified_seed42")
    args = parser.parse_args()

    build_and_evaluate_graphs_cli(
        dataset_name=args.dataset,
        split_id=args.split,
    )
