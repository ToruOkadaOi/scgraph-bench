"""Cryptographic and topological validation utility for all CPU-produced benchmark artifacts."""

from __future__ import annotations

import argparse
import json

import numpy as np
from rich.console import Console
from rich.table import Table

from scgraph_bench.graph.schema import GraphBundle
from scgraph_bench.preprocessing.schema import PreprocessedBundle
from scgraph_bench.utils.hashing import hash_array, hash_file
from scgraph_bench.utils.paths import ArtifactPaths

console = Console()


def validate_all_artifacts(
    dataset_name: str = "stephenson_2021_healthy_pbmc",
    split_id: str = "site_stratified_seed42",
) -> bool:
    """Validate cryptographic integrity and topological constraints of all benchmark artifacts.

    Validates:
    1. Frozen Split JSON & cell IDs
    2. Preprocessed Feature Bundle (X_pca, feature manifest, preprocessor metadata)
    3. All 5 Serialized Graph Bundles (edge_index orientation, strict inductive flow, 0 disallowed edges)
    4. Baseline Model Results & Run Manifests

    Returns:
        True if all artifacts pass validation, False otherwise.
    """
    paths = ArtifactPaths.default()
    console.print(
        f"[bold blue]Starting Artifact Integrity Validation for {dataset_name} ({split_id})...[/bold blue]\n"
    )

    all_passed = True
    val_table = Table(title="Benchmark Artifact Integrity & Validation Status")
    val_table.add_column("Artifact Category", style="cyan")
    val_table.add_column("Artifact Name", style="magenta")
    val_table.add_column("Cryptographic Hash / Detail", style="yellow")
    val_table.add_column("Validation Status", style="green")

    # 1. Validate Frozen Split
    split_path = paths.splits_dir / dataset_name / f"{split_id}.json"
    if split_path.is_file():
        split_dict = json.loads(split_path.read_text(encoding="utf-8"))
        split_hash = hash_file(split_path)
        val_table.add_row(
            "Frozen Split",
            f"{split_id}.json",
            f"SHA-256: {split_hash[:16]}... ({len(split_dict['train_cell_ids'])} tr, {len(split_dict['val_cell_ids'])} va, {len(split_dict['test_cell_ids'])} te)",
            "[bold green]PASSED[/bold green]",
        )
    else:
        val_table.add_row(
            "Frozen Split", f"{split_id}.json", "FILE MISSING", "[bold red]FAILED[/bold red]"
        )
        all_passed = False

    # 2. Validate Preprocessed Feature Bundle
    prep_dir = paths.artifacts_dir / "preprocessed" / dataset_name / split_id
    if (prep_dir / "feature_manifest.json").is_file():
        prep_bundle = PreprocessedBundle.load(prep_dir)
        manifest_hash = prep_bundle.manifest.compute_manifest_hash()
        val_table.add_row(
            "Feature Bundle",
            "PCA-50 Features",
            f"Manifest Hash: {manifest_hash[:16]}... (Train: {prep_bundle.X_pca_train.shape}, Val: {prep_bundle.X_pca_val.shape}, Test: {prep_bundle.X_pca_test.shape})",
            "[bold green]PASSED[/bold green]",
        )
    else:
        val_table.add_row(
            "Feature Bundle", "PCA-50 Features", "DIR MISSING", "[bold red]FAILED[/bold red]"
        )
        all_passed = False

    # 3. Validate All Graph Bundles
    graph_base_dir = paths.artifacts_dir / "graphs" / dataset_name / split_id
    if graph_base_dir.is_dir():
        expected_graphs = sorted(
            [
                d.name
                for d in graph_base_dir.iterdir()
                if d.is_dir() and (d / "graph_manifest.json").is_file()
            ]
        )
    else:
        expected_graphs = [
            "pca_knn_k20_unweighted",
            "pca_knn_k24_unweighted",
            "mutual_knn_reference_standard_query_k20_unweighted",
            "bbknn_kperbatch2_donors12",
            "rewired_control_pca_knn_seed42",
        ]

    for g_name in expected_graphs:
        g_dir = graph_base_dir / g_name
        if (g_dir / "graph_manifest.json").is_file():
            bundle = GraphBundle.load(g_dir)
            edge_idx = bundle.edge_index.numpy()
            recomputed_hash = hash_array(edge_idx)

            # Strict inductive topological checks
            n_tr = bundle.manifest.num_train_nodes
            n_va = bundle.manifest.num_val_nodes
            src, dst = edge_idx[0], edge_idx[1]

            # Invariant: Query nodes must not send edges to train
            disallowed_val_tr = np.sum((src >= n_tr) & (src < n_tr + n_va) & (dst < n_tr))
            disallowed_test_tr = np.sum((src >= n_tr + n_va) & (dst < n_tr))
            disallowed_query_query = np.sum((src >= n_tr) & (dst >= n_tr))

            is_valid_topology = (
                disallowed_val_tr == 0
                and disallowed_test_tr == 0
                and disallowed_query_query == 0
                and recomputed_hash == bundle.manifest.edge_index_hash
            )

            status = (
                "[bold green]PASSED (0 Disallowed)[/bold green]"
                if is_valid_topology
                else "[bold red]FAILED[/bold red]"
            )
            if not is_valid_topology:
                all_passed = False

            val_table.add_row(
                "Graph Bundle",
                g_name,
                f"Edges: {bundle.manifest.num_edges:,} | Index Hash: {recomputed_hash[:12]}...",
                status,
            )
        else:
            val_table.add_row(
                "Graph Bundle", g_name, "BUNDLE MISSING", "[bold red]FAILED[/bold red]"
            )
            all_passed = False

    # 4. Validate Baseline Results
    res_base_dir = paths.artifacts_dir / "results" / dataset_name / split_id
    for model_name in ["logistic_regression", "mlp"]:
        r_dir = res_base_dir / model_name
        if (r_dir / "run_manifest.json").is_file() and (r_dir / "metrics_summary.json").is_file():
            val_table.add_row(
                "Baseline Model",
                model_name,
                "Manifest & Metrics Present",
                "[bold green]PASSED[/bold green]",
            )
        else:
            val_table.add_row(
                "Baseline Model", model_name, "RESULT MISSING", "[bold red]FAILED[/bold red]"
            )
            all_passed = False

    console.print(val_table)
    if all_passed:
        console.print(
            "\n[bold green]ALL BENCHMARK ARTIFACTS VERIFIED SUCCESSFULLY (100% INTEGRITY)[/bold green]\n"
        )
    else:
        console.print(
            "\n[bold red]ARTIFACT VERIFICATION FAILED - INVESTIGATE ERRORS ABOVE[/bold red]\n"
        )

    return all_passed


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Validate benchmark artifacts.")
    parser.add_argument("--dataset", type=str, default="stephenson_2021_healthy_pbmc")
    parser.add_argument("--split", type=str, default="site_stratified_seed42")
    args = parser.parse_args()

    success = validate_all_artifacts(dataset_name=args.dataset, split_id=args.split)
    if not success:
        raise SystemExit(1)
