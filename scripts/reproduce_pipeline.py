"""Isolated full regeneration and cryptographic verification against canonical benchmark manifests."""

from __future__ import annotations

import argparse
import tempfile
from pathlib import Path

from rich.console import Console
from rich.table import Table

from scgraph_bench.graph.bbknn import StrictInductiveBBKNNGraphBuilder
from scgraph_bench.graph.mutual_knn import MutualKNNGraphBuilder
from scgraph_bench.graph.pca_knn import PCAkNNGraphBuilder
from scgraph_bench.graph.rewired_control import RewiredControlGraphBuilder
from scgraph_bench.graph.schema import (
    BBKNNConfig,
    GraphBundle,
    MutualkNNConfig,
    PCAkNNConfig,
    RewiredControlConfig,
)
from scgraph_bench.preprocessing.pipeline import LeakageSafePreprocessor
from scgraph_bench.preprocessing.schema import (
    PreprocessedBundle,
    PreprocessingConfig,
)
from scgraph_bench.splitting.schema import DatasetSplit
from scgraph_bench.utils.paths import ArtifactPaths

console = Console()


def reproduce_and_verify(
    dataset_name: str = "stephenson_2021_healthy_pbmc",
    split_id: str = "site_stratified_seed42",
) -> bool:
    """Regenerate features and graphs into an isolated directory and compare bit-for-bit with canonical manifests."""
    paths = ArtifactPaths.default()
    canonical_prep_dir = paths.artifacts_dir / "preprocessed" / dataset_name / split_id
    canonical_graphs_dir = paths.artifacts_dir / "graphs" / dataset_name / split_id

    # Check canonical existence
    if not (canonical_prep_dir / "feature_manifest.json").is_file():
        console.print(f"[bold red]Canonical features missing at {canonical_prep_dir}[/bold red]")
        return False

    canonical_prep = PreprocessedBundle.load(canonical_prep_dir)

    with tempfile.TemporaryDirectory(prefix="scgraph_reproduce_") as tmp_dir_str:
        tmp_dir = Path(tmp_dir_str)
        console.print(
            f"[bold blue]Executing isolated reproduction into temporary directory:[/bold blue] {tmp_dir}\n"
        )

        # 1. Load split and raw count dataset
        split_file = paths.splits_dir / dataset_name / f"{split_id}.json"
        split_obj = DatasetSplit.load(split_file)

        raw_h5ad = (
            paths.data_dir / "raw" / dataset_name / "stephenson_healthy_pbmc_unperturbed.h5ad"
        )
        if not raw_h5ad.is_file():
            # Fallback to TS or audited cache
            raw_h5ad = (
                paths.data_dir
                / "audits"
                / dataset_name
                / "stephenson_healthy_pbmc_unperturbed.h5ad"
            )

        import anndata as ad

        console.print(f"Loading raw AnnData source from {raw_h5ad}...")
        adata = ad.read_h5ad(raw_h5ad)

        # 2. Fit Preprocessing Pipeline in isolation
        console.print("Fitting Preprocessor on training cells in isolated directory...")
        preprocessor = LeakageSafePreprocessor(PreprocessingConfig())
        rep_prep_bundle, rep_prep_meta = preprocessor.fit_transform(
            adata=adata,
            split=split_obj,
            dataset_name=dataset_name,
            split_id=split_id,
        )

        rep_prep_bundle.save(tmp_dir / "preprocessed", metadata=rep_prep_meta)

        # 3. Compare Preprocessed Features
        results_table = Table(title="Isolated Reproduction vs Canonical Manifest Hash Comparison")
        results_table.add_column("Artifact Type", style="cyan")
        results_table.add_column("Artifact Name", style="magenta")
        results_table.add_column("Canonical Hash", style="yellow")
        results_table.add_column("Reproduction Hash", style="yellow")
        results_table.add_column("Match Status", style="green")

        all_matches = True

        feat_match = (
            rep_prep_bundle.manifest.features_train_hash
            == canonical_prep.manifest.features_train_hash
            and rep_prep_bundle.manifest.features_val_hash
            == canonical_prep.manifest.features_val_hash
            and rep_prep_bundle.manifest.features_test_hash
            == canonical_prep.manifest.features_test_hash
        )
        if not feat_match:
            all_matches = False

        results_table.add_row(
            "Feature Bundle",
            "PCA-50 Matrices (Train/Val/Test)",
            canonical_prep.manifest.features_train_hash[:16] + "...",
            rep_prep_bundle.manifest.features_train_hash[:16] + "...",
            "[bold green]IDENTICAL MATCH (100%)[/bold green]"
            if feat_match
            else "[bold red]MISMATCH[/bold red]",
        )

        # 4. Build Graphs in Isolation & Compare Hashes
        builders = [
            ("pca_knn_k20_unweighted", PCAkNNGraphBuilder(PCAkNNConfig(k=20))),
            ("pca_knn_k24_unweighted", PCAkNNGraphBuilder(PCAkNNConfig(k=24))),
            (
                "mutual_knn_reference_standard_query_k20_unweighted",
                MutualKNNGraphBuilder(MutualkNNConfig(k=20)),
            ),
            (
                "bbknn_kperbatch2_donors12",
                StrictInductiveBBKNNGraphBuilder(BBKNNConfig(k_per_batch=2)),
            ),
            (
                "rewired_control_pca_knn_seed42",
                RewiredControlGraphBuilder(RewiredControlConfig(seed=42, n_swaps_factor=10.0)),
            ),
        ]

        allowed_metadata = {
            "donor_ids_train": adata[split_obj.train_cell_ids].obs["donor_id"].values.tolist(),
            "donor_ids_val": adata[split_obj.val_cell_ids].obs["donor_id"].values.tolist(),
            "donor_ids_test": adata[split_obj.test_cell_ids].obs["donor_id"].values.tolist(),
        }

        for g_name, builder in builders:
            rep_bundle = builder.build(
                X_pca_train=rep_prep_bundle.X_pca_train,
                X_pca_val=rep_prep_bundle.X_pca_val,
                X_pca_test=rep_prep_bundle.X_pca_test,
                train_cell_ids=rep_prep_bundle.train_cell_ids,
                val_cell_ids=rep_prep_bundle.val_cell_ids,
                test_cell_ids=rep_prep_bundle.test_cell_ids,
                feature_manifest_hash=rep_prep_bundle.manifest.compute_manifest_hash(),
                dataset_name=dataset_name,
                split_id=split_id,
                allowed_metadata=allowed_metadata,
            )

            canonical_g_dir = canonical_graphs_dir / g_name
            if (canonical_g_dir / "graph_manifest.json").is_file():
                canonical_bundle = GraphBundle.load(canonical_g_dir)
                g_match = (
                    rep_bundle.manifest.edge_index_hash == canonical_bundle.manifest.edge_index_hash
                    and rep_bundle.manifest.num_edges == canonical_bundle.manifest.num_edges
                )
                if not g_match:
                    all_matches = False

                results_table.add_row(
                    "Graph Bundle",
                    g_name,
                    canonical_bundle.manifest.edge_index_hash[:16] + "...",
                    rep_bundle.manifest.edge_index_hash[:16] + "...",
                    "[bold green]IDENTICAL MATCH (100%)[/bold green]"
                    if g_match
                    else "[bold red]MISMATCH[/bold red]",
                )

        console.print(results_table)
        if all_matches:
            console.print(
                "\n[bold green]FULL REPRODUCTION VERIFICATION PASSED: All generated hashes identically match canonical benchmark artifacts.[/bold green]\n"
            )
        else:
            console.print(
                "\n[bold red]REPRODUCTION MISMATCH DETECTED: Generated hashes differ from canonical benchmark artifacts.[/bold red]\n"
            )

        return all_matches


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Full reproduction and manifest comparison.")
    parser.add_argument("--dataset", type=str, default="stephenson_2021_healthy_pbmc")
    parser.add_argument("--split", type=str, default="site_stratified_seed42")
    args = parser.parse_args()

    ok = reproduce_and_verify(dataset_name=args.dataset, split_id=args.split)
    if not ok:
        raise SystemExit(1)
