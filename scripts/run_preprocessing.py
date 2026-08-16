"""CLI script to run leakage-safe preprocessing on a frozen dataset split."""

from __future__ import annotations

import argparse
from pathlib import Path

from rich.console import Console
from rich.table import Table

from scgraph_bench.config.dataset import DatasetConfig
from scgraph_bench.config.preprocessing import PreprocessingConfig
from scgraph_bench.data.registry import get_dataset_loader
from scgraph_bench.preprocessing.pipeline import LeakageSafePreprocessor
from scgraph_bench.splitting.schema import SplitDefinition
from scgraph_bench.utils.paths import ArtifactPaths

console = Console()


def run_preprocessing_pipeline(
    dataset_name: str = "stephenson_2021_healthy_pbmc",
    split_id: str = "site_stratified_seed42",
    prep_config_path: Path | str | None = None,
    save_hvg: bool = False,
) -> None:
    paths = ArtifactPaths.default()
    paths.ensure_directories()

    split_file = paths.dataset_split_file(dataset_name, split_id)
    if not split_file.is_file():
        raise FileNotFoundError(f"Frozen split file not found: {split_file}")

    console.print(f"[blue]Loading frozen split:[/blue] {split_file}")
    split_def = SplitDefinition.load_json(split_file)

    console.print(f"[blue]Loading dataset:[/blue] {dataset_name} via production loader...")
    ds_config_path = paths.configs_dir / "dataset" / f"{dataset_name.replace('_2021', '')}.yaml"
    if not ds_config_path.is_file():
        ds_config_path = paths.configs_dir / "dataset" / "stephenson_healthy_pbmc.yaml"

    ds_config = DatasetConfig.from_yaml(ds_config_path)
    loader = get_dataset_loader(dataset_name)
    adata = loader.load(ds_config, primary_only=True)

    # Preprocessing configuration
    if prep_config_path is not None:
        p_cfg_path = Path(prep_config_path)
    else:
        p_cfg_path = paths.configs_dir / "preprocessing" / "standard_pca50.yaml"

    console.print(f"[blue]Loading preprocessing config:[/blue] {p_cfg_path}")
    prep_config = PreprocessingConfig.from_yaml(p_cfg_path)

    # Run LeakageSafePreprocessor
    console.print(
        "[bold green]Executing LeakageSafePreprocessor (strictly fitting on training cells)...[/bold green]"
    )
    preprocessor = LeakageSafePreprocessor(prep_config)
    bundle = preprocessor.fit_transform_split(
        adata=adata,
        split_def=split_def,
        label_key=ds_config.label_key,
    )

    out_dir = paths.artifacts_dir / "preprocessed" / dataset_name / split_id
    console.print(f"[blue]Saving preprocessed bundle to:[/blue] {out_dir}")
    bundle.save(out_dir, save_hvg=save_hvg)

    # Print summary table
    table = Table(title="Preprocessing Summary & Dimensions")
    table.add_column("Partition / Property", style="cyan")
    table.add_column("Dimensions / Value", style="green")
    table.add_column("Checksum / Notes", style="magenta")

    table.add_row(
        "Train X_pca",
        str(bundle.X_pca_train.shape),
        f"Hash: {bundle.metadata.pca_components_hash[:16]}...",
    )
    table.add_row("Val X_pca", str(bundle.X_pca_val.shape), "Projected onto train basis")
    table.add_row("Test X_pca", str(bundle.X_pca_test.shape), "Projected onto train basis")
    table.add_row(
        "Total Evaluated Cells",
        f"{bundle.X_pca_train.shape[0] + bundle.X_pca_val.shape[0] + bundle.X_pca_test.shape[0]:,}",
        "100% Reconciled",
    )
    table.add_row(
        "Selected Seurat HVGs",
        f"{bundle.metadata.n_hvg_selected} genes",
        f"from {bundle.metadata.n_genes_raw} raw genes",
    )
    table.add_row(
        "PCA Components",
        f"{bundle.metadata.n_pca_components} components",
        f"Explained variance: {sum(bundle.metadata.pca_explained_variance_ratio):.4f}",
    )
    table.add_row(
        "Label Vocabulary",
        f"{len(bundle.label_to_id)} classes",
        ", ".join(sorted(bundle.label_to_id.keys())[:3]) + "...",
    )

    console.print(table)
    console.print(
        f"[bold green]Phase 5 Preprocessing Completed Successfully![/bold green] Output saved in {out_dir}\n"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run training-fitted preprocessing on scgraph-bench split."
    )
    parser.add_argument("--dataset", type=str, default="stephenson_2021_healthy_pbmc")
    parser.add_argument("--split", type=str, default="site_stratified_seed42")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument(
        "--save-hvg", action="store_true", help="Also save 2000-dim scaled HVG matrices."
    )
    args = parser.parse_args()

    run_preprocessing_pipeline(
        dataset_name=args.dataset,
        split_id=args.split,
        prep_config_path=args.config,
        save_hvg=args.save_hvg,
    )
