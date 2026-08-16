"""Generate and save frozen donor-held-out splits."""

from __future__ import annotations

from rich.console import Console

from scgraph_bench.config.dataset import DatasetConfig
from scgraph_bench.config.split import SplitConfig
from scgraph_bench.data.registry import get_dataset_loader
from scgraph_bench.splitting.group_split import create_site_stratified_donor_split
from scgraph_bench.utils.paths import ArtifactPaths

console = Console()


def generate_stephenson_frozen_split(
    dataset_name: str = "stephenson_2021_healthy_pbmc",
    split_id: str = "site_stratified_seed42",
    seed: int = 42,
) -> None:
    paths = ArtifactPaths.default()
    paths.ensure_directories()

    ds_config = DatasetConfig.from_yaml(
        paths.configs_dir / "dataset" / "stephenson_healthy_pbmc.yaml"
    )
    sp_config = SplitConfig.from_yaml(paths.configs_dir / "split" / "site_stratified_default.yaml")

    console.print(f"[blue]Loading dataset:[/blue] {dataset_name} via production loader...")
    loader = get_dataset_loader(dataset_name)
    adata = loader.load(ds_config, primary_only=True)
    console.print(
        f"[green]Dataset loaded:[/green] shape={adata.shape}, donors={adata.obs['donor_id'].nunique()}"
    )

    console.print(
        f"[blue]Generating site-stratified donor split:[/blue] {split_id} (seed={seed})..."
    )
    split_def = create_site_stratified_donor_split(
        adata=adata,
        donor_key=ds_config.donor_key,
        site_key=ds_config.batch_key,
        label_key=ds_config.label_key,
        split_id=split_id,
        config=sp_config,
        seed=seed,
    )

    out_file = paths.dataset_split_file(dataset_name, split_id)
    split_def.save_json(out_file)
    console.print(f"[bold green]Frozen split saved successfully to:[/bold green] {out_file}")

    console.print("\n[bold]Split Partition Summary:[/bold]")
    console.print(
        f"  Train Donors ({len(split_def.train_donors)}): {', '.join(split_def.train_donors)} -> {len(split_def.train_cell_ids):,} cells"
    )
    console.print(
        f"  Val Donors   ({len(split_def.val_donors)}): {', '.join(split_def.val_donors)} -> {len(split_def.val_cell_ids):,} cells"
    )
    console.print(
        f"  Test Donors  ({len(split_def.test_donors)}): {', '.join(split_def.test_donors)} -> {len(split_def.test_cell_ids):,} cells"
    )

    console.print("\n[bold]Site Composition:[/bold]")
    console.print(f"  Train: {split_def.site_composition.get('train')}")
    console.print(f"  Val:   {split_def.site_composition.get('val')}")
    console.print(f"  Test:  {split_def.site_composition.get('test')}")


if __name__ == "__main__":
    generate_stephenson_frozen_split()
