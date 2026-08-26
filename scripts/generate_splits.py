"""Generate and save frozen donor-held-out splits."""

from __future__ import annotations

from rich.console import Console

from scgraph_bench.config.dataset import DatasetConfig
from scgraph_bench.config.split import SplitConfig
from scgraph_bench.data.registry import get_dataset_loader
from scgraph_bench.splitting.group_split import create_site_stratified_donor_split
from scgraph_bench.utils.paths import ArtifactPaths

console = Console()


def generate_frozen_split(
    dataset_name: str = "stephenson_2021_healthy_pbmc",
    split_id: str = "site_stratified_seed42",
    split_config_name: str | None = None,
    seed: int = 42,
) -> None:
    paths = ArtifactPaths.default()
    paths.ensure_directories()

    ds_config_file = paths.configs_dir / "dataset" / f"{dataset_name.replace('stephenson_2021_healthy_pbmc', 'stephenson_healthy_pbmc')}.yaml"
    if not ds_config_file.is_file():
        ds_config_file = paths.configs_dir / "dataset" / f"{dataset_name}.yaml"
    ds_config = DatasetConfig.from_yaml(ds_config_file)

    if split_config_name is None:
        if "hpv" in split_id:
            split_config_file = paths.configs_dir / "split" / "hpv_stratified_default.yaml"
        else:
            split_config_file = paths.configs_dir / "split" / "site_stratified_default.yaml"
    else:
        split_config_file = paths.configs_dir / "split" / f"{split_config_name}.yaml"

    sp_config = SplitConfig.from_yaml(split_config_file)

    console.print(f"[blue]Loading dataset:[/blue] {dataset_name} via production loader...")
    loader = get_dataset_loader(dataset_name)
    adata = loader.load(ds_config, primary_only=True)
    console.print(
        f"[green]Dataset loaded:[/green] shape={adata.shape}, donors={adata.obs['donor_id'].nunique()}"
    )

    strat_key = "hpv_status" if "hpv_status" in adata.obs.columns else ds_config.batch_key
    console.print(
        f"[blue]Generating stratified donor split:[/blue] {split_id} (key={strat_key}, seed={seed})..."
    )
    split_def = create_site_stratified_donor_split(
        adata=adata,
        donor_key=ds_config.donor_key,
        site_key=strat_key,
        label_key=ds_config.label_key,
        split_id=split_id,
        dataset_name=dataset_name,
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

    console.print("\n[bold]Stratification Composition:[/bold]")
    console.print(f"  Train: {split_def.site_composition.get('train')}")
    console.print(f"  Val:   {split_def.site_composition.get('val')}")
    console.print(f"  Test:  {split_def.site_composition.get('test')}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate frozen donor-held-out splits.")
    parser.add_argument("--dataset", type=str, default="stephenson_2021_healthy_pbmc")
    parser.add_argument("--split", type=str, default="site_stratified_seed42")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    generate_frozen_split(dataset_name=args.dataset, split_id=args.split, seed=args.seed)
