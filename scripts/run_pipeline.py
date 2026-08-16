"""End-to-End CPU Pipeline Runner strictly consuming verified precomputed artifacts without overwriting."""

from __future__ import annotations

import argparse
import subprocess
import sys

from rich.console import Console
from scripts.validate_artifacts import validate_all_artifacts

console = Console()


def run_pipeline(
    dataset: str = "stephenson_2021_healthy_pbmc",
    split: str = "site_stratified_seed42",
) -> None:
    """Consume existing verified artifacts, run baselines and aggregation, ensuring zero overwriting of precomputed graphs/features."""
    console.print(
        "[bold cyan]====================================================================[/bold cyan]"
    )
    console.print(
        f"[bold cyan]  Executing scGraph-Bench Safe CPU Pipeline: {dataset} ({split})[/bold cyan]"
    )
    console.print(
        "[bold cyan]====================================================================[/bold cyan]\n"
    )

    # 1. Pre-validation: Fail fast if canonical artifacts are missing or corrupt
    console.print(
        "[bold yellow]>>> Step 1: Verifying Canonical Precomputed Artifacts...[/bold yellow]"
    )
    is_valid = validate_all_artifacts(dataset_name=dataset, split_id=split)
    if not is_valid:
        console.print(
            "[bold red]Canonical artifact validation failed! Halting pipeline.[/bold red]"
        )
        sys.exit(1)

    # 2. Execute baselines and results aggregation strictly using existing preprocessed features
    steps = [
        (
            "Step 2: Train & Evaluate Logistic Regression Baseline",
            [
                sys.executable,
                "scripts/train_logistic_regression.py",
                "--dataset",
                dataset,
                "--split",
                split,
            ],
        ),
        (
            "Step 3: Train & Evaluate PyTorch MLP Baseline (Seed 42)",
            [
                sys.executable,
                "scripts/train_mlp.py",
                "--dataset",
                dataset,
                "--split",
                split,
                "--seed",
                "42",
            ],
        ),
        (
            "Step 4: Aggregate Benchmark Results & Verify Manifests",
            [
                sys.executable,
                "scripts/aggregate_results.py",
                "--dataset",
                dataset,
                "--split",
                split,
            ],
        ),
    ]

    for desc, cmd in steps:
        console.print(f"\n[bold yellow]>>> {desc}...[/bold yellow]")
        res = subprocess.run(cmd, check=False)
        if res.returncode != 0:
            console.print(f"[bold red]Pipeline failed at: {desc}[/bold red]")
            sys.exit(res.returncode)

    # 3. Post-validation: Re-verify canonical artifacts were untouched
    console.print(
        "\n[bold yellow]>>> Step 5: Post-execution Invariant & Artifact Check...[/bold yellow]"
    )
    is_valid_after = validate_all_artifacts(dataset_name=dataset, split_id=split)
    if not is_valid_after:
        console.print("[bold red]Post-execution artifact integrity violation![/bold red]")
        sys.exit(1)

    console.print(
        "\n[bold green]====================================================================[/bold green]"
    )
    console.print(
        "[bold green]  Safe Pipeline Run Completed Successfully (Artifacts Preserved).   [/bold green]"
    )
    console.print(
        "[bold green]====================================================================[/bold green]\n"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run safe scGraph-Bench CPU pipeline.")
    parser.add_argument("--dataset", type=str, default="stephenson_2021_healthy_pbmc")
    parser.add_argument("--split", type=str, default="site_stratified_seed42")
    args = parser.parse_args()

    run_pipeline(dataset=args.dataset, split=args.split)
