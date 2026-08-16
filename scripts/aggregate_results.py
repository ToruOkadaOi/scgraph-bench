"""CLI runner to aggregate all benchmark results into tidy tables and MLflow format."""

from __future__ import annotations

import argparse

from rich.console import Console
from rich.table import Table

from scgraph_bench.tracking.aggregator import ResultsAggregator
from scgraph_bench.utils.paths import ArtifactPaths

console = Console()


def aggregate_results_cli(
    dataset_name: str = "stephenson_2021_healthy_pbmc",
    split_id: str = "site_stratified_seed42",
) -> None:
    paths = ArtifactPaths.default()
    aggregator = ResultsAggregator()

    console.print(
        f"[bold blue]Aggregating benchmark results for {dataset_name} ({split_id})...[/bold blue]"
    )
    collection = aggregator.aggregate_all(dataset_name=dataset_name, split_id=split_id)

    out_dir = paths.artifacts_dir / "aggregated" / dataset_name / split_id
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Export tidy metrics dataframe
    df_metrics = collection.to_dataframe()
    csv_path = out_dir / "benchmark_results_tidy.csv"
    df_metrics.to_csv(csv_path, index=False)
    console.print(f"[green]Tidy metrics ({len(df_metrics)} records) saved to:[/green] {csv_path}")

    # 2. Export matched lifts if present
    df_lifts = collection.lifts_to_dataframe()
    if not df_lifts.empty:
        lift_csv_path = out_dir / "benchmark_graph_lifts.csv"
        df_lifts.to_csv(lift_csv_path, index=False)
        console.print(
            f"[green]Matched graph lifts ({len(df_lifts)} records) saved to:[/green] {lift_csv_path}"
        )

    # 3. Print summary table of completed runs
    runs_table = Table(title=f"Benchmark Evaluated Models ({dataset_name} - {split_id})")
    runs_table.add_column("Run ID", style="cyan")
    runs_table.add_column("Model", style="magenta")
    runs_table.add_column("Seed", style="yellow")
    runs_table.add_column("Test Macro-F1", style="green")
    runs_table.add_column("Test BalAcc", style="blue")
    runs_table.add_column("Cambridge F1", style="cyan")
    runs_table.add_column("Newcastle F1", style="cyan")

    # Filter to test partition primary metrics
    if not df_metrics.empty:
        unique_runs = df_metrics["run_id"].unique()
        for r_id in unique_runs:
            df_r = df_metrics[df_metrics["run_id"] == r_id]
            m_name = df_r["model_name"].iloc[0]
            seed = df_r["seed"].iloc[0]

            test_f1_row = df_r[(df_r["partition"] == "test") & (df_r["metric_name"] == "macro_f1")]
            test_f1 = (
                f"{test_f1_row['metric_value'].iloc[0]:.4f}" if not test_f1_row.empty else "N/A"
            )

            test_ba_row = df_r[
                (df_r["partition"] == "test") & (df_r["metric_name"] == "balanced_accuracy")
            ]
            test_ba = (
                f"{test_ba_row['metric_value'].iloc[0]:.4f}" if not test_ba_row.empty else "N/A"
            )

            cam_f1_row = df_r[
                (df_r["partition"] == "test")
                & (df_r["metric_name"] == "site_observed_macro_f1")
                & (df_r["site"] == "Cambridge")
            ]
            cam_f1 = f"{cam_f1_row['metric_value'].iloc[0]:.4f}" if not cam_f1_row.empty else "N/A"

            newc_f1_row = df_r[
                (df_r["partition"] == "test")
                & (df_r["metric_name"] == "site_observed_macro_f1")
                & (df_r["site"] == "Newcastle")
            ]
            newc_f1 = (
                f"{newc_f1_row['metric_value'].iloc[0]:.4f}" if not newc_f1_row.empty else "N/A"
            )

            runs_table.add_row(r_id, m_name, str(seed), test_f1, test_ba, cam_f1, newc_f1)

    console.print("\n")
    console.print(runs_table)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Aggregate benchmark results.")
    parser.add_argument("--dataset", type=str, default="stephenson_2021_healthy_pbmc")
    parser.add_argument("--split", type=str, default="site_stratified_seed42")
    args = parser.parse_args()

    aggregate_results_cli(
        dataset_name=args.dataset,
        split_id=args.split,
    )
