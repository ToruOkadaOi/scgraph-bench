"""CLI script to compute matched graph lift over baseline MLP."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import pandas as pd
from rich.console import Console
from rich.table import Table

from scgraph_bench.evaluation.schema import EvaluationSummary
from scgraph_bench.tracking.graph_lift import compute_matched_graph_lift
from scgraph_bench.tracking.schema import RunManifest
from scgraph_bench.utils.paths import ArtifactPaths

console = Console()


def compute_graph_lift_cli(
    dataset_name: str = "stephenson_2021_healthy_pbmc",
    split_id: str = "site_stratified_seed42",
    models: list[str] | None = None,
    graphs: list[str] | None = None,
    seeds: list[int] | None = None,
) -> pd.DataFrame:
    """Compute matched graph lift for specified models, graphs, and seeds."""
    paths = ArtifactPaths.default()
    res_root = paths.artifacts_dir / "results" / dataset_name / split_id

    if not res_root.is_dir():
        console.print(f"[bold red]Results directory not found: {res_root}[/bold red]")
        return pd.DataFrame()

    console.print(f"[bold cyan]Scanning result directory:[/bold cyan] {res_root}\n")

    # 1. Discover all MLP baselines
    mlp_runs: dict[int, tuple[RunManifest, EvaluationSummary]] = {}
    for d in res_root.iterdir():
        if not d.is_dir():
            continue
        if d.name.startswith("mlp"):
            m_file = d / "run_manifest.json"
            s_file = d / "metrics_summary.json"
            if m_file.is_file() and s_file.is_file():
                manifest = RunManifest.model_validate_json(m_file.read_text(encoding="utf-8"))
                metrics = json.loads(s_file.read_text(encoding="utf-8"))
                if "test" in metrics:
                    summary = EvaluationSummary.model_validate(metrics["test"])
                    mlp_runs[manifest.seed] = (manifest, summary)
                    console.print(
                        f"[green]Found MLP Baseline (Seed {manifest.seed}):[/green] Test Macro-F1 = [bold]{summary.macro_f1:.4f}[/bold] in {d.name}"
                    )

    # 2. Discover matching GNN runs
    gnn_runs: list[tuple[str, str, int, Path, RunManifest, EvaluationSummary]] = []
    target_models = [m.lower() for m in models] if models else ["gcn", "graphsage"]
    target_graphs = set(graphs) if graphs else None
    target_seeds = set(seeds) if seeds else None

    for d in sorted(res_root.iterdir()):
        if not d.is_dir():
            continue

        match = re.match(r"(gcn|graphsage)_(.+)_seed(\d+)", d.name)
        if not match:
            continue

        m_name = match.group(1).lower()
        g_name = match.group(2)
        s_val = int(match.group(3))

        if m_name not in target_models:
            continue
        if target_graphs and g_name not in target_graphs:
            continue
        if target_seeds and s_val not in target_seeds:
            continue

        m_file = d / "run_manifest.json"
        s_file = d / "metrics_summary.json"
        if not (m_file.is_file() and s_file.is_file()):
            continue

        try:
            manifest = RunManifest.model_validate_json(m_file.read_text(encoding="utf-8"))
            metrics = json.loads(s_file.read_text(encoding="utf-8"))
            if "test" in metrics:
                summary = EvaluationSummary.model_validate(metrics["test"])
                gnn_runs.append((m_name, g_name, s_val, d, manifest, summary))
        except Exception as err:
            console.print(f"[yellow]Skipping {d.name}: {err}[/yellow]")

    console.print(f"\n[bold green]Found {len(gnn_runs)} matching GNN runs.[/bold green]\n")

    records = []
    for model_name, graph_name, seed, run_dir, gnn_manifest, gnn_summary in gnn_runs:
        mlp_match = mlp_runs.get(seed)

        lift_overall = None
        mlp_macro_f1 = None
        match_status = "unmatched"

        if mlp_match is not None:
            mlp_manifest, mlp_summary = mlp_match
            try:
                lift_rec = compute_matched_graph_lift(
                    gnn_summary=gnn_summary,
                    mlp_summary=mlp_summary,
                    gnn_manifest=gnn_manifest,
                    mlp_manifest=mlp_manifest,
                    graph_name=graph_name,
                )
                lift_overall = lift_rec.overall_graph_lift
                mlp_macro_f1 = mlp_summary.macro_f1
                match_status = "matched"
            except Exception as err:
                match_status = f"mismatch: {err}"
                console.print(f"[yellow]Match warning for {run_dir.name}: {err}[/yellow]")

        records.append(
            {
                "model": model_name,
                "graph": graph_name,
                "seed": seed,
                "gnn_macro_f1": gnn_summary.macro_f1,
                "mlp_macro_f1": mlp_macro_f1,
                "lift": lift_overall,
                "match_status": match_status,
                "overall_accuracy": gnn_summary.overall_accuracy,
                "balanced_accuracy": gnn_summary.balanced_accuracy,
            }
        )

    df_results = pd.DataFrame(records)

    table = Table(title=f"Matched Graph Lift ({dataset_name} - {split_id})")
    table.add_column("Model", style="bold cyan")
    table.add_column("Graph", style="cyan")
    table.add_column("Seed", style="yellow")
    table.add_column("GNN Macro-F1", style="green")
    table.add_column("MLP Macro-F1", style="magenta")
    table.add_column("Lift (Δ)", style="bold yellow")

    for _, row in df_results.iterrows():
        lift_str = f"{row['lift']:+.4f}" if pd.notna(row["lift"]) else "N/A"
        mlp_str = f"{row['mlp_macro_f1']:.4f}" if pd.notna(row["mlp_macro_f1"]) else "N/A"
        table.add_row(
            str(row["model"]).upper(),
            str(row["graph"]),
            str(row["seed"]),
            f"{row['gnn_macro_f1']:.4f}",
            mlp_str,
            f"[bold]{lift_str}[/bold]",
        )
    console.print(table)

    # Save to graph_lift_summary.csv
    out_csv = res_root / "graph_lift_summary.csv"
    df_results.to_csv(out_csv, index=False)
    console.print(f"\n[bold green]Saved summary to:[/bold green] {out_csv}")
    return df_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute matched graph lift over MLP.")
    parser.add_argument("--dataset", type=str, default="stephenson_2021_healthy_pbmc")
    parser.add_argument("--split", type=str, default="site_stratified_seed42")
    parser.add_argument("--models", nargs="+", default=["gcn"])
    parser.add_argument("--graphs", nargs="+", default=None)
    parser.add_argument("--seeds", nargs="+", type=int, default=[7, 17, 42, 73, 101])
    args = parser.parse_args()

    compute_graph_lift_cli(
        dataset_name=args.dataset,
        split_id=args.split,
        models=args.models,
        graphs=args.graphs,
        seeds=args.seeds,
    )
