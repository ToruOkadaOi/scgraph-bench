"""Offline Matched Graph Lift Aggregator.

Scans artifacts/results/<dataset>/<split>/ for all GNN and MLP runs,
performs cryptographic and seed-matched alignment, computes overall & stratified graph lifts,
and exports comprehensive summary tables.
"""

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


def compute_offline_graph_lift(
    dataset_name: str = "stephenson_2021_healthy_pbmc",
    split_id: str = "site_stratified_seed42",
) -> pd.DataFrame:
    """Aggregate all GNN runs and compute matched graph lifts against identically-seeded MLP baselines."""
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

    if not mlp_runs:
        console.print("[bold yellow]Warning: No MLP baselines found on disk.[/bold yellow]")

    # 2. Discover all GCN runs
    gnn_runs: list[tuple[str, int, Path, RunManifest, EvaluationSummary]] = []
    for d in sorted(res_root.iterdir()):
        if not d.is_dir() or not d.name.startswith("gcn_"):
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

                # Extract graph name from run directory or manifest
                match = re.match(r"gcn_(.+)_seed(\d+)", d.name)
                graph_name = (
                    match.group(1) if match else (manifest.graph_artifact_hash or "unknown")
                )
                gnn_runs.append((graph_name, manifest.seed, d, manifest, summary))
        except Exception as err:
            console.print(f"[yellow]Skipping {d.name}: {err}[/yellow]")

    console.print(
        f"\n[bold green]Found {len(gnn_runs)} GCN runs across {len({r[0] for r in gnn_runs})} graph variants.[/bold green]\n"
    )

    # 3. Compute Matched Graph Lift for each GNN run
    records = []
    for graph_name, seed, run_dir, gnn_manifest, gnn_summary in gnn_runs:
        mlp_match = mlp_runs.get(seed)

        lift_overall = None
        lift_balacc = None
        mlp_macro_f1 = None
        mlp_balacc = None
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
                lift_balacc = lift_rec.balanced_accuracy_lift
                mlp_macro_f1 = mlp_summary.macro_f1
                mlp_balacc = mlp_summary.balanced_accuracy
                match_status = "matched"
            except Exception as err:
                match_status = f"mismatch: {err}"
                console.print(f"[yellow]Match warning for {run_dir.name}: {err}[/yellow]")

        camb_obs_f1 = (
            gnn_summary.per_site[0].observed_class_macro_f1 if gnn_summary.per_site else None
        )
        newc_obs_f1 = (
            gnn_summary.per_site[1].observed_class_macro_f1
            if len(gnn_summary.per_site) > 1
            else None
        )

        records.append(
            {
                "graph_name": graph_name,
                "seed": seed,
                "model_name": "gcn",
                "match_status": match_status,
                "gnn_test_macro_f1": gnn_summary.macro_f1,
                "mlp_test_macro_f1": mlp_macro_f1,
                "matched_graph_lift": lift_overall,
                "gnn_test_balacc": gnn_summary.balanced_accuracy,
                "mlp_test_balacc": mlp_balacc,
                "balacc_lift": lift_balacc,
                "cambridge_obs_f1": camb_obs_f1,
                "newcastle_obs_f1": newc_obs_f1,
                "best_epoch": gnn_manifest.best_epoch,
                "training_time_seconds": gnn_manifest.training_time_seconds,
                "run_dir": str(run_dir.name),
            }
        )

    df_results = pd.DataFrame(records)

    # 4. Display Formatted Summary Table
    table = Table(title=f"Offline Matched Graph Lift Summary ({dataset_name} - {split_id})")
    table.add_column("Graph Variant", style="cyan")
    table.add_column("Seed", style="yellow")
    table.add_column("GCN Test F1", style="green")
    table.add_column("MLP Ref F1", style="magenta")
    table.add_column("Matched Lift (Δ)", style="yellow")
    table.add_column("GCN BalAcc", style="blue")
    table.add_column("Best Epoch", style="cyan")
    table.add_column("Match Status", style="green")

    for _, row in df_results.iterrows():
        lift_str = (
            f"{row['matched_graph_lift']:+.4f}" if pd.notna(row["matched_graph_lift"]) else "N/A"
        )
        mlp_str = f"{row['mlp_test_macro_f1']:.4f}" if pd.notna(row["mlp_test_macro_f1"]) else "N/A"
        table.add_row(
            str(row["graph_name"]),
            str(row["seed"]),
            f"{row['gnn_test_macro_f1']:.4f}",
            mlp_str,
            f"[bold]{lift_str}[/bold]",
            f"{row['gnn_test_balacc']:.4f}",
            str(row["best_epoch"]),
            str(row["match_status"]),
        )
    console.print(table)

    # 5. Compute mean and standard deviation per graph variant if multiple seeds exist
    if not df_results.empty and "matched_graph_lift" in df_results.columns:
        matched_df = df_results[df_results["matched_graph_lift"].notna()]
        if not matched_df.empty:
            agg_table = Table(title="Aggregated Graph Lift by Graph Topology (Mean ± Std)")
            agg_table.add_column("Graph Variant", style="cyan")
            agg_table.add_column("N Seeds", style="yellow")
            agg_table.add_column("GCN Test Macro-F1", style="green")
            agg_table.add_column("Matched Graph Lift (Δ)", style="bold yellow")
            agg_table.add_column("GCN Balanced Acc", style="blue")

            for g_name, grp in matched_df.groupby("graph_name"):
                n_s = len(grp)
                gcn_f1_mean = grp["gnn_test_macro_f1"].mean()
                gcn_f1_std = grp["gnn_test_macro_f1"].std() if n_s > 1 else 0.0
                lift_mean = grp["matched_graph_lift"].mean()
                lift_std = grp["matched_graph_lift"].std() if n_s > 1 else 0.0
                balacc_mean = grp["gnn_test_balacc"].mean()
                balacc_std = grp["gnn_test_balacc"].std() if n_s > 1 else 0.0

                agg_table.add_row(
                    str(g_name),
                    str(n_s),
                    f"{gcn_f1_mean:.4f} ± {gcn_f1_std:.4f}",
                    f"[bold]{lift_mean:+.4f} ± {lift_std:.4f}[/bold]",
                    f"{balacc_mean:.4f} ± {balacc_std:.4f}",
                )
            console.print("\n")
            console.print(agg_table)

    # 6. Save summary files to disk
    out_csv = res_root / "graph_lift_summary.csv"
    out_json = res_root / "graph_lift_summary.json"
    df_results.to_csv(out_csv, index=False)
    out_json.write_text(df_results.to_json(orient="records", indent=2), encoding="utf-8")
    console.print(f"\n[bold green]Saved summary table to:[/bold green] {out_csv}")

    return df_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Offline Matched Graph Lift Aggregator.")
    parser.add_argument("--dataset", type=str, default="stephenson_2021_healthy_pbmc")
    parser.add_argument("--split", type=str, default="site_stratified_seed42")
    args = parser.parse_args()

    compute_offline_graph_lift(dataset_name=args.dataset, split_id=args.split)
