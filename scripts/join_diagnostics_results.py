"""Join graph diagnostics with benchmark outcomes to explain when graphs help or hurt.

Merges GraphDiagnosticsReport metrics (homophily, purity, mixing entropy, topology)
with per-graph matched lift statistics computed from run artifacts, then exports a
joined table plus correlation analysis.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from rich.console import Console
from rich.table import Table

from scgraph_bench.analysis.flatten import describe_run, discover_run_records
from scgraph_bench.diagnostics.schema import GraphDiagnosticsReport
from scgraph_bench.evaluation.schema import EvaluationSummary
from scgraph_bench.tracking.graph_lift import compute_matched_graph_lift
from scgraph_bench.tracking.schema import RunManifest
from scgraph_bench.utils.paths import ArtifactPaths

console = Console()

GRAPH_ALIASES = {
    "pca_knn_k20_unweighted_rewired": "rewired_control_pca_knn_seed42",
    "pca_knn_k20_weighted": "pca_knn_k20_rbf_weighted",
}

DIAGNOSTIC_METRICS = [
    ("num_edges", "topology.num_edges"),
    ("density", "topology.density"),
    ("isolated_node_fraction", "topology.isolated_node_fraction"),
    ("overall_edge_homophily", "label.overall_edge_homophily"),
    ("train_train_edge_homophily", "label.train_train_edge_homophily"),
    ("test_to_train_query_homophily", "label.test_to_train_query_homophily"),
    ("expected_random_homophily", "label.expected_random_homophily"),
    ("macro_average_class_purity", "label.macro_average_class_purity"),
    ("train_intra_donor_edge_fraction", "metadata.train_intra_donor_edge_fraction"),
    ("test_to_train_site_match_fraction", "metadata.test_to_train_site_match_fraction"),
    ("mean_train_donor_entropy", "metadata.mean_train_donor_entropy"),
    ("mean_test_query_donor_entropy", "metadata.mean_test_query_donor_entropy"),
]

CORRELATION_FEATURES = [
    "overall_edge_homophily",
    "train_train_edge_homophily",
    "test_to_train_query_homophily",
    "macro_average_class_purity",
    "train_intra_site_edge_fraction",
    "mean_train_donor_entropy",
    "log_num_edges",
]


def canonical_graph_name(name: str) -> str:
    return GRAPH_ALIASES.get(name, name)


def load_diagnostics(diag_root: Path) -> dict[str, GraphDiagnosticsReport]:
    reports: dict[str, GraphDiagnosticsReport] = {}
    if not diag_root.is_dir():
        console.print(f"[yellow]No diagnostics directory at {diag_root}[/yellow]")
        return reports
    for d in sorted(p for p in diag_root.iterdir() if p.is_dir()):
        report_path = d / "graph_diagnostics.json"
        if not report_path.is_file():
            continue
        try:
            report = GraphDiagnosticsReport.model_validate_json(
                report_path.read_text(encoding="utf-8")
            )
            reports[report.graph_name] = report
        except Exception as err:
            console.print(f"[yellow]Skipping unreadable diagnostics {d.name}: {err}[/yellow]")
    return reports


def load_lift_stats(res_root: Path) -> pd.DataFrame | None:
    """Prefer the offline lift summary; otherwise compute lifts from raw run artifacts."""
    summary_csv = res_root / "graph_lift_summary.csv"
    if summary_csv.is_file():
        df = pd.read_csv(summary_csv)
        needed = {"model_name", "graph_name", "seed", "matched_graph_lift"}
        if needed.issubset(df.columns):
            console.print(f"[cyan]Using existing lift summary:[/cyan] {summary_csv.name}")
            return df
        console.print("[yellow]Existing lift summary lacks required columns; recomputing.[/yellow]")

    records = discover_run_records(res_root)
    mlp_by_seed: dict[int, tuple[RunManifest, EvaluationSummary]] = {}
    for rec in records:
        if rec.manifest.model_name == "mlp":
            summary = rec.summaries.get("test")
            if summary is not None:
                mlp_by_seed[rec.manifest.seed] = (rec.manifest, summary)

    rows: list[dict[str, object]] = []
    for rec in records:
        if rec.manifest.model_name not in {"gcn", "graphsage"}:
            continue
        summary = rec.summaries.get("test")
        if summary is None:
            continue
        _, graph_name = describe_run(rec.run_id, rec.manifest.model_name)
        match = mlp_by_seed.get(rec.manifest.seed)
        lift = np.nan
        if match is not None:
            try:
                record = compute_matched_graph_lift(
                    gnn_summary=summary,
                    mlp_summary=match[1],
                    gnn_manifest=rec.manifest,
                    mlp_manifest=match[0],
                    graph_name=graph_name,
                )
                lift = record.overall_graph_lift
            except Exception as err:
                console.print(f"[yellow]Lift skipped for {rec.run_id}: {err}[/yellow]")
        rows.append(
            {
                "model_name": rec.manifest.model_name,
                "graph_name": graph_name,
                "seed": rec.manifest.seed,
                "matched_graph_lift": lift,
            }
        )
    df = pd.DataFrame(rows)
    if df.empty:
        return None
    console.print("[cyan]Computed lifts directly from run artifacts.[/cyan]")
    return df


def build_join_table(
    reports: dict[str, GraphDiagnosticsReport],
    lift_df: pd.DataFrame,
) -> pd.DataFrame:
    gnn = lift_df[lift_df["matched_graph_lift"].notna()].copy()
    gnn["canonical_graph"] = gnn["graph_name"].map(canonical_graph_name)
    grouped = (
        gnn.groupby("canonical_graph")["matched_graph_lift"]
        .agg(mean_lift="mean", std_lift="std", n_seeds="count")
        .reset_index()
    )

    rows: list[dict[str, object]] = []
    for _, lift_row in grouped.iterrows():
        graph_name = str(lift_row["canonical_graph"])
        report = reports.get(graph_name)
        row: dict[str, object] = {
            "graph_name": graph_name,
            "mean_lift": float(lift_row["mean_lift"]),
            "std_lift": float(0.0 if pd.isna(lift_row["std_lift"]) else lift_row["std_lift"]),
            "n_seeds": int(lift_row["n_seeds"]),
        }
        if report is not None:
            row["num_nodes"] = report.topology.num_nodes
            row["num_edges"] = report.topology.num_edges
            row["log_num_edges"] = float(np.log1p(report.topology.num_edges))
            row["density"] = report.topology.density
            row["isolated_node_fraction"] = report.topology.isolated_node_fraction
            ld = report.label_diagnostics
            md = report.metadata_diagnostics
            if ld is not None:
                row["overall_edge_homophily"] = ld.overall_edge_homophily
                row["train_train_edge_homophily"] = ld.train_train_edge_homophily
                row["test_to_train_query_homophily"] = ld.test_to_train_query_homophily
                row["expected_random_homophily"] = ld.expected_random_homophily
                row["macro_average_class_purity"] = ld.macro_average_class_purity
            if md is not None:
                row["train_intra_donor_edge_fraction"] = md.train_intra_donor_edge_fraction
                row["train_intra_site_edge_fraction"] = md.train_intra_site_edge_fraction
                row["test_to_train_site_match_fraction"] = md.test_to_train_site_match_fraction
                row["mean_train_donor_entropy"] = md.mean_train_donor_entropy
                row["mean_test_query_donor_entropy"] = md.mean_test_query_donor_entropy
        else:
            row["missing_diagnostics"] = True
        rows.append(row)

    join_df = pd.DataFrame(rows)
    feature_cols = [c for c in ["num_nodes", "num_edges"] if c in join_df.columns]
    join_df = join_df.sort_values(feature_cols or ["graph_name"]).reset_index(drop=True)
    return join_df


def compute_correlations(join_df: pd.DataFrame) -> pd.DataFrame:
    available = [c for c in CORRELATION_FEATURES if c in join_df.columns]
    sub = join_df.dropna(subset=["mean_lift", *available])
    rows: list[dict[str, object]] = []
    if len(sub) < 3:
        console.print(
            "[yellow]Fewer than 3 complete graph conditions; correlations skipped.[/yellow]"
        )
        return pd.DataFrame(columns=["feature", "pearson_r", "spearman_rho"])
    for feat in available:
        rows.append(
            {
                "feature": feat,
                "pearson_r": float(sub[feat].corr(sub["mean_lift"], method="pearson")),
                "spearman_rho": float(sub[feat].corr(sub["mean_lift"], method="spearman")),
            }
        )
    return pd.DataFrame(rows).sort_values("spearman_rho", key=np.negative)


def write_scatter(out_dir: Path, join_df: pd.DataFrame) -> Path | None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        console.print("[yellow]matplotlib unavailable; scatter skipped.[/yellow]")
        return None
    sub = join_df.dropna(subset=["overall_edge_homophily", "mean_lift"])
    if len(sub) < 2:
        return None
    fig, ax = plt.subplots(figsize=(7, 5.5))
    ax.scatter(sub["overall_edge_homophily"], sub["mean_lift"], s=48, color="#2980b9")
    for _, r in sub.iterrows():
        ax.annotate(
            str(r["graph_name"]),
            (r["overall_edge_homophily"], r["mean_lift"]),
            fontsize=7,
            xytext=(4, 4),
            textcoords="offset points",
        )
    ax.axhline(0.0, color="k", linewidth=0.8, linestyle="--")
    ax.set_xlabel("Overall Edge Homophily")
    ax.set_ylabel("Mean Matched Graph Lift (GNN − MLP)")
    ax.set_title("Graph Homophily vs Benchmark Lift")
    fig.tight_layout()
    path = out_dir / "homophily_vs_lift.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def run_join(dataset_name: str, split_id: str, make_plots: bool) -> None:
    paths = ArtifactPaths.default()
    res_root = paths.artifacts_dir / "results" / dataset_name / split_id
    diag_root = paths.artifacts_dir / "diagnostics" / dataset_name / split_id
    out_dir = paths.artifacts_dir / "aggregated" / dataset_name / split_id / "diagnostics_join"
    out_dir.mkdir(parents=True, exist_ok=True)

    reports = load_diagnostics(diag_root)
    lift_df = load_lift_stats(res_root)
    if lift_df is None or lift_df.empty:
        raise RuntimeError("No GNN lift statistics available; run sweeps first.")

    join_df = build_join_table(reports, lift_df)
    join_path = out_dir / "diagnostics_join.csv"
    join_df.to_csv(join_path, index=False)
    console.print(f"[green]Wrote[/green] {join_path}")

    corr_df = compute_correlations(join_df)
    corr_path = out_dir / "correlations.csv"
    corr_df.to_csv(corr_path, index=False)
    console.print(f"[green]Wrote[/green] {corr_path}")

    missing = [str(g) for g in join_df.get("graph_name", pd.Series(dtype=str)) if g not in reports]
    info = {
        "dataset_name": dataset_name,
        "split_id": split_id,
        "n_graphs": len(join_df),
        "n_diagnostics_reports": len(reports),
        "graphs_missing_diagnostics": missing,
    }
    (out_dir / "join_manifest.json").write_text(json.dumps(info, indent=2), encoding="utf-8")

    table = Table(title="Graph Diagnostics ↔ Mean Matched Lift")
    display_cols = [
        c
        for c in [
            "graph_name",
            "n_seeds",
            "mean_lift",
            "std_lift",
            "overall_edge_homophily",
            "test_to_train_query_homophily",
            "macro_average_class_purity",
            "train_intra_site_edge_fraction",
            "num_edges",
        ]
        if c in join_df.columns
    ]
    for col in display_cols:
        table.add_column(col, style="cyan" if col == "graph_name" else "white")
    for _, r in join_df.iterrows():
        cells = []
        for col in display_cols:
            v = r[col]
            cells.append(f"{v:.4f}" if isinstance(v, float) else str(v))
        table.add_row(*cells)
    console.print(table)

    if not corr_df.empty:
        corr_table = Table(title="Correlation of Diagnostics with Mean Lift")
        corr_table.add_column("Feature", style="cyan")
        corr_table.add_column("Pearson r", style="green")
        corr_table.add_column("Spearman ρ", style="magenta")
        for _, r in corr_df.iterrows():
            corr_table.add_row(
                str(r["feature"]), f"{r['pearson_r']:+.3f}", f"{r['spearman_rho']:+.3f}"
            )
        console.print(corr_table)

    if make_plots:
        plot_path = write_scatter(out_dir, join_df)
        if plot_path is not None:
            console.print(f"[green]Wrote plot[/green] {plot_path.name}")

    console.print(f"\n[bold green]Diagnostics join artifacts saved to:[/bold green] {out_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=str, default="stephenson_2021_healthy_pbmc")
    parser.add_argument("--split", type=str, default="site_stratified_seed42")
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args()

    run_join(dataset_name=args.dataset, split_id=args.split, make_plots=not args.no_plots)
