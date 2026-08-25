"""Per-class, per-donor, and confidence-calibration analysis over benchmark run artifacts.

Scans artifacts/results/<dataset>/<split>/ for all valid runs, emits tidy CSV tables,
matched GNN-vs-MLP per-class deltas, calibration summaries (retroactively computed from
saved probability matrices), and diagnostic plots.
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd
from rich.console import Console
from rich.table import Table

from scgraph_bench.analysis.calibration import summarize_confidence
from scgraph_bench.analysis.flatten import (
    RunRecord,
    compute_matched_per_class_deltas,
    describe_run,
    discover_run_records,
    flatten_per_class,
    flatten_per_donor,
)
from scgraph_bench.preprocessing.schema import PreprocessedBundle
from scgraph_bench.utils.paths import ArtifactPaths

console = Console()

PARTITIONS = ("train", "val", "test")
LOW_MARGIN_THRESHOLD = 0.1


def _shorten(name: str, width: int = 26) -> str:
    return name if len(name) <= width else name[: width - 1] + "…"


def compute_calibration_records(
    records: list[RunRecord],
    prep_dir,
    partitions: tuple[str, ...],
    n_bins: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compute run-level confidence summaries and per-bin reliability data."""
    bundle = PreprocessedBundle.load(prep_dir)
    label_arrays = {
        "train": bundle.train_labels,
        "val": bundle.val_labels,
        "test": bundle.test_labels,
    }

    summary_rows: list[dict[str, object]] = []
    bin_rows: list[dict[str, object]] = []
    for rec in records:
        model_name, graph_name = describe_run(rec.run_id, rec.manifest.model_name)
        probs_by_part = _load_probs(rec.run_dir)
        if not probs_by_part:
            console.print(f"[yellow]No probability files in {rec.run_id}; skipping[/yellow]")
            continue
        for partition in partitions:
            probs = probs_by_part.get(partition)
            y_true = label_arrays.get(partition)
            if probs is None or y_true is None or len(probs) != len(y_true):
                continue
            summary = summarize_confidence(
                y_true=y_true,
                probs=np.asarray(probs, dtype=np.float64),
                run_id=rec.run_id,
                partition=partition,
                n_bins=n_bins,
                low_margin_threshold=LOW_MARGIN_THRESHOLD,
            )
            summary_rows.append(
                {
                    "run_id": rec.run_id,
                    "model_name": model_name,
                    "graph_name": graph_name,
                    "seed": rec.manifest.seed,
                    "partition": partition,
                    "n_samples": summary.n_samples,
                    "accuracy": summary.accuracy,
                    "ece": summary.ece,
                    "brier_score": summary.brier_score,
                    "mean_max_confidence": summary.mean_max_confidence,
                    "mean_entropy_nats": summary.mean_entropy_nats,
                    "mean_margin": summary.mean_margin,
                    "fraction_low_margin": summary.fraction_low_margin,
                }
            )
            for b in summary.bins:
                bin_rows.append(
                    {
                        "run_id": rec.run_id,
                        "model_name": model_name,
                        "graph_name": graph_name,
                        "seed": rec.manifest.seed,
                        "partition": partition,
                        "bin_index": b.bin_index,
                        "bin_lower": b.bin_lower,
                        "bin_upper": b.bin_upper,
                        "count": b.count,
                        "accuracy": b.accuracy,
                        "mean_confidence": b.mean_confidence,
                    }
                )
    return pd.DataFrame(summary_rows), pd.DataFrame(bin_rows)


def _load_probs(run_dir) -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {}
    for partition in PARTITIONS:
        path = run_dir / f"{partition}_probs.npy"
        if path.is_file():
            try:
                out[partition] = np.load(path)
            except Exception as err:
                console.print(f"[yellow]Unreadable {path.name} in {run_dir.name}: {err}[/yellow]")
    return out


def compute_matched_class_deltas(per_class_df: pd.DataFrame) -> pd.DataFrame | None:
    """Matched GNN-vs-MLP per-class deltas (delegates to the analysis library)."""
    return compute_matched_per_class_deltas(per_class_df)


def write_plots(
    out_dir,
    per_class_df: pd.DataFrame,
    deltas: pd.DataFrame | None,
    per_donor_df: pd.DataFrame,
    calib_bins: pd.DataFrame,
) -> list:
    """Render analysis plots; returns list of written paths."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        console.print("[yellow]matplotlib unavailable; plots skipped.[/yellow]")
        return []

    written: list = []
    test_pc = per_class_df[per_class_df["partition"] == "test"]

    if deltas is not None and not deltas.empty:
        agg = (
            deltas.groupby(["model_name", "graph_name", "class_name"])["class_delta_f1"]
            .mean()
            .reset_index()
        )
        agg["row"] = agg["model_name"] + "\n" + agg["graph_name"].str.replace("_", " ")
        pivot = agg.pivot(index="row", columns="class_name", values="class_delta_f1")
        fig, ax = plt.subplots(figsize=(14, 0.6 * len(pivot) + 3))
        vmax = float(np.nanmax(np.abs(pivot.to_numpy(dtype=float))))
        vmax = vmax if vmax > 0 else 0.05
        im = ax.imshow(pivot.to_numpy(dtype=float), cmap="RdBu_r", vmin=-vmax, vmax=vmax)
        ax.set_xticks(range(len(pivot.columns)))
        ax.set_xticklabels([_shorten(c) for c in pivot.columns], rotation=45, ha="right")
        ax.set_yticks(range(len(pivot.index)))
        ax.set_yticklabels(pivot.index)
        plt.colorbar(im, ax=ax, label="Mean ΔF1 (GNN − MLP)")
        ax.set_title("Per-Class Test Macro-F1 Delta vs Matched MLP (mean across seeds)")
        fig.tight_layout()
        path = out_dir / "per_class_delta_heatmap.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        written.append(path)

    class_means = test_pc.groupby("class_name")["f1"].mean().sort_values()
    if not class_means.empty:
        worst = class_means.head(8)
        fig, ax = plt.subplots(figsize=(9, max(4, 0.5 * len(worst) + 1)))
        ax.barh([_shorten(c) for c in worst.index], worst.to_numpy(), color="#c0392b")
        ax.set_xlabel("Mean Test F1 (all runs)")
        ax.set_title("Worst-Class Performance Across All Runs")
        fig.tight_layout()
        path = out_dir / "worst_classes_bar.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        written.append(path)

    donor = per_donor_df[per_donor_df["partition"] == "test"]
    mlp_donor = donor[donor["model_name"] == "mlp"].drop_duplicates(
        subset=["seed", "donor_id"], keep="first"
    )
    mlp_lookup = mlp_donor.set_index(["seed", "donor_id"])["observed_class_macro_f1"].to_dict()
    gnn_donor = donor[donor["model_name"].isin(["gcn", "graphsage"])].copy()
    if not gnn_donor.empty and mlp_lookup:
        keys = list(zip(gnn_donor["seed"], gnn_donor["donor_id"], strict=True))
        gnn_donor["mlp_f1"] = [float(mlp_lookup.get(k, np.nan)) for k in keys]
        valid = gnn_donor.dropna(subset=["mlp_f1"])
        if not valid.empty:
            fig, ax = plt.subplots(figsize=(6, 6))
            ax.scatter(valid["mlp_f1"], valid["observed_class_macro_f1"], alpha=0.7, s=28)
            lims = [
                min(valid["mlp_f1"].min(), valid["observed_class_macro_f1"].min()) - 0.02,
                max(valid["mlp_f1"].max(), valid["observed_class_macro_f1"].max()) + 0.02,
            ]
            ax.plot(lims, lims, "k--", linewidth=1, label="y = x")
            ax.set_xlabel("Matched MLP Donor Macro-F1")
            ax.set_ylabel("GNN Donor Macro-F1")
            ax.set_title("Per-Donor Test Performance (observed classes)")
            ax.legend()
            fig.tight_layout()
            path = out_dir / "per_donor_scatter.png"
            fig.savefig(path, dpi=150)
            plt.close(fig)
            written.append(path)

    test_bins = calib_bins[calib_bins["partition"] == "test"] if not calib_bins.empty else None
    if test_bins is not None and not test_bins.empty:
        conds = test_bins[["model_name", "graph_name"]].drop_duplicates().reset_index(drop=True)
        n_cols = min(3, max(1, len(conds)))
        n_rows = int(np.ceil(len(conds) / n_cols)) if len(conds) else 1
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3.6 * n_rows), squeeze=False)
        for idx, cond in conds.iterrows():
            ax = axes[idx // n_cols][idx % n_cols]
            sub = test_bins[
                (test_bins["model_name"] == cond["model_name"])
                & (test_bins["graph_name"] == cond["graph_name"])
                & (test_bins["count"] > 0)
            ]
            for _seed, grp in sub.groupby("seed"):
                ax.plot(grp["mean_confidence"], grp["accuracy"], marker="o", ms=3, lw=1, alpha=0.7)
            ax.plot([0, 1], [0, 1], "k--", lw=1)
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.set_title(f"{cond['model_name']} · {cond['graph_name']}", fontsize=9)
            ax.set_xlabel("Confidence")
            ax.set_ylabel("Accuracy")
        total = n_rows * n_cols
        for idx in range(len(conds), total):
            axes[idx // n_cols][idx % n_cols].axis("off")
        fig.suptitle("Reliability Diagrams (Test Partition, one line per seed)")
        fig.tight_layout()
        path = out_dir / "reliability_diagrams.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        written.append(path)

    return written


def run_analysis(
    dataset_name: str,
    split_id: str,
    partitions: tuple[str, ...],
    ece_bins: int,
    make_plots: bool,
) -> None:
    paths = ArtifactPaths.default()
    res_root = paths.artifacts_dir / "results" / dataset_name / split_id
    if not res_root.is_dir():
        raise FileNotFoundError(f"Results directory not found: {res_root}")

    records = discover_run_records(res_root)
    if not records:
        raise RuntimeError(f"No valid runs discovered under {res_root}")
    console.print(f"[bold cyan]Discovered {len(records)} runs under[/bold cyan] {res_root}")

    out_dir = paths.artifacts_dir / "aggregated" / dataset_name / split_id / "per_class_batch"
    out_dir.mkdir(parents=True, exist_ok=True)

    per_class_df = flatten_per_class(records, partitions=partitions)
    per_donor_df = flatten_per_donor(records, partitions=partitions)
    per_class_df.to_csv(out_dir / "per_class_metrics.csv", index=False)
    per_donor_df.to_csv(out_dir / "per_donor_metrics.csv", index=False)
    console.print(f"[green]Wrote[/green] per_class_metrics.csv ({len(per_class_df)} rows)")
    console.print(f"[green]Wrote[/green] per_donor_metrics.csv ({len(per_donor_df)} rows)")

    deltas = compute_matched_class_deltas(per_class_df)
    if deltas is not None:
        delta_cols = [
            "run_id",
            "model_name",
            "graph_name",
            "seed",
            "class_index",
            "class_name",
            "f1",
            "mlp_f1",
            "class_delta_f1",
        ]
        deltas[delta_cols].to_csv(out_dir / "per_class_matched_delta.csv", index=False)
        console.print(f"[green]Wrote[/green] per_class_matched_delta.csv ({len(deltas)} rows)")

    prep_dir = paths.artifacts_dir / "preprocessed" / dataset_name / split_id
    if (prep_dir / "feature_manifest.json").is_file():
        calib_summary, calib_bins = compute_calibration_records(
            records, prep_dir, partitions, ece_bins
        )
        calib_summary.to_csv(out_dir / "confidence_summary.csv", index=False)
        calib_bins.to_csv(out_dir / "calibration_bins.csv", index=False)
        console.print(
            f"[green]Wrote[/green] confidence_summary.csv ({len(calib_summary)} rows) "
            f"and calibration_bins.csv ({len(calib_bins)} rows)"
        )
    else:
        calib_bins = pd.DataFrame()
        console.print("[yellow]Preprocessed bundle missing; calibration skipped.[/yellow]")

    if make_plots:
        written = write_plots(out_dir, per_class_df, deltas, per_donor_df, calib_bins)
        for p in written:
            console.print(f"[green]Wrote plot[/green] {p.name}")

    test_calib_path = out_dir / "confidence_summary.csv"
    if test_calib_path.is_file():
        calib = pd.read_csv(test_calib_path)
        calib = calib[calib["partition"] == "test"]
        if not calib.empty:
            table = Table(title="Confidence & Calibration (Test Partition)")
            table.add_column("Model", style="cyan")
            table.add_column("Graph", style="cyan")
            table.add_column("Seed", style="yellow")
            table.add_column("Acc", style="green")
            table.add_column("ECE", style="red")
            table.add_column("Brier", style="magenta")
            table.add_column("Entropy", style="blue")
            table.add_column("Margin", style="blue")
            for _, r in calib.iterrows():
                table.add_row(
                    str(r["model_name"]),
                    str(r["graph_name"]),
                    str(r["seed"]),
                    f"{r['accuracy']:.4f}",
                    f"{r['ece']:.4f}",
                    f"{r['brier_score']:.4f}",
                    f"{r['mean_entropy_nats']:.3f}",
                    f"{r['mean_margin']:.3f}",
                )
            console.print(table)

    manifest_info = {
        "dataset_name": dataset_name,
        "split_id": split_id,
        "n_runs": len(records),
        "ece_bins": ece_bins,
        "partitions": list(partitions),
    }
    (out_dir / "analysis_manifest.json").write_text(
        json.dumps(manifest_info, indent=2), encoding="utf-8"
    )
    console.print(f"\n[bold green]Analysis artifacts saved to:[/bold green] {out_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=str, default="stephenson_2021_healthy_pbmc")
    parser.add_argument("--split", type=str, default="site_stratified_seed42")
    parser.add_argument("--partitions", nargs="+", choices=PARTITIONS, default=["test"])
    parser.add_argument("--ece-bins", type=int, default=15)
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args()

    run_analysis(
        dataset_name=args.dataset,
        split_id=args.split,
        partitions=tuple(args.partitions),
        ece_bins=args.ece_bins,
        make_plots=not args.no_plots,
    )
