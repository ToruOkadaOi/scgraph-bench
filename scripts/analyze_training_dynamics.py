"""Analyze training dynamics from persisted per-epoch histories.

Aggregates training_history.csv files across runs to compare convergence speed,
loss/F1 trajectories, overfitting gaps, and seed stability across graph variants.
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
from rich.console import Console
from rich.table import Table

from scgraph_bench.analysis.flatten import describe_run, discover_run_records
from scgraph_bench.utils.paths import ArtifactPaths

console = Console()

HISTORY_COLUMNS = {"epoch", "train_loss", "val_macro_f1"}


def load_histories(res_root) -> pd.DataFrame:
    """Concatenate all run histories into one tidy DataFrame with run identity columns."""
    records = discover_run_records(res_root)
    frames: list[pd.DataFrame] = []
    for rec in records:
        hist_path = rec.run_dir / "training_history.csv"
        if not hist_path.is_file():
            continue
        try:
            df = pd.read_csv(hist_path)
        except Exception as err:
            console.print(f"[yellow]Unreadable history in {rec.run_id}: {err}[/yellow]")
            continue
        if not HISTORY_COLUMNS.issubset(df.columns):
            continue
        model_name, graph_name = describe_run(rec.run_id, rec.manifest.model_name)
        df = df.copy()
        df.insert(0, "run_id", rec.run_id)
        df.insert(1, "model_name", model_name)
        df.insert(2, "graph_name", graph_name)
        df.insert(3, "seed", rec.manifest.seed)
        if "val_loss" not in df.columns:
            df["val_loss"] = np.nan
        frames.append(df)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def summarize_dynamics(df: pd.DataFrame) -> pd.DataFrame:
    def _agg(grp: pd.DataFrame) -> pd.Series:
        per_run = grp.groupby("run_id")
        n_seeds = grp["seed"].nunique()
        final_epochs = per_run["epoch"].max()
        final = grp[grp.apply(lambda r: r["epoch"] == final_epochs[r["run_id"]], axis=1)]
        return pd.Series(
            {
                "n_runs": len(per_run),
                "n_seeds": n_seeds,
                "epochs_trained_mean": float(final_epochs.mean()),
                "best_epoch_mean": float(grp.loc[grp["val_macro_f1"].idxmax(), "epoch"]),
                "best_val_f1_mean": float(per_run["val_macro_f1"].max().mean()),
                "best_val_f1_std": float(per_run["val_macro_f1"].max().std()),
                "final_train_loss": float(final["train_loss"].mean()),
                "final_val_loss": (
                    float(final["val_loss"].mean()) if final["val_loss"].notna().any() else np.nan
                ),
            }
        )

    return df.groupby(["model_name", "graph_name"]).apply(_agg, include_groups=False).reset_index()


def write_plots(out_dir, df: pd.DataFrame) -> list:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        console.print("[yellow]matplotlib unavailable; plots skipped.[/yellow]")
        return []

    written: list = []
    df["condition"] = df["model_name"] + " · " + df["graph_name"]
    conditions = sorted(df["condition"].unique())

    palette = plt.get_cmap("tab10")
    color_map = {c: palette(i % 10) for i, c in enumerate(conditions)}

    fig, ax = plt.subplots(figsize=(11, 6))
    for cond, grp in df.groupby("condition"):
        for _seed, seed_grp in grp.groupby("seed"):
            ax.plot(
                seed_grp["epoch"],
                seed_grp["val_macro_f1"],
                color=color_map[cond],
                alpha=0.35,
                lw=1,
            )
        boot = grp.groupby("epoch")["val_macro_f1"].agg(["mean", "std"]).reset_index()
        ax.plot(boot["epoch"], boot["mean"], color=color_map[cond], lw=2, label=cond)
        if len(grp["seed"].unique()) > 1:
            ax.fill_between(
                boot["epoch"],
                boot["mean"] - boot["std"].fillna(0),
                boot["mean"] + boot["std"].fillna(0),
                color=color_map[cond],
                alpha=0.12,
            )
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Validation Macro-F1")
    ax.set_title("Validation Macro-F1 Trajectories (thin=seed, bold=mean±std)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    path = out_dir / "val_f1_curves.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    written.append(path)

    if "val_loss" in df.columns and df["val_loss"].notna().any():
        fig, ax = plt.subplots(figsize=(11, 6))
        for cond, grp in df.groupby("condition"):
            agg = grp.groupby("epoch")[["train_loss", "val_loss"]].mean().reset_index()
            ax.plot(
                agg["epoch"],
                agg["train_loss"],
                "--",
                color=color_map[cond],
                lw=1.5,
                label=f"{cond} train",
            )
            ax.plot(
                agg["epoch"],
                agg["val_loss"],
                "-",
                color=color_map[cond],
                lw=1.5,
                label=f"{cond} val",
            )
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Cross-Entropy Loss (mean over seeds)")
        ax.set_title("Loss Curves: train (dashed) vs validation (solid)")
        ax.legend(fontsize=7)
        fig.tight_layout()
        path = out_dir / "loss_curves.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        written.append(path)

    best_by_run = df[
        df["val_macro_f1"] == df.groupby("run_id")["val_macro_f1"].transform("max")
    ].drop_duplicates("run_id")
    if not best_by_run.empty:
        fig, ax = plt.subplots(figsize=(9, 4.5))
        stats = best_by_run.groupby("condition")["epoch"]
        positions = np.arange(len(stats))
        ax.violinplot(
            [g.to_numpy() for _, g in stats],
            positions=positions,
            showmedians=True,
            widths=0.7,
        )
        ax.set_xticks(positions)
        ax.set_xticklabels([c.replace(" · ", "\n· ") for c in stats.count().index], fontsize=8)
        ax.set_ylabel("Best Epoch")
        ax.set_title("Convergence Speed by Condition")
        fig.tight_layout()
        path = out_dir / "best_epoch_distribution.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        written.append(path)

    return written


def run_dynamics_analysis(dataset_name: str, split_id: str, make_plots: bool) -> None:
    paths = ArtifactPaths.default()
    res_root = paths.artifacts_dir / "results" / dataset_name / split_id
    out_dir = paths.artifacts_dir / "aggregated" / dataset_name / split_id / "training_dynamics"
    out_dir.mkdir(parents=True, exist_ok=True)

    df = load_histories(res_root)
    if df.empty:
        raise RuntimeError(
            f"No training histories found under {res_root}. "
            "Instrumented sweeps persist training_history.csv per run."
        )
    console.print(f"[cyan]Loaded {df['run_id'].nunique()} run histories[/cyan]")

    summary = summarize_dynamics(df)
    summary.to_csv(out_dir / "dynamics_summary.csv", index=False)

    table = Table(title="Training Dynamics Summary")
    table.add_column("Model", style="cyan")
    table.add_column("Graph", style="cyan")
    table.add_column("Runs", justify="right")
    table.add_column("Best Val F1", justify="right")
    table.add_column("SD", justify="right")
    table.add_column("Final Tr Loss", justify="right")
    table.add_column("Final Va Loss", justify="right")

    def fmt(v) -> str:
        return "-" if pd.isna(v) else f"{v:.4f}"

    for _, r in summary.iterrows():
        table.add_row(
            str(r["model_name"]),
            str(r["graph_name"]),
            str(int(r["n_runs"])),
            fmt(r["best_val_f1_mean"]),
            fmt(r["best_val_f1_std"]),
            fmt(r["final_train_loss"]),
            fmt(r["final_val_loss"]),
        )
    console.print(table)

    if make_plots:
        for p in write_plots(out_dir, df.drop(columns=["condition"], errors="ignore")):
            console.print(f"[green]Wrote plot[/green] {p.name}")

    console.print(f"\n[bold green]Dynamics artifacts saved to:[/bold green] {out_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=str, default="stephenson_2021_healthy_pbmc")
    parser.add_argument("--split", type=str, default="site_stratified_seed42")
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args()

    run_dynamics_analysis(
        dataset_name=args.dataset,
        split_id=args.split,
        make_plots=not args.no_plots,
    )
