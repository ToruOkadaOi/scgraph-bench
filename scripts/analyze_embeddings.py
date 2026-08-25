"""Analyze embedding quality across GNN, MLP, and raw-input representations.

For each selected run, loads saved hidden-layer embeddings and compares their
geometry (silhouette, kNN separability, centroid separation) against the raw
PCA input space. Also exports 2D UMAP coordinates when umap-learn is available
(falls back to PCA projection otherwise).
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
from rich.console import Console
from rich.table import Table

from scgraph_bench.analysis.embedding_quality import (
    compare_representations,
    compute_embedding_quality,
)
from scgraph_bench.analysis.flatten import describe_run, discover_run_records
from scgraph_bench.preprocessing.schema import PreprocessedBundle
from scgraph_bench.utils.paths import ArtifactPaths

console = Console()


def _project_2d(emb: np.ndarray) -> np.ndarray:
    try:
        import umap

        return np.asarray(umap.UMAP(n_components=2, random_state=42, n_jobs=1).fit_transform(emb))
    except ImportError:
        from sklearn.decomposition import PCA

        return np.asarray(PCA(n_components=2, random_state=42).fit_transform(emb))


def _plot_projection(
    coords: np.ndarray,
    y: np.ndarray,
    donor: np.ndarray | None,
    site: np.ndarray | None,
    title: str,
    out_path,
    label_names: list[str] | None,
    max_points: int = 6000,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rng = np.random.default_rng(42)
    idx = rng.permutation(len(coords))[:max_points]
    c, yy = coords[idx], y[idx]

    panels = [("Cell type", yy)]
    if donor is not None:
        panels.append(("Donor", np.asarray(donor)[idx]))
    if site is not None:
        panels.append(("Site", np.asarray(site)[idx]))

    fig, axes = plt.subplots(1, len(panels), figsize=(5.5 * len(panels), 4.6), squeeze=False)
    for ax, (label, values) in zip(axes[0], panels, strict=True):
        codes = pd.factorize(values)[0]
        sc = ax.scatter(c[:, 0], c[:, 1], c=codes, cmap="tab20", s=3, alpha=0.6, linewidths=0)
        ax.set_title(f"{title} — {label}", fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])
        n_unique = len(np.unique(values))
        if label != "Site" and n_unique <= len(plt.get_cmap("tab20").colors):
            handles = [
                plt.Line2D(
                    [],
                    [],
                    marker="o",
                    ls="",
                    color=sc.to_rgba(i),
                    ms=4,
                    label=(label_names[i] if label_names and i < len(label_names) else str(i)),
                )
                for i in range(n_unique)
            ]
            ax.legend(handles=handles, fontsize=4.5, loc="best", ncol=2, framealpha=0.5)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def run_embedding_analysis(
    dataset_name: str,
    split_id: str,
    seeds: list[int] | None,
    make_plots: bool,
) -> None:
    paths = ArtifactPaths.default()
    res_root = paths.artifacts_dir / "results" / dataset_name / split_id
    prep_dir = paths.artifacts_dir / "preprocessed" / dataset_name / split_id
    out_dir = paths.artifacts_dir / "aggregated" / dataset_name / split_id / "embedding_quality"
    out_dir.mkdir(parents=True, exist_ok=True)

    bundle = PreprocessedBundle.load(prep_dir)
    inv = {v: k for k, v in bundle.label_to_id.items()}
    label_names = [inv[i] for i in range(len(inv))]
    y_train, y_test = bundle.train_labels, bundle.test_labels

    cell_meta_path = prep_dir / "cell_metadata.json"
    donor_by_part = site_by_part = {}
    if cell_meta_path.is_file():
        import json

        meta = json.loads(cell_meta_path.read_text(encoding="utf-8"))
        donor_by_part = {
            "train": meta["train_donors"],
            "val": meta["val_donors"],
            "test": meta["test_donors"],
        }
        site_by_part = {
            "train": meta["train_sites"],
            "val": meta["val_sites"],
            "test": meta["test_sites"],
        }

    records = discover_run_records(res_root)
    if not records:
        raise RuntimeError(f"No runs discovered under {res_root}")

    reports = []
    plot_jobs = []
    for rec in records:
        _, graph_name = describe_run(rec.run_id, rec.manifest.model_name)
        if seeds is not None and rec.manifest.seed not in seeds:
            continue
        emb_test_path = rec.run_dir / "embeddings_test.npy"
        if not emb_test_path.is_file():
            continue
        emb_test = np.load(emb_test_path)
        name = (
            f"{rec.manifest.model_name}:{graph_name}"
            if graph_name != "none"
            else rec.manifest.model_name
        )
        ref_emb = None
        ref_y = None
        emb_train_path = rec.run_dir / "embeddings_train.npy"
        if emb_train_path.is_file():
            ref_emb = np.load(emb_train_path)
            ref_y = y_train
        reports.append(
            compute_embedding_quality(
                emb_test,
                y_test,
                representation_name=name,
                partition="test",
                reference_emb=ref_emb,
                reference_y_train=ref_y,
            )
        )
        if make_plots:
            plot_jobs.append(
                (
                    name,
                    rec.manifest.seed,
                    emb_test,
                    y_test,
                    donor_by_part.get("test"),
                    site_by_part.get("test"),
                )
            )

    for part_name, X_ref, y_ref in [
        ("raw_pca_input", bundle.X_pca_test, y_test),
    ]:
        reports.append(
            compute_embedding_quality(
                X_ref,
                y_ref,
                representation_name=part_name,
                partition="test",
                reference_emb=bundle.X_pca_train,
                reference_y_train=y_train,
            )
        )

    df = compare_representations(reports)
    df.to_csv(out_dir / "embedding_quality.csv", index=False)
    console.print(f"[green]Wrote[/green] {out_dir / 'embedding_quality.csv'}")

    table = Table(title="Embedding Quality — Test Partition")
    table.add_column("Representation", style="cyan")
    table.add_column("Silhouette", justify="right")
    table.add_column("kNN Acc", justify="right")
    table.add_column("Centroid Sep", justify="right")
    table.add_column("Mean Radius", justify="right")

    def fmt(v) -> str:
        return f"{v:.4f}" if isinstance(v, float) and v == v else "-"

    for _, r in df.iterrows():
        table.add_row(
            str(r["representation_name"]),
            fmt(r.get("silhouette_euclidean")),
            fmt(r.get("knn_accuracy")),
            fmt(r.get("centroid_separation")),
            fmt(r.get("mean_class_radius")),
        )
    console.print(table)

    if make_plots and plot_jobs:
        plots_dir = out_dir / "projections"
        plots_dir.mkdir(exist_ok=True)
        for name, seed, emb, y, donors, sites in plot_jobs:
            coords = _project_2d(emb)
            safe = f"{name.replace('/', '_')}_seed{seed}"
            _plot_projection(
                coords,
                y,
                donors,
                sites,
                f"{name} (test)",
                plots_dir / f"{safe}.png",
                label_names=label_names,
            )
            console.print(f"[green]Wrote plot[/green] {plots_dir / f'{safe}.png'}")

    console.print(f"\n[bold green]Embedding analysis saved to:[/bold green] {out_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=str, default="stephenson_2021_healthy_pbmc")
    parser.add_argument("--split", type=str, default="site_stratified_seed42")
    parser.add_argument("--seeds", nargs="+", type=int, default=None)
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args()

    run_embedding_analysis(
        dataset_name=args.dataset,
        split_id=args.split,
        seeds=args.seeds,
        make_plots=not args.no_plots,
    )
