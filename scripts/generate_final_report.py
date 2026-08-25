"""Compose the consolidated analysis report from aggregated benchmark artifacts.

Pulls per-class deltas, calibration summaries, training dynamics, embedding quality,
and diagnostics joins into a single dated markdown report under results/reports/.
"""

from __future__ import annotations

import argparse
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from scgraph_bench.utils.paths import ArtifactPaths

AGG_SUBDIRS = {
    "per_class_batch": "per_class_batch",
    "diagnostics_join": "diagnostics_join",
    "embedding_quality": "embedding_quality",
    "training_dynamics": "training_dynamics",
}


def _load_csv(path: Path) -> pd.DataFrame | None:
    return pd.read_csv(path) if path.is_file() else None


def _md_table(df: pd.DataFrame, float_fmt: str = "{:.4f}") -> str:
    if df is None or df.empty:
        return "_not available_"
    out = df.copy()
    for col in out.select_dtypes(include="float").columns:
        out[col] = out[col].map(lambda v: float_fmt.format(v) if pd.notna(v) else "-")
    header = "| " + " | ".join(out.columns) + " |"
    sep = "|" + "---|" * len(out.columns)
    rows = ["| " + " | ".join(str(v) for v in row) + " |" for row in out.to_numpy()]
    return "\n".join([header, sep, *rows])


def generate_report(dataset_name: str, split_id: str) -> Path:
    paths = ArtifactPaths.default()
    agg = paths.artifacts_dir / "aggregated" / dataset_name / split_id
    res_root = paths.artifacts_dir / "results" / dataset_name / split_id
    reports_dir = paths.root_dir / "results" / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    stamp = datetime.now(UTC).strftime("%Y-%m-%d")
    out_path = reports_dir / f"{stamp}_analysis_report.md"

    lift_df = _load_csv(res_root / "graph_lift_summary.csv")
    delta_df = _load_csv(agg / AGG_SUBDIRS["per_class_batch"] / "per_class_matched_delta.csv")
    calib_df = _load_csv(agg / AGG_SUBDIRS["per_class_batch"] / "confidence_summary.csv")
    donor_df = _load_csv(agg / AGG_SUBDIRS["per_class_batch"] / "per_donor_metrics.csv")
    join_df = _load_csv(agg / AGG_SUBDIRS["diagnostics_join"] / "diagnostics_join.csv")
    corr_df = _load_csv(agg / AGG_SUBDIRS["diagnostics_join"] / "correlations.csv")
    emb_df = _load_csv(agg / AGG_SUBDIRS["embedding_quality"] / "embedding_quality.csv")
    dyn_df = _load_csv(agg / AGG_SUBDIRS["training_dynamics"] / "dynamics_summary.csv")

    lines: list[str] = [
        f"# scgraph-bench Analysis Report — {stamp}",
        "",
        f"Dataset: `{dataset_name}` · Split: `{split_id}`",
        "",
        "## 1. Headline: Matched Graph Lift",
        "",
    ]

    if lift_df is not None and not lift_df.empty:
        matched = lift_df[lift_df.get("matched_graph_lift", pd.Series(dtype=float)).notna()]
        if not matched.empty:
            summary = (
                matched.groupby(["model_name", "graph_name"])
                .agg(
                    n_seeds=("seed", "count"),
                    gnn_f1=("gnn_test_macro_f1", "mean"),
                    mlp_f1=("mlp_test_macro_f1", "mean"),
                    lift=("matched_graph_lift", "mean"),
                    lift_sd=("matched_graph_lift", "std"),
                )
                .reset_index()
            )
            lines.append(_md_table(summary))
        else:
            lines.append("_No matched lifts available yet._")
    else:
        lines.append("_Run `compute_graph_lift_offline.py` to populate lift summary._")

    lines.extend(["", "## 2. Per-Class Findings (GNN − MLP ΔF1)", ""])
    if delta_df is not None and not delta_df.empty:
        worst = (
            delta_df.groupby("class_name")["class_delta_f1"]
            .mean()
            .sort_values()
            .head(6)
            .reset_index()
        )
        best = (
            delta_df.groupby("class_name")["class_delta_f1"]
            .mean()
            .sort_values(ascending=False)
            .head(4)
            .reset_index()
        )
        lines.append("**Classes most hurt by graphs:**")
        lines.append("")
        lines.append(_md_table(worst.rename(columns={"class_delta_f1": "mean_delta_f1"})))
        lines.append("")
        lines.append("**Classes most helped by graphs:**")
        lines.append("")
        lines.append(_md_table(best.rename(columns={"class_delta_f1": "mean_delta_f1"})))
    else:
        lines.append(
            "_Per-class deltas require matched GNN + MLP runs (run `analyze_per_class_batch.py`)._"
        )

    lines.extend(["", "## 3. Confidence & Calibration (Test)", ""])
    if calib_df is not None and not calib_df.empty:
        test_cal = calib_df[calib_df["partition"] == "test"]
        cols = [
            "model_name",
            "graph_name",
            "seed",
            "accuracy",
            "ece",
            "brier_score",
            "mean_entropy_nats",
            "mean_margin",
        ]
        lines.append(_md_table(test_cal[[c for c in cols if c in test_cal.columns]]))
    else:
        lines.append("_Calibration summaries pending (`analyze_per_class_batch.py`)._")

    lines.extend(["", "## 4. Per-Donor Stability", ""])
    if donor_df is not None and not donor_df.empty:
        test_donors = donor_df[donor_df["partition"] == "test"]
        spread = (
            test_donors.groupby("model_name")["observed_class_macro_f1"]
            .agg(["min", "mean", "max"])
            .reset_index()
        )
        lines.append(_md_table(spread))
    else:
        lines.append("_Donor breakdowns pending._")

    lines.extend(["", "## 5. Graph Diagnostics vs Lift", ""])
    if join_df is not None and not join_df.empty:
        show = [
            c
            for c in [
                "graph_name",
                "n_seeds",
                "mean_lift",
                "std_lift",
                "overall_edge_homophily",
                "test_to_train_query_homophily",
                "macro_average_class_purity",
            ]
            if c in join_df.columns
        ]
        lines.append(_md_table(join_df[show]))
    else:
        lines.append("_Diagnostics join pending (`join_diagnostics_results.py`)._")
    if corr_df is not None and not corr_df.empty:
        lines.extend(["", "**Correlations with mean lift:**", "", _md_table(corr_df)])

    lines.extend(["", "## 6. Embedding Quality (Test)", ""])
    if emb_df is not None and not emb_df.empty:
        show = [
            c
            for c in [
                "representation_name",
                "silhouette_euclidean",
                "knn_accuracy",
                "centroid_separation",
                "mean_class_radius",
            ]
            if c in emb_df.columns
        ]
        lines.append(_md_table(emb_df[show]))
    else:
        lines.append(
            "_Embedding metrics pending (`analyze_embeddings.py`; requires "
            "instrumented runs with saved embeddings)._"
        )

    lines.extend(["", "## 7. Training Dynamics", ""])
    if dyn_df is not None and not dyn_df.empty:
        show = [
            c
            for c in [
                "model_name",
                "graph_name",
                "n_runs",
                "best_val_f1_mean",
                "best_val_f1_std",
                "epochs_trained_mean",
                "final_train_loss",
                "final_val_loss",
            ]
            if c in dyn_df.columns
        ]
        lines.append(_md_table(dyn_df[show]))
    else:
        lines.append(
            "_Training dynamics pending instrumented re-runs (training_history.csv per run)._"
        )

    lines.extend(
        [
            "",
            "## Reproduction Commands",
            "",
            "```bash",
            f"uv run python scripts/compute_graph_lift_offline.py --dataset {dataset_name} --split {split_id}",
            f"uv run python scripts/analyze_per_class_batch.py --dataset {dataset_name} --split {split_id}",
            f"uv run python scripts/join_diagnostics_results.py --dataset {dataset_name} --split {split_id}",
            f"uv run python scripts/analyze_embeddings.py --dataset {dataset_name} --split {split_id}",
            f"uv run python scripts/analyze_training_dynamics.py --dataset {dataset_name} --split {split_id}",
            f"uv run python scripts/generate_final_report.py --dataset {dataset_name} --split {split_id}",
            "```",
            "",
        ]
    )

    out_path.write_text("\n".join(lines), encoding="utf-8")
    return out_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=str, default="stephenson_2021_healthy_pbmc")
    parser.add_argument("--split", type=str, default="site_stratified_seed42")
    args = parser.parse_args()

    path = generate_report(dataset_name=args.dataset, split_id=args.split)
    print(f"Report written: {path}")
