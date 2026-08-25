"""Flatten persisted run artifacts into tidy per-class and per-donor DataFrames."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from scgraph_bench.evaluation.schema import EvaluationSummary
from scgraph_bench.tracking.schema import RunManifest
from scgraph_bench.utils.logging import get_logger

logger = get_logger("analysis.flatten")

GNN_RUN_PATTERN = re.compile(r"^(gcn|graphsage)_(.+)_seed(\d+)$")

NON_GRAPH_MODELS = frozenset({"mlp", "logistic_regression"})


@dataclass(frozen=True)
class RunRecord:
    """A validated on-disk benchmark run with manifest and evaluation summaries."""

    run_id: str
    run_dir: Path
    manifest: RunManifest
    summaries: dict[str, EvaluationSummary]

    @property
    def model_name(self) -> str:
        return self.manifest.model_name

    @property
    def graph_name(self) -> str:
        return describe_run(self.run_id, self.manifest.model_name)[1]

    @property
    def seed(self) -> int:
        return self.manifest.seed


def describe_run(run_id: str, model_name: str) -> tuple[str, str]:
    """Resolve (model_name, graph_name) from a run identifier.

    GNN runs follow the convention '<model>_<graph>_seed<seed>'; non-graph baselines are
    reported against the implicit 'none' graph.
    """
    if model_name in NON_GRAPH_MODELS:
        return model_name, "none"
    match = GNN_RUN_PATTERN.match(run_id)
    if match is None:
        logger.warning("Could not parse graph name from run id '%s'", run_id)
        return model_name, "unknown"
    return match.group(1), match.group(2)


def load_run_record(run_dir: Path) -> RunRecord | None:
    """Load a single run directory, returning None if it lacks valid artifacts."""
    manifest_path = run_dir / "run_manifest.json"
    metrics_path = run_dir / "metrics_summary.json"
    if not (manifest_path.is_file() and metrics_path.is_file()):
        return None
    try:
        manifest = RunManifest.model_validate_json(manifest_path.read_text(encoding="utf-8"))
        raw = json.loads(metrics_path.read_text(encoding="utf-8"))
        summaries = {part: EvaluationSummary.model_validate(v) for part, v in raw.items()}
    except Exception as err:
        logger.warning("Skipping unreadable run dir %s: %s", run_dir.name, err)
        return None
    return RunRecord(
        run_id=manifest.run_id or run_dir.name,
        run_dir=run_dir,
        manifest=manifest,
        summaries=summaries,
    )


def discover_run_records(results_root: Path | str) -> list[RunRecord]:
    """Discover and validate all run directories under a results root."""
    root = Path(results_root)
    if not root.is_dir():
        logger.warning("Results root does not exist: %s", root)
        return []
    records: list[RunRecord] = []
    for run_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        record = load_run_record(run_dir)
        if record is not None:
            records.append(record)
    return records


def flatten_per_class(
    records: list[RunRecord],
    partitions: tuple[str, ...] = ("test",),
) -> pd.DataFrame:
    """Flatten per-class metrics across runs into one tidy DataFrame."""
    rows: list[dict[str, object]] = []
    for rec in records:
        model_name, graph_name = describe_run(rec.run_id, rec.manifest.model_name)
        for partition in partitions:
            summary = rec.summaries.get(partition)
            if summary is None:
                continue
            for pc in summary.per_class:
                rows.append(
                    {
                        "run_id": rec.run_id,
                        "dataset_name": rec.manifest.dataset_name,
                        "split_id": rec.manifest.split_id,
                        "model_name": model_name,
                        "graph_name": graph_name,
                        "seed": rec.manifest.seed,
                        "partition": partition,
                        "class_index": pc.class_index,
                        "class_name": pc.class_name,
                        "precision": pc.precision,
                        "recall": pc.recall,
                        "f1": pc.f1,
                        "support": pc.support,
                    }
                )
    return pd.DataFrame(rows)


def compute_matched_per_class_deltas(per_class_df: pd.DataFrame) -> pd.DataFrame | None:
    """Join GNN per-class F1 with seed-matched MLP per-class F1 to produce class-level deltas.

    Returns None when no MLP baseline or no GNN runs are present in the input.
    """
    if "partition" not in per_class_df.columns or per_class_df.empty:
        return None
    test_df = per_class_df[per_class_df["partition"] == "test"]
    mlp = test_df[test_df["model_name"] == "mlp"]
    gnn = test_df[test_df["model_name"].isin(("gcn", "graphsage"))]
    if mlp.empty or gnn.empty:
        return None
    n_dupes = int(mlp.duplicated(subset=["seed", "class_index"]).sum())
    if n_dupes:
        logger.warning(
            "Dropping %d duplicate MLP (seed, class_index) rows; keeping first occurrence.",
            n_dupes,
        )
    lookup = (
        mlp.drop_duplicates(subset=["seed", "class_index"], keep="first")
        .set_index(["seed", "class_index"])["f1"]
        .sort_index()
        .to_dict()
    )
    out = gnn.copy()
    keys = list(zip(out["seed"], out["class_index"], strict=True))
    out["mlp_f1"] = [float(lookup.get(k, np.nan)) for k in keys]
    out["class_delta_f1"] = out["f1"] - out["mlp_f1"]
    return out


def flatten_per_donor(
    records: list[RunRecord],
    partitions: tuple[str, ...] = ("test",),
) -> pd.DataFrame:
    """Flatten stratified per-donor breakdowns across runs into one tidy DataFrame."""
    rows: list[dict[str, object]] = []
    for rec in records:
        model_name, graph_name = describe_run(rec.run_id, rec.manifest.model_name)
        for partition in partitions:
            summary = rec.summaries.get(partition)
            if summary is None:
                continue
            for donor in summary.per_donor:
                rows.append(
                    {
                        "run_id": rec.run_id,
                        "dataset_name": rec.manifest.dataset_name,
                        "split_id": rec.manifest.split_id,
                        "model_name": model_name,
                        "graph_name": graph_name,
                        "seed": rec.manifest.seed,
                        "partition": partition,
                        "donor_id": donor.donor_id,
                        "site": donor.site,
                        "observed_class_macro_f1": donor.observed_class_macro_f1,
                        "global_label_macro_f1": donor.global_label_macro_f1,
                        "balanced_accuracy": donor.balanced_accuracy,
                        "support": donor.support,
                    }
                )
    return pd.DataFrame(rows)
