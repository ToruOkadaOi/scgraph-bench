"""Tidy benchmark results schema with run status, failure tracking, provenance, and graph lift."""

from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum
from typing import Any

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field

from scgraph_bench.utils.hashing import hash_dict


class RunStatus(StrEnum):
    """Execution status of an experimental run or evaluation."""

    SUCCESS = "success"
    FAILED = "failed"
    SKIPPED = "skipped"


class FailureMetadata(BaseModel):
    """Detailed error and failure context for failed/skipped runs."""

    error_type: str | None = None
    error_message: str | None = None
    traceback_summary: str | None = None
    failed_phase: str | None = None
    timestamp: str = Field(
        default_factory=lambda: datetime.now(UTC).isoformat(),
    )


class LabelSupportTracking(BaseModel):
    """Explicit tracking of label support status across donor partitions."""

    unsupported_labels: list[str] = Field(
        default_factory=list,
        description="Labels completely absent from the training partition.",
    )
    low_support_labels: list[str] = Field(
        default_factory=list,
        description="Labels with cell count below the configured minimum threshold.",
    )
    excluded_labels: list[str] = Field(
        default_factory=list,
        description="Labels filtered out by explicit configuration.",
    )
    evaluated_labels: list[str] = Field(
        default_factory=list,
        description="Active labels included in evaluation metrics calculation.",
    )


class RunManifest(BaseModel):
    """Cryptographic provenance manifest for a trained model or baseline run."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    run_id: str
    status: RunStatus = RunStatus.SUCCESS
    model_name: str
    model_config_hash: str
    dataset_name: str
    dataset_version: str = "2025-11-08"
    split_id: str
    split_hash: str = ""
    feature_manifest_hash: str
    preprocessing_config_hash: str = ""
    graph_artifact_hash: str | None = None
    label_mapping_hash: str
    seed: int
    device: str | dict[str, Any] = "cpu"
    code_version: str | None = Field(
        default=None,
        description="Git commit SHA (with dirty flag) capturing the exact source version used.",
    )
    torch_geometric_version: str | None = Field(
        default=None,
        description="Version of the installed torch_geometric library at run time.",
    )
    parameter_count: int | None = None
    best_epoch: int | None = None
    best_val_macro_f1: float | None = None
    selected_params: dict[str, Any] | None = None
    training_time_seconds: float = 0.0
    failure_metadata: FailureMetadata | None = None
    created_at_utc: str = Field(
        default_factory=lambda: datetime.now(UTC).isoformat(),
    )

    def compute_manifest_hash(self) -> str:
        """Compute SHA-256 hash of run manifest."""
        return hash_dict(self.model_dump(mode="json"))


class MetricRecord(BaseModel):
    """Individual atomic metric record adhering to the tidy benchmark schema.

    Schema: dataset × split_id × seed × graph_name × model × metric × partition × stratum
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    run_id: str = ""
    dataset_name: str
    dataset_version: str = "2025-11-08"
    split_id: str
    split_hash: str = ""
    seed: int
    graph_name: str = "none"
    model_name: str
    metric_name: str
    metric_value: float | None = None
    partition: str = Field(
        default="test",
        description="Evaluation split partition (train, val, test).",
    )
    donor_id: str | None = None
    site: str | None = None
    class_label: str | None = None
    observed_support: int | None = None
    status: RunStatus = RunStatus.SUCCESS
    failure_metadata: FailureMetadata | None = None
    feature_manifest_hash: str = ""
    preprocessing_config_hash: str = ""
    graph_artifact_hash: str | None = None
    label_mapping_hash: str = ""
    model_config_hash: str = ""
    runtime_seconds: float = 0.0
    timestamp: str = Field(
        default_factory=lambda: datetime.now(UTC).isoformat(),
    )


class GraphLiftRecord(BaseModel):
    """Explicit matched graph lift comparison record.

    graph_lift = macro_f1(GNN) - macro_f1(matched_MLP)
    Enforces identical dataset, version, split, split_hash, seed, features, preprocessing, and label policy.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    dataset_name: str
    dataset_version: str = "2025-11-08"
    split_id: str
    split_hash: str = ""
    seed: int
    graph_name: str
    gnn_model_name: str
    matched_mlp_model_name: str
    gnn_macro_f1: float
    matched_mlp_macro_f1: float
    overall_graph_lift: float
    gnn_balanced_accuracy: float = 0.0
    matched_mlp_balanced_accuracy: float = 0.0
    balanced_accuracy_lift: float = 0.0
    cambridge_gnn_f1: float | None = None
    cambridge_mlp_f1: float | None = None
    cambridge_lift: float | None = None
    newcastle_gnn_f1: float | None = None
    newcastle_mlp_f1: float | None = None
    newcastle_lift: float | None = None
    per_donor_lifts: dict[str, float] = Field(default_factory=dict)
    per_class_lifts: dict[str, float] = Field(default_factory=dict)
    feature_manifest_hash: str = ""
    preprocessing_config_hash: str = ""
    label_mapping_hash: str = ""
    is_valid_match: bool = True
    notes: str | None = None
    timestamp: str = Field(
        default_factory=lambda: datetime.now(UTC).isoformat(),
    )


class TidyResultsCollection(BaseModel):
    """Collection of metric records and graph lift comparisons."""

    records: list[MetricRecord] = Field(default_factory=list)
    lifts: list[GraphLiftRecord] = Field(default_factory=list)

    def add_record(self, record: MetricRecord) -> None:
        self.records.append(record)

    def add_lift(self, lift: GraphLiftRecord) -> None:
        self.lifts.append(lift)

    def to_dataframe(self) -> pd.DataFrame:
        """Convert metric records to a flat pandas DataFrame."""
        rows = []
        for r in self.records:
            d = r.model_dump()
            d["error_type"] = r.failure_metadata.error_type if r.failure_metadata else None
            d["error_message"] = r.failure_metadata.error_message if r.failure_metadata else None
            rows.append(d)
        return pd.DataFrame(rows)

    def lifts_to_dataframe(self) -> pd.DataFrame:
        """Convert graph lift records to a flat pandas DataFrame."""
        rows = [lift_rec.model_dump() for lift_rec in self.lifts]
        return pd.DataFrame(rows)
