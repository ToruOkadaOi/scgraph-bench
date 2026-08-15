"""Tidy benchmark results schema with run status, failure tracking, and graph lift."""

from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum
from typing import Any

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field


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


class MetricRecord(BaseModel):
    """Individual atomic metric record adhering to the tidy benchmark schema.

    Schema: dataset × split_id × seed × graph_name × graph_settings × model × metric
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    dataset: str
    split_id: str
    seed: int
    graph_name: str
    graph_settings: dict[str, Any] = Field(default_factory=dict)
    model: str
    metric: str
    value: float | None = None
    partition: str = Field(
        default="test",
        description="Target evaluation split partition (train, val, test).",
    )
    status: RunStatus = RunStatus.SUCCESS
    failure_metadata: FailureMetadata | None = None
    config_hash: str
    artifact_hash: str | None = None
    timestamp: str = Field(
        default_factory=lambda: datetime.now(UTC).isoformat(),
    )
    runtime_seconds: float = 0.0


class GraphLiftRecord(BaseModel):
    """Explicit matched graph lift comparison record.

    graph_lift = macro_f1(GNN) - macro_f1(matched_MLP)
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    dataset: str
    split_id: str
    seed: int
    graph_name: str
    gnm_model: str
    matched_mlp_model: str
    gnn_macro_f1: float
    matched_mlp_macro_f1: float
    graph_lift: float
    config_hash: str
    is_valid_match: bool = True
    notes: str | None = None


class TidyResultsCollection(BaseModel):
    """Collection of metric records with tabular export capability."""

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
            rows.append(
                {
                    "dataset": r.dataset,
                    "split_id": r.split_id,
                    "seed": r.seed,
                    "graph_name": r.graph_name,
                    "model": r.model,
                    "metric": r.metric,
                    "value": r.value,
                    "partition": r.partition,
                    "status": r.status.value,
                    "error_type": r.failure_metadata.error_type if r.failure_metadata else None,
                    "error_message": r.failure_metadata.error_message
                    if r.failure_metadata
                    else None,
                    "config_hash": r.config_hash,
                    "artifact_hash": r.artifact_hash,
                    "runtime_seconds": r.runtime_seconds,
                    "timestamp": r.timestamp,
                }
            )
        return pd.DataFrame(rows)
