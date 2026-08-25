"""Schemas for confidence calibration summaries."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class CalibrationBin(BaseModel):
    """Single reliability-diagram bin aggregating prediction confidence and observed accuracy."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    bin_index: int
    bin_lower: float
    bin_upper: float
    count: int
    accuracy: float
    mean_confidence: float
    gap: float


class CalibrationSummary(BaseModel):
    """Run-level confidence and calibration aggregates computed from saved class probabilities."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    run_id: str = ""
    partition: str = "test"
    n_samples: int
    accuracy: float
    ece: float
    brier_score: float
    mean_max_confidence: float
    mean_entropy_nats: float
    mean_margin: float
    fraction_low_margin: float = Field(
        default=0.0,
        description="Fraction of samples with top1-top2 probability margin below 0.1.",
    )
    bins: list[CalibrationBin] = Field(default_factory=list)
