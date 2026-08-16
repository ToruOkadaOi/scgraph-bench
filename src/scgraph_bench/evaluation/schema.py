"""Schemas for baseline and model evaluation metrics and stratified breakdowns."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class PerClassMetric(BaseModel):
    """Detailed performance metrics for a single cell-type class."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    class_index: int
    class_name: str
    precision: float
    recall: float
    f1: float
    support: int


class StratifiedDonorMetric(BaseModel):
    """Per-donor evaluation breakdown with explicit observed vs global label distinction."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    donor_id: str
    site: str
    observed_class_macro_f1: float
    global_label_macro_f1: float
    balanced_accuracy: float
    support: int
    present_classes: list[str] = Field(default_factory=list)
    absent_classes: list[str] = Field(default_factory=list)
    class_support: dict[str, int] = Field(default_factory=dict)


class StratifiedSiteMetric(BaseModel):
    """Per-site (e.g. Cambridge vs Newcastle) evaluation breakdown."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    site: str
    observed_class_macro_f1: float
    global_label_macro_f1: float
    weighted_f1: float
    balanced_accuracy: float
    overall_accuracy: float
    support: int
    present_classes: list[str] = Field(default_factory=list)
    absent_classes: list[str] = Field(default_factory=list)
    class_support: dict[str, int] = Field(default_factory=dict)


class EvaluationSummary(BaseModel):
    """Comprehensive performance report across standard, per-class, and stratified dimensions."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    partition: str
    num_samples: int
    macro_f1: float
    weighted_f1: float
    balanced_accuracy: float
    overall_accuracy: float
    macro_precision: float
    macro_recall: float
    per_class: list[PerClassMetric]
    per_donor: list[StratifiedDonorMetric] = Field(default_factory=list)
    per_site: list[StratifiedSiteMetric] = Field(default_factory=list)
    confusion_matrix: list[list[int]] = Field(default_factory=list)
