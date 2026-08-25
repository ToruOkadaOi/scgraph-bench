"""Post hoc analysis utilities for benchmark results, confidence calibration, and embeddings."""

from scgraph_bench.analysis.calibration import (
    expected_calibration_error,
    multiclass_brier_score,
    prediction_entropy,
    reliability_diagram_data,
    summarize_confidence,
)
from scgraph_bench.analysis.embedding_quality import (
    EmbeddingQualityReport,
    compare_representations,
    compute_embedding_quality,
)
from scgraph_bench.analysis.flatten import (
    RunRecord,
    compute_matched_per_class_deltas,
    describe_run,
    discover_run_records,
    flatten_per_class,
    flatten_per_donor,
)
from scgraph_bench.analysis.schema import CalibrationBin, CalibrationSummary

__all__ = [
    "CalibrationBin",
    "CalibrationSummary",
    "EmbeddingQualityReport",
    "RunRecord",
    "compare_representations",
    "compute_embedding_quality",
    "compute_matched_per_class_deltas",
    "describe_run",
    "discover_run_records",
    "expected_calibration_error",
    "flatten_per_class",
    "flatten_per_donor",
    "multiclass_brier_score",
    "prediction_entropy",
    "reliability_diagram_data",
    "summarize_confidence",
]
