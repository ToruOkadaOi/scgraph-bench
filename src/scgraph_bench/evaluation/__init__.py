"""Evaluation engine for macro-F1, secondary metrics, and stratified breakdowns."""

from scgraph_bench.evaluation.metrics import (
    compute_evaluation_summary,
    confusion_matrix_to_dataframe,
)
from scgraph_bench.evaluation.schema import (
    EvaluationSummary,
    PerClassMetric,
    StratifiedDonorMetric,
    StratifiedSiteMetric,
)

__all__ = [
    "compute_evaluation_summary",
    "confusion_matrix_to_dataframe",
    "EvaluationSummary",
    "PerClassMetric",
    "StratifiedDonorMetric",
    "StratifiedSiteMetric",
]
