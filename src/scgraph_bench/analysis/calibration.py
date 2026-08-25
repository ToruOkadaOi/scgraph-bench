"""Confidence and calibration metrics computed from saved class probabilities.

All functions operate on full probability matrices and are therefore retroactively
applicable to any historical run that persisted ``test_probs.npy``.
"""

from __future__ import annotations

import numpy as np

from scgraph_bench.analysis.schema import CalibrationBin, CalibrationSummary


def _validate_inputs(y_true: np.ndarray, probs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    y = np.asarray(y_true, dtype=np.int64)
    p = np.asarray(probs, dtype=np.float64)
    if p.ndim != 2 or p.shape[0] != y.shape[0]:
        raise ValueError(
            f"probs must be (n_samples, n_classes) matching y_true length {y.shape[0]}; "
            f"got shape {p.shape}"
        )
    if np.any(p < 0.0) or np.any(p > 1.0):
        raise ValueError("probs must lie within [0, 1]")
    return y, p


def expected_calibration_error(
    y_true: np.ndarray,
    probs: np.ndarray,
    n_bins: int = 15,
) -> float:
    """Top-label Expected Calibration Error: sum_b (n_b / N) * |acc_b - conf_b|."""
    y, p = _validate_inputs(y_true, probs)
    if len(y) == 0:
        return 0.0
    confidence = p.max(axis=1)
    predictions = p.argmax(axis=1)
    correct = (predictions == y).astype(np.float64)

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_ids = np.digitize(confidence, bin_edges[1:-1], right=False)
    ece = 0.0
    for b in range(n_bins):
        mask = bin_ids == b
        count = int(mask.sum())
        if count == 0:
            continue
        acc_b = float(correct[mask].mean())
        conf_b = float(confidence[mask].mean())
        ece += (count / len(y)) * abs(acc_b - conf_b)
    return float(ece)


def multiclass_brier_score(y_true: np.ndarray, probs: np.ndarray) -> float:
    """Multiclass Brier score: mean over samples of sum_c (p_c - onehot_c)^2."""
    y, p = _validate_inputs(y_true, probs)
    if len(y) == 0:
        return 0.0
    n_classes = p.shape[1]
    onehot = np.zeros_like(p)
    onehot[np.arange(len(y)), np.clip(y, 0, n_classes - 1)] = 1.0
    return float(np.mean(np.sum((p - onehot) ** 2, axis=1)))


def max_probability_confidence(probs: np.ndarray) -> np.ndarray:
    """Per-sample maximum class probability."""
    return np.asarray(probs, dtype=np.float64).max(axis=1)


def prediction_entropy(probs: np.ndarray) -> np.ndarray:
    """Per-sample Shannon entropy of the predicted distribution, in nats."""
    p = np.asarray(probs, dtype=np.float64)
    safe = np.clip(p, np.finfo(np.float64).eps, 1.0)
    return -(safe * np.log(safe)).sum(axis=1)


def confidence_margin(probs: np.ndarray) -> np.ndarray:
    """Per-sample margin: top-1 probability minus top-2 probability."""
    p = np.asarray(probs, dtype=np.float64)
    if p.shape[1] < 2:
        return np.ones(len(p), dtype=np.float64)
    top2 = np.sort(p, axis=1)[:, -2:]
    return top2[:, 1] - top2[:, 0]


def reliability_diagram_data(
    y_true: np.ndarray,
    probs: np.ndarray,
    n_bins: int = 15,
) -> list[CalibrationBin]:
    """Aggregate predictions into reliability-diagram bins sorted by confidence."""
    y, p = _validate_inputs(y_true, probs)
    confidence = p.max(axis=1)
    correct = (p.argmax(axis=1) == y).astype(np.float64)

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_ids = np.digitize(confidence, bin_edges[1:-1], right=False)

    bins: list[CalibrationBin] = []
    for b in range(n_bins):
        mask = bin_ids == b
        count = int(mask.sum())
        acc_b = float(correct[mask].mean()) if count else 0.0
        conf_b = (
            float(confidence[mask].mean()) if count else 0.5 * (bin_edges[b] + bin_edges[b + 1])
        )
        bins.append(
            CalibrationBin(
                bin_index=b,
                bin_lower=float(bin_edges[b]),
                bin_upper=float(bin_edges[b + 1]),
                count=count,
                accuracy=acc_b,
                mean_confidence=conf_b,
                gap=(acc_b - conf_b) if count else 0.0,
            )
        )
    return bins


def summarize_confidence(
    y_true: np.ndarray,
    probs: np.ndarray,
    run_id: str = "",
    partition: str = "test",
    n_bins: int = 15,
    low_margin_threshold: float = 0.1,
) -> CalibrationSummary:
    """Compute the full calibration summary for a set of predictions."""
    y, p = _validate_inputs(y_true, probs)
    margins = confidence_margin(p)
    accuracy = float((p.argmax(axis=1) == y).mean()) if len(y) else 0.0
    return CalibrationSummary(
        run_id=run_id,
        partition=partition,
        n_samples=int(len(y)),
        accuracy=accuracy,
        ece=expected_calibration_error(y, p, n_bins=n_bins),
        brier_score=multiclass_brier_score(y, p),
        mean_max_confidence=float(max_probability_confidence(p).mean()) if len(y) else 0.0,
        mean_entropy_nats=float(prediction_entropy(p).mean()) if len(y) else 0.0,
        mean_margin=float(margins.mean()) if len(y) else 0.0,
        fraction_low_margin=float((margins < low_margin_threshold).mean()) if len(y) else 0.0,
        bins=reliability_diagram_data(y, p, n_bins=n_bins),
    )
