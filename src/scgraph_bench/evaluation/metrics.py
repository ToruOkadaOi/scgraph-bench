"""Evaluation engine computing primary macro-F1, secondary metrics, and stratified breakdowns."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    precision_score,
    recall_score,
)

from scgraph_bench.evaluation.schema import (
    EvaluationSummary,
    PerClassMetric,
    StratifiedDonorMetric,
    StratifiedSiteMetric,
)


def compute_evaluation_summary(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    partition: str = "test",
    label_names: list[str] | None = None,
    donor_ids: list[str] | None = None,
    site_ids: list[str] | None = None,
) -> EvaluationSummary:
    """Compute comprehensive evaluation metrics for model predictions.

    Args:
        y_true: Ground truth integer labels (N).
        y_pred: Predicted integer labels (N).
        partition: Partition name (e.g. 'train', 'val', 'test').
        label_names: Optional string label names mapping integer class indices.
        donor_ids: Optional list of donor IDs for stratified breakdown.
        site_ids: Optional list of site IDs for stratified breakdown.

    Returns:
        EvaluationSummary object containing overall, per-class, per-donor, and per-site metrics.
    """
    y_t = np.asarray(y_true, dtype=np.int64)
    y_p = np.asarray(y_pred, dtype=np.int64)
    n_samples = len(y_t)

    if len(y_p) != n_samples:
        raise ValueError(f"y_pred length ({len(y_p)}) != y_true length ({n_samples})")

    # Determine unique classes from true labels and/or label_names
    num_classes = (
        len(label_names) if label_names is not None else int(max(np.max(y_t), np.max(y_p)) + 1)
    )
    all_classes = np.arange(num_classes)

    # 1. Global Metrics
    macro_f1 = float(f1_score(y_t, y_p, labels=all_classes, average="macro", zero_division=0.0))
    weighted_f1 = float(
        f1_score(y_t, y_p, labels=all_classes, average="weighted", zero_division=0.0)
    )
    balanced_acc = float(balanced_accuracy_score(y_t, y_p))
    overall_acc = float(accuracy_score(y_t, y_p))
    macro_prec = float(
        precision_score(y_t, y_p, labels=all_classes, average="macro", zero_division=0.0)
    )
    macro_rec = float(
        recall_score(y_t, y_p, labels=all_classes, average="macro", zero_division=0.0)
    )

    # 2. Per-Class Metrics
    prec_arr, rec_arr, f1_arr, supp_arr = precision_recall_fscore_support(
        y_t,
        y_p,
        labels=all_classes,
        zero_division=0.0,
    )

    per_class: list[PerClassMetric] = []
    for c in all_classes:
        c_int = int(c)
        c_name = (
            label_names[c_int]
            if label_names is not None and c_int < len(label_names)
            else f"class_{c_int}"
        )
        per_class.append(
            PerClassMetric(
                class_index=c_int,
                class_name=c_name,
                precision=float(prec_arr[c_int]),
                recall=float(rec_arr[c_int]),
                f1=float(f1_arr[c_int]),
                support=int(supp_arr[c_int]),
            )
        )

    # 3. Stratified Per-Donor Breakdown
    per_donor: list[StratifiedDonorMetric] = []
    if donor_ids is not None:
        donors_np = np.asarray(donor_ids, dtype=object)
        sites_np = (
            np.asarray(site_ids, dtype=object)
            if site_ids is not None
            else np.full(n_samples, "unknown")
        )

        unique_donors = np.unique(donors_np)
        for d in sorted(unique_donors):
            mask_d = donors_np == d
            if not np.any(mask_d):
                continue
            y_t_d = y_t[mask_d]
            y_p_d = y_p[mask_d]
            d_site = str(sites_np[mask_d][0])

            # Observed classes in this donor
            observed_classes_d = np.unique(y_t_d)
            obs_macro_f1 = float(
                f1_score(
                    y_t_d, y_p_d, labels=observed_classes_d, average="macro", zero_division=0.0
                )
            )
            global_macro_f1 = float(
                f1_score(y_t_d, y_p_d, labels=all_classes, average="macro", zero_division=0.0)
            )
            d_bal_acc = float(balanced_accuracy_score(y_t_d, y_p_d))

            present_names: list[str] = []
            absent_names: list[str] = []
            d_class_counts: dict[str, int] = {}

            for c in all_classes:
                c_int = int(c)
                c_name = (
                    label_names[c_int]
                    if label_names is not None and c_int < len(label_names)
                    else f"class_{c_int}"
                )
                cnt = int(np.sum(y_t_d == c_int))
                d_class_counts[c_name] = cnt
                if cnt > 0:
                    present_names.append(c_name)
                else:
                    absent_names.append(c_name)

            per_donor.append(
                StratifiedDonorMetric(
                    donor_id=str(d),
                    site=d_site,
                    observed_class_macro_f1=obs_macro_f1,
                    global_label_macro_f1=global_macro_f1,
                    balanced_accuracy=d_bal_acc,
                    support=int(np.sum(mask_d)),
                    present_classes=present_names,
                    absent_classes=absent_names,
                    class_support=d_class_counts,
                )
            )

    # 4. Stratified Per-Site Breakdown (e.g. Cambridge vs Newcastle)
    per_site: list[StratifiedSiteMetric] = []
    if site_ids is not None:
        sites_np = np.asarray(site_ids, dtype=object)
        unique_sites = np.unique(sites_np)

        for s in sorted(unique_sites):
            mask_s = sites_np == s
            if not np.any(mask_s):
                continue
            y_t_s = y_t[mask_s]
            y_p_s = y_p[mask_s]

            observed_classes_s = np.unique(y_t_s)
            obs_s_macro_f1 = float(
                f1_score(
                    y_t_s, y_p_s, labels=observed_classes_s, average="macro", zero_division=0.0
                )
            )
            global_s_macro_f1 = float(
                f1_score(y_t_s, y_p_s, labels=all_classes, average="macro", zero_division=0.0)
            )
            s_weighted_f1 = float(
                f1_score(y_t_s, y_p_s, labels=all_classes, average="weighted", zero_division=0.0)
            )
            s_bal_acc = float(balanced_accuracy_score(y_t_s, y_p_s))
            s_acc = float(accuracy_score(y_t_s, y_p_s))

            s_present_names: list[str] = []
            s_absent_names: list[str] = []
            s_class_counts: dict[str, int] = {}

            for c in all_classes:
                c_int = int(c)
                c_name = (
                    label_names[c_int]
                    if label_names is not None and c_int < len(label_names)
                    else f"class_{c_int}"
                )
                cnt = int(np.sum(y_t_s == c_int))
                s_class_counts[c_name] = cnt
                if cnt > 0:
                    s_present_names.append(c_name)
                else:
                    s_absent_names.append(c_name)

            per_site.append(
                StratifiedSiteMetric(
                    site=str(s),
                    observed_class_macro_f1=obs_s_macro_f1,
                    global_label_macro_f1=global_s_macro_f1,
                    weighted_f1=s_weighted_f1,
                    balanced_accuracy=s_bal_acc,
                    overall_accuracy=s_acc,
                    support=int(np.sum(mask_s)),
                    present_classes=s_present_names,
                    absent_classes=s_absent_names,
                    class_support=s_class_counts,
                )
            )

    # 5. Confusion Matrix
    cm = confusion_matrix(y_t, y_p, labels=all_classes)
    cm_list = cm.tolist()

    return EvaluationSummary(
        partition=partition,
        num_samples=n_samples,
        macro_f1=macro_f1,
        weighted_f1=weighted_f1,
        balanced_accuracy=balanced_acc,
        overall_accuracy=overall_acc,
        macro_precision=macro_prec,
        macro_recall=macro_rec,
        per_class=per_class,
        per_donor=per_donor,
        per_site=per_site,
        confusion_matrix=cm_list,
    )


def confusion_matrix_to_dataframe(
    cm: list[list[int]],
    label_names: list[str],
) -> pd.DataFrame:
    """Convert confusion matrix to pandas DataFrame with annotated row/column class headers."""
    return pd.DataFrame(
        cm,
        index=[f"True_{lbl}" for lbl in label_names],
        columns=[f"Pred_{lbl}" for lbl in label_names],
    )
