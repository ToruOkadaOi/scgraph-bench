"""Post hoc label-aware diagnostics: homophily and neighborhood purity."""

from __future__ import annotations

import numpy as np

from scgraph_bench.diagnostics.schema import LabelDiagnostics
from scgraph_bench.graph.schema import GraphBundle


def compute_label_diagnostics(
    graph_bundle: GraphBundle,
    y_all: np.ndarray,
    label_names: list[str] | None = None,
) -> LabelDiagnostics:
    """Compute post hoc label-aware homophily and neighborhood purity metrics.

    Inviolable Rule: This function is strictly post hoc and must never be called
    during graph construction or model forward inference.

    Args:
        graph_bundle: Serialized or constructed GraphBundle.
        y_all: Full integer label vector of length N_total.
        label_names: Optional string label names mapping integer class indices.

    Returns:
        LabelDiagnostics container.
    """
    y = np.asarray(y_all, dtype=np.int64)
    n_nodes = graph_bundle.num_nodes
    if len(y) != n_nodes:
        raise ValueError(f"Label array length ({len(y)}) != num_nodes ({n_nodes})")

    edge_index_np = graph_bundle.edge_index.cpu().numpy()
    src = edge_index_np[0]
    dst = edge_index_np[1]
    n_edges = len(src)

    n_tr = graph_bundle.manifest.num_train_nodes
    n_va = graph_bundle.manifest.num_val_nodes
    val_start = n_tr
    val_end = n_tr + n_va
    test_start = n_tr + n_va
    test_end = n_nodes

    # 1. Overall edge homophily
    same_label_edges = (y[src] == y[dst]).astype(np.float32)
    overall_edge_homophily = float(np.mean(same_label_edges)) if n_edges > 0 else 0.0

    # 2. Train-train edge homophily
    mask_tr_tr = (src < val_start) & (dst < val_start)
    if np.any(mask_tr_tr):
        train_train_edge_homophily = float(np.mean(same_label_edges[mask_tr_tr]))
    else:
        train_train_edge_homophily = 0.0

    # 3. Node-level homophily (incoming neighborhood purity per node)
    node_same_counts = np.zeros(n_nodes, dtype=np.float32)
    node_in_degrees = np.bincount(dst, minlength=n_nodes).astype(np.float32)
    np.add.at(node_same_counts, dst, same_label_edges)

    non_isolated = node_in_degrees > 0
    node_homophily = np.zeros(n_nodes, dtype=np.float32)
    node_homophily[non_isolated] = node_same_counts[non_isolated] / node_in_degrees[non_isolated]

    overall_node_homophily = (
        float(np.mean(node_homophily[non_isolated])) if np.any(non_isolated) else 0.0
    )

    # Train-only node homophily
    tr_mask = np.zeros(n_nodes, dtype=bool)
    tr_mask[:val_start] = True
    tr_eval_mask = tr_mask & non_isolated
    train_train_node_homophily = (
        float(np.mean(node_homophily[tr_eval_mask])) if np.any(tr_eval_mask) else 0.0
    )

    # 4. Validation query -> Train reference homophily
    va_mask = np.zeros(n_nodes, dtype=bool)
    va_mask[val_start:val_end] = True
    va_eval_mask = va_mask & non_isolated
    val_to_train_query_homophily = (
        float(np.mean(node_homophily[va_eval_mask])) if np.any(va_eval_mask) else 0.0
    )

    # 5. Test query -> Train reference homophily
    te_mask = np.zeros(n_nodes, dtype=bool)
    te_mask[test_start:test_end] = True
    te_eval_mask = te_mask & non_isolated
    test_to_train_query_homophily = (
        float(np.mean(node_homophily[te_eval_mask])) if np.any(te_eval_mask) else 0.0
    )

    # 6. Expected class-composition random homophily baselines
    num_classes = int(np.max(y)) + 1
    counts_all = np.bincount(y, minlength=num_classes).astype(np.float64)
    p_all = counts_all / len(y)
    expected_random_homophily = float(np.sum(p_all**2))
    homophily_lift_over_random = float(overall_edge_homophily - expected_random_homophily)

    counts_tr = np.bincount(y[:val_start], minlength=num_classes).astype(np.float64)
    p_tr = counts_tr / len(y[:val_start]) if val_start > 0 else np.zeros(num_classes)

    counts_va = np.bincount(y[val_start:val_end], minlength=num_classes).astype(np.float64)
    p_va = counts_va / (val_end - val_start) if val_end > val_start else np.zeros(num_classes)

    counts_te = np.bincount(y[test_start:test_end], minlength=num_classes).astype(np.float64)
    p_te = counts_te / (test_end - test_start) if test_end > test_start else np.zeros(num_classes)

    expected_train_train_homophily = float(np.sum(p_tr**2))
    train_train_homophily_lift = float(train_train_edge_homophily - expected_train_train_homophily)

    expected_train_to_val_homophily = float(np.sum(p_tr * p_va))
    val_to_train_query_homophily_lift = float(
        val_to_train_query_homophily - expected_train_to_val_homophily
    )

    expected_train_to_test_homophily = float(np.sum(p_tr * p_te))
    test_to_train_query_homophily_lift = float(
        test_to_train_query_homophily - expected_train_to_test_homophily
    )

    # 7. Per-class neighborhood purity
    unique_classes = np.unique(y)
    per_class_purity: dict[str, float] = {}

    for c in unique_classes:
        c_int = int(c)
        c_name = (
            label_names[c_int]
            if label_names is not None and c_int < len(label_names)
            else f"class_{c_int}"
        )
        mask_c = (y == c_int) & non_isolated
        if np.any(mask_c):
            per_class_purity[c_name] = float(np.mean(node_homophily[mask_c]))
        else:
            per_class_purity[c_name] = 0.0

    macro_avg_purity = (
        float(np.mean(list(per_class_purity.values()))) if len(per_class_purity) > 0 else 0.0
    )

    return LabelDiagnostics(
        overall_edge_homophily=overall_edge_homophily,
        overall_node_homophily=overall_node_homophily,
        train_train_edge_homophily=train_train_edge_homophily,
        train_train_node_homophily=train_train_node_homophily,
        val_to_train_query_homophily=val_to_train_query_homophily,
        test_to_train_query_homophily=test_to_train_query_homophily,
        expected_random_homophily=expected_random_homophily,
        homophily_lift_over_random=homophily_lift_over_random,
        expected_train_train_homophily=expected_train_train_homophily,
        train_train_homophily_lift=train_train_homophily_lift,
        expected_train_to_val_homophily=expected_train_to_val_homophily,
        val_to_train_query_homophily_lift=val_to_train_query_homophily_lift,
        expected_train_to_test_homophily=expected_train_to_test_homophily,
        test_to_train_query_homophily_lift=test_to_train_query_homophily_lift,
        per_class_neighborhood_purity=per_class_purity,
        macro_average_class_purity=macro_avg_purity,
    )
