"""Runner for end-to-end graph diagnostics suite."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import numpy as np

from scgraph_bench.diagnostics.homophily import compute_label_diagnostics
from scgraph_bench.diagnostics.metadata_mixing import compute_metadata_diagnostics
from scgraph_bench.diagnostics.schema import (
    GraphDiagnosticsReport,
)
from scgraph_bench.diagnostics.topology import compute_topology_diagnostics
from scgraph_bench.graph.schema import GraphBundle
from scgraph_bench.utils.logging import get_logger

logger = get_logger("diagnostics.runner")


def run_graph_diagnostics(
    graph_bundle: GraphBundle,
    y_all: np.ndarray | None = None,
    label_names: list[str] | None = None,
    donor_ids: list[str] | None = None,
    site_ids: list[str] | None = None,
    label_policy_hash: str = "",
) -> GraphDiagnosticsReport:
    """Execute complete graph diagnostics suite.

    Args:
        graph_bundle: Constructed/loaded GraphBundle.
        y_all: Optional complete label array (for post hoc label diagnostics).
        label_names: Optional class label names.
        donor_ids: Optional donor identifiers.
        site_ids: Optional site identifiers.
        label_policy_hash: Optional label policy hash.

    Returns:
        GraphDiagnosticsReport instance.
    """
    logger.info(
        "Computing topology diagnostics for graph '%s'...", graph_bundle.manifest.graph_name
    )
    topology = compute_topology_diagnostics(graph_bundle)

    label_diag = None
    if y_all is not None:
        logger.info("Computing post hoc label homophily and class purity...")
        label_diag = compute_label_diagnostics(graph_bundle, y_all=y_all, label_names=label_names)

    meta_diag = None
    if donor_ids is not None and site_ids is not None:
        logger.info("Computing donor and site mixing diagnostics...")
        meta_diag = compute_metadata_diagnostics(
            graph_bundle, donor_ids=donor_ids, site_ids=site_ids
        )

    return GraphDiagnosticsReport(
        graph_name=graph_bundle.manifest.graph_name,
        dataset_name=graph_bundle.manifest.dataset_name,
        split_id=graph_bundle.manifest.split_id,
        graph_manifest_hash=graph_bundle.manifest.compute_manifest_hash(),
        edge_index_hash=graph_bundle.manifest.edge_index_hash,
        feature_manifest_hash=graph_bundle.manifest.feature_manifest_hash,
        label_policy_hash=label_policy_hash,
        topology=topology,
        label_diagnostics=label_diag,
        metadata_diagnostics=meta_diag,
    )


def save_diagnostics_report(report: GraphDiagnosticsReport, output_dir: Path | str) -> None:
    """Save diagnostics report as structured JSON and flattened summary CSV."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # 1. Save complete JSON report
    (out / "graph_diagnostics.json").write_text(report.model_dump_json(indent=2), encoding="utf-8")

    # 2. Save flat key-value summary CSV
    flat_data: list[tuple[str, Any]] = [
        ("graph_name", report.graph_name),
        ("dataset_name", report.dataset_name),
        ("split_id", report.split_id),
        ("graph_manifest_hash", report.graph_manifest_hash),
        ("edge_index_hash", report.edge_index_hash),
        ("feature_manifest_hash", report.feature_manifest_hash),
        ("num_nodes", report.topology.num_nodes),
        ("num_edges", report.topology.num_edges),
        ("density", report.topology.density),
        ("in_degree_mean", report.topology.in_degree_mean),
        ("out_degree_mean", report.topology.out_degree_mean),
        ("isolated_node_count", report.topology.isolated_node_count),
        ("num_connected_components", report.topology.num_connected_components),
        ("largest_component_fraction", report.topology.largest_component_fraction),
        ("train_to_train_edges", report.topology.partition_edge_counts.get("train_to_train", 0)),
        ("train_to_val_edges", report.topology.partition_edge_counts.get("train_to_val", 0)),
        ("train_to_test_edges", report.topology.partition_edge_counts.get("train_to_test", 0)),
        ("disallowed_edges", report.topology.partition_edge_counts.get("disallowed", 0)),
    ]

    if report.label_diagnostics is not None:
        ld = report.label_diagnostics
        flat_data.extend(
            [
                ("overall_edge_homophily", ld.overall_edge_homophily),
                ("overall_node_homophily", ld.overall_node_homophily),
                ("train_train_edge_homophily", ld.train_train_edge_homophily),
                ("val_to_train_query_homophily", ld.val_to_train_query_homophily),
                ("test_to_train_query_homophily", ld.test_to_train_query_homophily),
                ("macro_average_class_purity", ld.macro_average_class_purity),
            ]
        )

    if report.metadata_diagnostics is not None:
        md = report.metadata_diagnostics
        flat_data.extend(
            [
                ("train_intra_donor_edge_fraction", md.train_intra_donor_edge_fraction),
                ("train_intra_site_edge_fraction", md.train_intra_site_edge_fraction),
                ("val_to_train_site_match_fraction", md.val_to_train_site_match_fraction),
                ("test_to_train_site_match_fraction", md.test_to_train_site_match_fraction),
                ("mean_train_donor_entropy", md.mean_train_donor_entropy),
                ("mean_train_site_entropy", md.mean_train_site_entropy),
            ]
        )

    csv_path = out / "graph_diagnostics_summary.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["metric_name", "value"])
        for k, v in flat_data:
            writer.writerow([k, v])
