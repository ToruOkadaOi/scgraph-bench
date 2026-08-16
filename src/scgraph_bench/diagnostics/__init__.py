"""Graph diagnostics suite: structural topology, post hoc homophily, and donor/site mixing."""

from scgraph_bench.diagnostics.homophily import compute_label_diagnostics
from scgraph_bench.diagnostics.metadata_mixing import compute_metadata_diagnostics
from scgraph_bench.diagnostics.runner import run_graph_diagnostics, save_diagnostics_report
from scgraph_bench.diagnostics.schema import (
    GraphDiagnosticsReport,
    LabelDiagnostics,
    MetadataDiagnostics,
    TopologyDiagnostics,
)
from scgraph_bench.diagnostics.topology import compute_topology_diagnostics

__all__ = [
    "compute_topology_diagnostics",
    "compute_label_diagnostics",
    "compute_metadata_diagnostics",
    "run_graph_diagnostics",
    "save_diagnostics_report",
    "TopologyDiagnostics",
    "LabelDiagnostics",
    "MetadataDiagnostics",
    "GraphDiagnosticsReport",
]
