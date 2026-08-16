"""Structural and topological diagnostics for single-cell graphs."""

from __future__ import annotations

import numpy as np
from scipy import sparse
from scipy.sparse.csgraph import connected_components

from scgraph_bench.diagnostics.schema import TopologyDiagnostics
from scgraph_bench.graph.schema import GraphBundle


def compute_topology_diagnostics(graph_bundle: GraphBundle) -> TopologyDiagnostics:
    """Compute topology-only graph diagnostics.

    Evaluates node/edge counts, density, degree distributions, connected components,
    and partition-specific edge types without accessing cell-type labels.
    """
    n_nodes = graph_bundle.num_nodes
    edge_index_np = graph_bundle.edge_index.cpu().numpy()
    src = edge_index_np[0]
    dst = edge_index_np[1]
    n_edges = len(src)

    # 1. Degree distributions
    in_degrees = np.bincount(dst, minlength=n_nodes)
    out_degrees = np.bincount(src, minlength=n_nodes)
    total_degrees = in_degrees + out_degrees

    isolated_count = int(np.sum(total_degrees == 0))
    isolated_fraction = float(isolated_count / n_nodes) if n_nodes > 0 else 0.0

    # 2. Graph density
    max_possible_edges = n_nodes * (n_nodes - 1)
    density = float(n_edges / max_possible_edges) if max_possible_edges > 0 else 0.0

    # 3. Connected components (treating graph as undirected underlying structure)
    adj = sparse.csr_matrix(
        (np.ones(n_edges, dtype=np.int32), (src, dst)),
        shape=(n_nodes, n_nodes),
    )
    # Symmetrize adjacency for component analysis
    adj_sym = adj + adj.T
    n_components, labels = connected_components(adj_sym, directed=False)
    _, comp_counts = np.unique(labels, return_counts=True)
    largest_comp_size = int(np.max(comp_counts)) if len(comp_counts) > 0 else 0
    largest_comp_fraction = float(largest_comp_size / n_nodes) if n_nodes > 0 else 0.0

    # 4. Partition edge counts
    partition_edge_counts = {
        "train_to_train": graph_bundle.manifest.num_train_train_edges,
        "train_to_val": graph_bundle.manifest.num_train_to_val_edges,
        "train_to_test": graph_bundle.manifest.num_train_to_test_edges,
        "disallowed": graph_bundle.manifest.num_disallowed_edges,
    }

    # 5. Edge weight summary
    edge_weight_summary = None
    if graph_bundle.edge_weight is not None:
        weights = graph_bundle.edge_weight.cpu().numpy()
        edge_weight_summary = {
            "min": float(np.min(weights)),
            "max": float(np.max(weights)),
            "mean": float(np.mean(weights)),
            "median": float(np.median(weights)),
            "std": float(np.std(weights)),
        }

    return TopologyDiagnostics(
        num_nodes=n_nodes,
        num_edges=n_edges,
        density=density,
        in_degree_mean=float(np.mean(in_degrees)),
        in_degree_median=float(np.median(in_degrees)),
        in_degree_std=float(np.std(in_degrees)),
        in_degree_min=int(np.min(in_degrees)),
        in_degree_max=int(np.max(in_degrees)),
        out_degree_mean=float(np.mean(out_degrees)),
        out_degree_median=float(np.median(out_degrees)),
        out_degree_std=float(np.std(out_degrees)),
        out_degree_min=int(np.min(out_degrees)),
        out_degree_max=int(np.max(out_degrees)),
        isolated_node_count=isolated_count,
        isolated_node_fraction=isolated_fraction,
        num_connected_components=int(n_components),
        largest_component_size=largest_comp_size,
        largest_component_fraction=largest_comp_fraction,
        partition_edge_counts=partition_edge_counts,
        edge_weight_summary=edge_weight_summary,
    )
