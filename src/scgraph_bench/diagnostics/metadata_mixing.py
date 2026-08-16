"""Donor and sequencing site mixing diagnostics."""

from __future__ import annotations

import numpy as np

from scgraph_bench.diagnostics.schema import MetadataDiagnostics
from scgraph_bench.graph.schema import GraphBundle


def _compute_entropy_per_node(
    dst_indices: np.ndarray,
    src_groups: np.ndarray,
    n_nodes: int,
) -> np.ndarray:
    """Compute Shannon entropy of source group distribution per target node."""
    entropies = np.zeros(n_nodes, dtype=np.float32)

    # Group incoming sources by target node
    order = np.argsort(dst_indices)
    sorted_dst = dst_indices[order]
    sorted_src_groups = src_groups[order]

    # Find boundaries for each target node
    split_indices = np.where(np.diff(sorted_dst))[0] + 1
    node_groups_list = np.split(sorted_src_groups, split_indices)
    unique_nodes = np.unique(sorted_dst)

    for node_idx, groups in zip(unique_nodes, node_groups_list, strict=False):
        if len(groups) == 0:
            continue
        _, counts = np.unique(groups, return_counts=True)
        probs = counts / float(len(groups))
        # Shannon entropy in bits (base 2)
        h = -np.sum(probs * np.log2(probs + 1e-12))
        entropies[node_idx] = float(h)

    return entropies


def compute_metadata_diagnostics(
    graph_bundle: GraphBundle,
    donor_ids: list[str],
    site_ids: list[str],
) -> MetadataDiagnostics:
    """Compute donor and site mixing metrics across partitions.

    Args:
        graph_bundle: Serialized or constructed GraphBundle.
        donor_ids: Ordered donor identifiers for all N_total nodes.
        site_ids: Ordered site identifiers for all N_total nodes.

    Returns:
        MetadataDiagnostics container.
    """
    n_nodes = graph_bundle.num_nodes
    if len(donor_ids) != n_nodes or len(site_ids) != n_nodes:
        raise ValueError("Metadata length != graph num_nodes")

    donors = np.asarray(donor_ids, dtype=object)
    sites = np.asarray(site_ids, dtype=object)

    edge_index_np = graph_bundle.edge_index.cpu().numpy()
    src = edge_index_np[0]
    dst = edge_index_np[1]

    n_tr = graph_bundle.manifest.num_train_nodes
    n_va = graph_bundle.manifest.num_val_nodes
    val_start = n_tr
    val_end = n_tr + n_va
    test_start = n_tr + n_va
    test_end = n_nodes

    # 1. Intra-donor and intra-site fractions in Train -> Train
    mask_tr_tr = (src < val_start) & (dst < val_start)
    if np.any(mask_tr_tr):
        src_tr = src[mask_tr_tr]
        dst_tr = dst[mask_tr_tr]
        train_intra_donor = float(np.mean(donors[src_tr] == donors[dst_tr]))
        train_intra_site = float(np.mean(sites[src_tr] == sites[dst_tr]))
    else:
        train_intra_donor, train_intra_site = 0.0, 0.0

    # 2. Site match in Train -> Val and Train -> Test
    mask_tr_va = (src < val_start) & (dst >= val_start) & (dst < val_end)
    if np.any(mask_tr_va):
        val_site_match = float(np.mean(sites[src[mask_tr_va]] == sites[dst[mask_tr_va]]))
    else:
        val_site_match = 0.0

    mask_tr_te = (src < val_start) & (dst >= test_start) & (dst < test_end)
    if np.any(mask_tr_te):
        test_site_match = float(np.mean(sites[src[mask_tr_te]] == sites[dst[mask_tr_te]]))
    else:
        test_site_match = 0.0

    # 3. Neighborhood entropies
    donor_entropies = _compute_entropy_per_node(dst, donors[src], n_nodes)
    site_entropies = _compute_entropy_per_node(dst, sites[src], n_nodes)

    mean_tr_donor_entropy = float(np.mean(donor_entropies[:val_start]))
    mean_tr_site_entropy = float(np.mean(site_entropies[:val_start]))
    mean_va_donor_entropy = float(np.mean(donor_entropies[val_start:val_end])) if n_va > 0 else 0.0
    mean_te_donor_entropy = (
        float(np.mean(donor_entropies[test_start:test_end])) if (test_end - test_start) > 0 else 0.0
    )

    return MetadataDiagnostics(
        train_intra_donor_edge_fraction=train_intra_donor,
        train_intra_site_edge_fraction=train_intra_site,
        val_to_train_site_match_fraction=val_site_match,
        test_to_train_site_match_fraction=test_site_match,
        mean_train_donor_entropy=mean_tr_donor_entropy,
        mean_train_site_entropy=mean_tr_site_entropy,
        mean_val_query_donor_entropy=mean_va_donor_entropy,
        mean_test_query_donor_entropy=mean_te_donor_entropy,
    )
