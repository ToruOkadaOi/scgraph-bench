"""Degree-preserving randomized rewiring negative control graph builder."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch

from scgraph_bench.config.graph import RewiredControlConfig
from scgraph_bench.graph.base import BaseGraphBuilder
from scgraph_bench.graph.pca_knn import PCAkNNGraphBuilder
from scgraph_bench.graph.registry import register_graph_builder
from scgraph_bench.graph.schema import GraphBundle, GraphManifest
from scgraph_bench.utils.hashing import hash_array
from scgraph_bench.utils.logging import get_logger

logger = get_logger("graph.rewired_control")


def _rewire_edge_array(
    edges: np.ndarray,
    n_swaps: int,
    rng: np.random.Generator,
    allow_self_loops: bool = False,
) -> np.ndarray:
    """Perform partition-isolated degree-preserving double edge swaps on a directed edge array.

    Args:
        edges: (M x 2) numpy array of directed edges [src, dst].
        n_swaps: Number of successful swaps to perform.
        rng: NumPy random generator.
        allow_self_loops: Whether to allow u == v.

    Returns:
        Rewired (M x 2) edge array with identical in-degrees and out-degrees.
    """
    m = len(edges)
    if m < 2:
        return edges.copy()

    edge_list = [tuple(e) for e in edges]
    edge_set = set(edge_list)

    swaps_done = 0
    max_attempts = n_swaps * 10
    attempts = 0

    while swaps_done < n_swaps and attempts < max_attempts:
        attempts += 1
        idx1, idx2 = rng.integers(0, m, size=2)
        if idx1 == idx2:
            continue

        u1, v1 = edge_list[idx1]
        u2, v2 = edge_list[idx2]

        if u1 == u2 or v1 == v2:
            continue

        new_e1 = (u1, v2)
        new_e2 = (u2, v1)

        if not allow_self_loops and (new_e1[0] == new_e1[1] or new_e2[0] == new_e2[1]):
            continue

        if new_e1 in edge_set or new_e2 in edge_set:
            continue

        # Valid swap: apply
        edge_set.remove((u1, v1))
        edge_set.remove((u2, v2))
        edge_set.add(new_e1)
        edge_set.add(new_e2)

        edge_list[idx1] = new_e1
        edge_list[idx2] = new_e2
        swaps_done += 1

    logger.info(
        "Completed %d degree-preserving swaps (out of %d requested, %d attempts)",
        swaps_done,
        n_swaps,
        attempts,
    )
    return np.array(edge_list, dtype=np.int64)


@register_graph_builder("rewired_control")
class RewiredControlGraphBuilder(BaseGraphBuilder):
    """Constructs degree-preserving randomized negative control graphs."""

    def __init__(self, config: RewiredControlConfig | None = None) -> None:
        self.config = config or RewiredControlConfig()

    def build(
        self,
        X_pca_train: np.ndarray,
        X_pca_val: np.ndarray,
        X_pca_test: np.ndarray,
        train_cell_ids: list[str],
        val_cell_ids: list[str],
        test_cell_ids: list[str],
        feature_manifest_hash: str,
        dataset_name: str,
        split_id: str,
        allowed_metadata: dict[str, Any] | None = None,
    ) -> GraphBundle:
        """Construct degree-preserving rewired control graph from reference PCA-kNN."""
        # 1. Build base reference PCA-kNN graph
        base_builder = PCAkNNGraphBuilder()
        base_bundle = base_builder.build(
            X_pca_train=X_pca_train,
            X_pca_val=X_pca_val,
            X_pca_test=X_pca_test,
            train_cell_ids=train_cell_ids,
            val_cell_ids=val_cell_ids,
            test_cell_ids=test_cell_ids,
            feature_manifest_hash=feature_manifest_hash,
            dataset_name=dataset_name,
            split_id=split_id,
            allowed_metadata=allowed_metadata,
        )

        n_tr = len(train_cell_ids)
        n_va = len(val_cell_ids)
        n_te = len(test_cell_ids)
        n_total = n_tr + n_va + n_te

        val_start = n_tr
        val_end = n_tr + n_va
        test_start = n_tr + n_va
        test_end = n_total

        base_edge_index = base_bundle.edge_index.numpy()
        srcs = base_edge_index[0]
        dsts = base_edge_index[1]

        # 2. Partition base edges
        tr_tr_mask = (srcs < val_start) & (dsts < val_start)
        tr_va_mask = (srcs < val_start) & (dsts >= val_start) & (dsts < val_end)
        tr_te_mask = (srcs < val_start) & (dsts >= test_start) & (dsts < test_end)

        tr_tr_edges = np.column_stack((srcs[tr_tr_mask], dsts[tr_tr_mask]))
        tr_va_edges = np.column_stack((srcs[tr_va_mask], dsts[tr_va_mask]))
        tr_te_edges = np.column_stack((srcs[tr_te_mask], dsts[tr_te_mask]))

        rng = np.random.default_rng(self.config.seed)
        factor = self.config.n_swaps_factor

        # 3. Rewire each partition preserving degree sequence and partition boundaries
        rewired_tr_tr = _rewire_edge_array(
            tr_tr_edges,
            n_swaps=int(len(tr_tr_edges) * factor),
            rng=rng,
            allow_self_loops=False,
        )
        rewired_tr_va = _rewire_edge_array(
            tr_va_edges,
            n_swaps=int(len(tr_va_edges) * factor),
            rng=rng,
            allow_self_loops=False,
        )
        rewired_tr_te = _rewire_edge_array(
            tr_te_edges,
            n_swaps=int(len(tr_te_edges) * factor),
            rng=rng,
            allow_self_loops=False,
        )

        all_rewired_edges = np.vstack((rewired_tr_tr, rewired_tr_va, rewired_tr_te))
        all_rewired_edges = all_rewired_edges[
            np.lexsort((all_rewired_edges[:, 1], all_rewired_edges[:, 0]))
        ]

        src_rewired = all_rewired_edges[:, 0]
        dst_rewired = all_rewired_edges[:, 1]

        edge_index_np = np.array([src_rewired, dst_rewired], dtype=np.int64)
        edge_index = torch.from_numpy(edge_index_np)
        edge_index_hash = hash_array(edge_index_np)

        # 4. Verify degree preservation against base graph
        base_in_deg = np.bincount(dsts, minlength=n_total)
        rewired_in_deg = np.bincount(dst_rewired, minlength=n_total)
        assert np.array_equal(base_in_deg, rewired_in_deg), "Rewiring altered in-degree sequence!"

        base_out_deg = np.bincount(srcs, minlength=n_total)
        rewired_out_deg = np.bincount(src_rewired, minlength=n_total)
        assert np.array_equal(base_out_deg, rewired_out_deg), (
            "Rewiring altered out-degree sequence!"
        )

        manifest = GraphManifest(
            graph_name=f"rewired_control_pca_knn_seed{self.config.seed}",
            builder_type="rewired_control",
            dataset_name=dataset_name,
            split_id=split_id,
            k=base_bundle.manifest.k,
            nominal_query_k=base_bundle.manifest.nominal_query_k,
            nominal_train_k=base_bundle.manifest.nominal_train_k,
            reference_edge_policy="rewired_double_edge_swap",
            query_edge_policy="rewired_double_edge_swap",
            metric=base_bundle.manifest.metric,
            weighting="unweighted",
            sigma_k=None,
            edge_index_convention="source_to_target",
            message_flow_train="train_to_train",
            message_flow_validation="train_to_validation",
            message_flow_test="train_to_test",
            query_nodes_affect_training_representations=False,
            num_nodes=n_total,
            num_edges=len(all_rewired_edges),
            num_train_nodes=n_tr,
            num_val_nodes=n_va,
            num_test_nodes=n_te,
            num_train_train_edges=len(rewired_tr_tr),
            num_train_to_val_edges=len(rewired_tr_va),
            num_train_to_test_edges=len(rewired_tr_te),
            num_disallowed_edges=0,
            edge_index_hash=edge_index_hash,
            edge_weight_hash=None,
            feature_manifest_hash=feature_manifest_hash,
        )

        return GraphBundle(
            edge_index=edge_index,
            num_nodes=n_total,
            train_mask=base_bundle.train_mask,
            val_mask=base_bundle.val_mask,
            test_mask=base_bundle.test_mask,
            node_cell_ids=base_bundle.node_cell_ids,
            manifest=manifest,
            edge_weight=None,
        )
