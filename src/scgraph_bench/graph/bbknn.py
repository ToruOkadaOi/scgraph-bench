"""Strict-inductive Batch-Balanced k-Nearest Neighbors (BBKNN) graph construction."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from sklearn.neighbors import NearestNeighbors

from scgraph_bench.config.graph import BBKNNConfig
from scgraph_bench.graph.base import BaseGraphBuilder
from scgraph_bench.graph.registry import register_graph_builder
from scgraph_bench.graph.schema import GraphBundle, GraphManifest
from scgraph_bench.utils.hashing import hash_array
from scgraph_bench.utils.logging import get_logger

logger = get_logger("graph.bbknn")


@register_graph_builder("bbknn")
class StrictInductiveBBKNNGraphBuilder(BaseGraphBuilder):
    """Constructs strict-inductive Batch-Balanced k-Nearest Neighbors (BBKNN) graphs.

    Semantics:
    - Balanced Training Reference: Each cell connects to k_per_batch nearest neighbors within EACH training donor.
    - Balanced Query Projection: Each validation and test cell receives incoming directed edges from
      k_per_batch nearest neighbors within EACH training donor partition.
    - Zero query -> train edges, zero query -> query edges.
    """

    def __init__(self, config: BBKNNConfig | None = None) -> None:
        self.config = config or BBKNNConfig()

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
        """Construct strict-inductive BBKNN graph."""
        if allowed_metadata is None or "donor_ids_train" not in allowed_metadata:
            raise ValueError(
                "BBKNN requires 'donor_ids_train' in allowed_metadata for batch identification."
            )

        donor_ids_tr = np.asarray(allowed_metadata["donor_ids_train"])
        unique_donors = sorted(np.unique(donor_ids_tr))
        n_donors = len(unique_donors)

        n_tr = len(train_cell_ids)
        n_va = len(val_cell_ids)
        n_te = len(test_cell_ids)
        n_total = n_tr + n_va + n_te

        k_b = self.config.k_per_batch
        metric = self.config.metric

        logger.info(
            "Building strict-inductive BBKNN graph (k_per_batch=%d across %d donors, metric='%s') "
            "for %d total nodes (%d train, %d val, %d test)...",
            k_b,
            n_donors,
            metric,
            n_total,
            n_tr,
            n_va,
            n_te,
        )

        # 1. Fit NearestNeighbors index for each training donor separately
        donor_models: dict[str, NearestNeighbors] = {}
        donor_train_indices: dict[str, np.ndarray] = {}

        for d in unique_donors:
            mask_d = donor_ids_tr == d
            indices_d = np.where(mask_d)[0]
            donor_train_indices[d] = indices_d

            X_d = X_pca_train[mask_d]
            # Need at least k_b + 1 neighbors or as many as available
            n_avail = len(indices_d)
            n_k = min(k_b + 1, n_avail)

            nn_d = NearestNeighbors(n_neighbors=n_k, metric=metric, algorithm="auto")
            nn_d.fit(X_d)
            donor_models[d] = nn_d

        # 2. Build Train -> Train batch-balanced edges
        train_train_edges: set[tuple[int, int]] = set()

        for d, nn_d in donor_models.items():
            indices_d = donor_train_indices[d]
            # Query all training cells against donor d
            n_req = min(k_b + 1, len(indices_d))
            _, d_knn_indices = nn_d.kneighbors(X_pca_train, n_neighbors=n_req)

            for i in range(n_tr):
                donor_of_i = donor_ids_tr[i]
                retrieved_local = d_knn_indices[i]

                if donor_of_i == d:
                    # Exclude self if querying own donor
                    valid_local = [idx for idx in retrieved_local if indices_d[idx] != i][:k_b]
                else:
                    valid_local = retrieved_local[:k_b].tolist()

                for loc_idx in valid_local:
                    global_j = int(indices_d[loc_idx])
                    train_train_edges.add((global_j, i))
                    train_train_edges.add((i, global_j))

        # 3. Build Train -> Validation query edges (source in train, target in val)
        train_to_val_edges: set[tuple[int, int]] = set()
        for d, nn_d in donor_models.items():
            indices_d = donor_train_indices[d]
            n_req = min(k_b, len(indices_d))
            _, val_knn_indices = nn_d.kneighbors(X_pca_val, n_neighbors=n_req)

            for u in range(n_va):
                target_val_idx = n_tr + u
                for loc_idx in val_knn_indices[u]:
                    source_train_idx = int(indices_d[loc_idx])
                    train_to_val_edges.add((source_train_idx, target_val_idx))

        # 4. Build Train -> Test query edges (source in train, target in test)
        train_to_test_edges: set[tuple[int, int]] = set()
        for d, nn_d in donor_models.items():
            indices_d = donor_train_indices[d]
            n_req = min(k_b, len(indices_d))
            _, test_knn_indices = nn_d.kneighbors(X_pca_test, n_neighbors=n_req)

            for w in range(n_te):
                target_test_idx = n_tr + n_va + w
                for loc_idx in test_knn_indices[w]:
                    source_train_idx = int(indices_d[loc_idx])
                    train_to_test_edges.add((source_train_idx, target_test_idx))

        # 5. Assemble all edges (source -> target order)
        all_edges = sorted(train_train_edges | train_to_val_edges | train_to_test_edges)
        src_nodes = [e[0] for e in all_edges]
        dst_nodes = [e[1] for e in all_edges]

        edge_index_np = np.array([src_nodes, dst_nodes], dtype=np.int64)
        edge_index = torch.from_numpy(edge_index_np)

        # 6. Verify strict inductive constraints and validate message flow
        val_start = n_tr
        val_end = n_tr + n_va
        test_start = n_tr + n_va
        test_end = n_total

        num_disallowed = 0
        num_tr_tr = 0
        num_tr_to_va = 0
        num_tr_to_te = 0

        for src, dst in all_edges:
            is_src_tr = src < val_start
            is_dst_tr = dst < val_start
            is_dst_va = val_start <= dst < val_end
            is_dst_te = test_start <= dst < test_end

            if is_src_tr and is_dst_tr:
                num_tr_tr += 1
            elif is_src_tr and is_dst_va:
                num_tr_to_va += 1
            elif is_src_tr and is_dst_te:
                num_tr_to_te += 1
            else:
                num_disallowed += 1

        if num_disallowed > 0:
            raise ValueError(
                f"Strict inductive violation: found {num_disallowed} disallowed edges!"
            )

        # 7. Construct masks and manifests
        train_mask = torch.zeros(n_total, dtype=torch.bool)
        val_mask = torch.zeros(n_total, dtype=torch.bool)
        test_mask = torch.zeros(n_total, dtype=torch.bool)

        train_mask[:n_tr] = True
        val_mask[val_start:val_end] = True
        test_mask[test_start:test_end] = True

        all_node_cell_ids = train_cell_ids + val_cell_ids + test_cell_ids
        edge_index_hash = hash_array(edge_index_np)

        manifest = GraphManifest(
            graph_name=f"bbknn_kperbatch{k_b}_donors{n_donors}",
            builder_type="bbknn",
            dataset_name=dataset_name,
            split_id=split_id,
            k=k_b * n_donors,
            nominal_query_k=k_b * n_donors,
            nominal_train_k=k_b * n_donors,
            reference_edge_policy="bbknn",
            query_edge_policy="bbknn",
            metric=metric,
            weighting="unweighted",
            sigma_k=None,
            edge_index_convention="source_to_target",
            message_flow_train="train_to_train",
            message_flow_validation="train_to_validation",
            message_flow_test="train_to_test",
            query_nodes_affect_training_representations=False,
            num_nodes=n_total,
            num_edges=len(all_edges),
            num_train_nodes=n_tr,
            num_val_nodes=n_va,
            num_test_nodes=n_te,
            num_train_train_edges=num_tr_tr,
            num_train_to_val_edges=num_tr_to_va,
            num_train_to_test_edges=num_tr_to_te,
            num_disallowed_edges=0,
            edge_index_hash=edge_index_hash,
            edge_weight_hash=None,
            feature_manifest_hash=feature_manifest_hash,
        )

        logger.info(
            "BBKNN construction complete: %d total edges (%d train->train, %d train->val, %d train->test, 0 disallowed)",
            len(all_edges),
            num_tr_tr,
            num_tr_to_va,
            num_tr_to_te,
        )

        return GraphBundle(
            edge_index=edge_index,
            num_nodes=n_total,
            train_mask=train_mask,
            val_mask=val_mask,
            test_mask=test_mask,
            node_cell_ids=all_node_cell_ids,
            manifest=manifest,
            edge_weight=None,
        )
