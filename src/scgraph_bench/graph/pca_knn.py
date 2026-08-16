"""Strict-inductive PCA-kNN graph construction with explicit PyG message flow."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from sklearn.neighbors import NearestNeighbors

from scgraph_bench.config.graph import EdgeWeightingMode, PCAkNNConfig
from scgraph_bench.graph.base import BaseGraphBuilder
from scgraph_bench.graph.registry import register_graph_builder
from scgraph_bench.graph.schema import GraphBundle, GraphManifest
from scgraph_bench.utils.hashing import hash_array
from scgraph_bench.utils.logging import get_logger

logger = get_logger("graph.pca_knn")


@register_graph_builder("pca_knn")
class PCAkNNGraphBuilder(BaseGraphBuilder):
    """Constructs strict-inductive PCA-kNN graphs with directional PyG message flow.

    Strict Inductive Connectivity & Message Flow Semantics:
    - Train -> Train: Symmetrized kNN graph among training reference cells.
    - Train -> Validation: Directed edges from training neighbors (source) to validation query nodes (target).
    - Train -> Test: Directed edges from training neighbors (source) to test query nodes (target).
    - Query -> Train edges are explicitly forbidden (0 val->train, 0 test->train).
    - Query -> Query edges are explicitly forbidden (0 val->val, 0 test->test, 0 test->val).
    """

    def __init__(self, config: PCAkNNConfig | None = None) -> None:
        self.config = config or PCAkNNConfig()

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
        """Construct strict-inductive PCA-kNN graph."""
        _ = allowed_metadata
        n_tr = len(train_cell_ids)
        n_va = len(val_cell_ids)
        n_te = len(test_cell_ids)
        n_total = n_tr + n_va + n_te

        k = self.config.k
        metric = self.config.metric
        logger.info(
            "Building strict-inductive PCA-kNN graph (k=%d, metric='%s', weighting='%s') "
            "for %d total nodes (%d train, %d val, %d test)...",
            k,
            metric,
            self.config.weighting.value,
            n_total,
            n_tr,
            n_va,
            n_te,
        )

        # 1. Fit NearestNeighbors exclusively on training partition
        nn_train = NearestNeighbors(n_neighbors=k + 1, metric=metric, algorithm="auto")
        nn_train.fit(X_pca_train)

        # 2. Compute training-to-training edges
        train_dists, train_indices = nn_train.kneighbors(X_pca_train)

        # Compute RBF bandwidth heuristic sigma_k: median distance to k-th nearest neighbor in train
        kth_dists = train_dists[:, k]
        sigma_k = float(np.median(kth_dists))
        logger.info(
            "Computed RBF bandwidth sigma_%d = %.4f (median k-th neighbor distance in train)",
            k,
            sigma_k,
        )

        # Build train-train edge set (source -> target)
        train_train_edges: set[tuple[int, int]] = set()
        edge_dists: dict[tuple[int, int], float] = {}

        for i in range(n_tr):
            for rank in range(1, k + 1):
                j = int(train_indices[i, rank])
                d = float(train_dists[i, rank])
                # Edge from neighbor j to cell i (source j, target i)
                train_train_edges.add((j, i))
                edge_dists[(j, i)] = d
                if self.config.symmetrize:
                    train_train_edges.add((i, j))
                    edge_dists[(i, j)] = d

        # 3. Compute Train -> Validation edges (source = train neighbor, target = validation cell)
        val_dists, val_indices = nn_train.kneighbors(X_pca_val, n_neighbors=k)
        train_to_val_edges: set[tuple[int, int]] = set()

        for u in range(n_va):
            target_val_idx = n_tr + u
            for rank in range(k):
                source_train_idx = int(val_indices[u, rank])
                d = float(val_dists[u, rank])
                edge = (source_train_idx, target_val_idx)
                train_to_val_edges.add(edge)
                edge_dists[edge] = d

        # 4. Compute Train -> Test edges (source = train neighbor, target = test cell)
        test_dists, test_indices = nn_train.kneighbors(X_pca_test, n_neighbors=k)
        train_to_test_edges: set[tuple[int, int]] = set()

        for w in range(n_te):
            target_test_idx = n_tr + n_va + w
            for rank in range(k):
                source_train_idx = int(test_indices[w, rank])
                d = float(test_dists[w, rank])
                edge = (source_train_idx, target_test_idx)
                train_to_test_edges.add(edge)
                edge_dists[edge] = d

        # 5. Assemble all edges (source -> target order)
        all_edges = sorted(train_train_edges | train_to_val_edges | train_to_test_edges)
        src_nodes = [e[0] for e in all_edges]
        dst_nodes = [e[1] for e in all_edges]

        edge_index_np = np.array([src_nodes, dst_nodes], dtype=np.int64)
        edge_index = torch.from_numpy(edge_index_np)

        edge_weight: torch.Tensor | None = None
        edge_weight_hash: str | None = None

        if self.config.weighting == EdgeWeightingMode.RBF_WEIGHTED:
            weights = []
            two_sigma_sq = 2.0 * (sigma_k**2)
            for e in all_edges:
                dist = edge_dists.get(e, 0.0)
                w = np.exp(-(dist**2) / two_sigma_sq)
                weights.append(w)
            weights_np = np.array(weights, dtype=np.float32)
            edge_weight = torch.from_numpy(weights_np)
            edge_weight_hash = hash_array(weights_np)

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
                # Any edge where source is not in train, or target is invalid
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
            graph_name=f"pca_knn_k{k}_{self.config.weighting.value}",
            builder_type="pca_knn",
            dataset_name=dataset_name,
            split_id=split_id,
            k=k,
            nominal_query_k=k,
            nominal_train_k=k,
            reference_edge_policy="standard_knn",
            query_edge_policy="standard_knn",
            metric=metric,
            weighting=self.config.weighting.value,
            sigma_k=sigma_k,
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
            edge_weight_hash=edge_weight_hash,
            feature_manifest_hash=feature_manifest_hash,
        )

        logger.info(
            "Graph construction complete: %d total edges (%d train->train, %d train->val, %d train->test, 0 disallowed)",
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
            edge_weight=edge_weight,
        )
