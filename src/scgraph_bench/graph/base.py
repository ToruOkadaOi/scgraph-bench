"""Abstract base class for leakage-safe graph builders."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np

from scgraph_bench.graph.schema import GraphBundle


class BaseGraphBuilder(ABC):
    """Abstract base class for all single-cell graph construction algorithms.

    Inviolable Label Leakage Guard:
    Graph builders receive restricted feature matrices and allowed metadata tables only.
    They must never receive AnnData objects, cell-type labels, or label encoding dictionaries.
    """

    @abstractmethod
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
        """Construct graph bundle obeying strict inductive connectivity semantics.

        Args:
            X_pca_train: Fixed training feature matrix (N_train x D).
            X_pca_val: Fixed validation feature matrix (N_val x D).
            X_pca_test: Fixed test feature matrix (N_test x D).
            train_cell_ids: Ordered list of training cell IDs.
            val_cell_ids: Ordered list of validation cell IDs.
            test_cell_ids: Ordered list of test cell IDs.
            feature_manifest_hash: Cryptographic SHA-256 hash of the feature manifest.
            dataset_name: Dataset identifier.
            split_id: Frozen split identifier.
            allowed_metadata: Optional dictionary with non-label metadata (e.g. donor/site IDs).

        Returns:
            Constructed and validated GraphBundle.
        """
        raise NotImplementedError
