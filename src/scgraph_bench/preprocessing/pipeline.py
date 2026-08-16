"""Leakage-safe, training-fitted preprocessing pipeline for scRNA-seq benchmark."""

from __future__ import annotations

import importlib.metadata
import json

import anndata as ad
import numpy as np
from scipy import sparse
from sklearn.decomposition import PCA

from scgraph_bench.config.preprocessing import PreprocessingConfig
from scgraph_bench.preprocessing.hvg import select_seurat_hvgs_train_only
from scgraph_bench.preprocessing.schema import (
    FeatureManifest,
    PreprocessedBundle,
    PreprocessorMetadata,
)
from scgraph_bench.splitting.schema import SplitDefinition
from scgraph_bench.utils.hashing import hash_array, hash_dict, hash_string
from scgraph_bench.utils.logging import get_logger

logger = get_logger("preprocessing.pipeline")


class LeakageSafePreprocessor:
    """Strictly training-fitted scRNA-seq preprocessing pipeline.

    Enforces that library size target sum, log1p transformation, Seurat HVG selection,
    feature standardisation (mean/std), and PCA basis projection are fitted exclusively
    on training partition cells. Validation and test partitions are transformed strictly
    as out-of-sample query batches.
    """

    def __init__(self, config: PreprocessingConfig | None = None) -> None:
        self.config = config or PreprocessingConfig()
        self.is_fitted: bool = False

        # Fitted training parameters
        self.raw_gene_names_: list[str] = []
        self.hvg_indices_: np.ndarray = np.array([], dtype=int)
        self.hvg_gene_names_: list[str] = []
        self.train_means_: np.ndarray = np.array([], dtype=np.float32)
        self.train_stds_: np.ndarray = np.array([], dtype=np.float32)
        self.pca_: PCA | None = None

    @staticmethod
    def _normalize_and_log1p(
        X: sparse.spmatrix | np.ndarray,
        target_sum: float = 1e4,
        apply_log1p: bool = True,
    ) -> sparse.csr_matrix | np.ndarray:
        """Apply cell-wise library size normalisation to target_sum and log(1 + x)."""
        if sparse.issparse(X):
            X_csr = X.tocsr(copy=True).astype(np.float32)
            counts_per_cell = np.asarray(X_csr.sum(axis=1)).flatten()
            counts_per_cell[counts_per_cell == 0] = 1.0
            scale_factors = (target_sum / counts_per_cell).astype(np.float32)

            # Multiply rows by scaling factors
            diag_scale = sparse.diags(scale_factors)
            X_norm = diag_scale.dot(X_csr)

            if apply_log1p:
                X_norm = X_norm.log1p()
            return X_norm
        else:
            X_arr = np.asarray(X, dtype=np.float32).copy()
            counts_per_cell = X_arr.sum(axis=1, keepdims=True)
            counts_per_cell[counts_per_cell == 0] = 1.0
            X_norm = X_arr * (target_sum / counts_per_cell)

            if apply_log1p:
                X_norm = np.log1p(X_norm)
            return X_norm

    def fit(
        self,
        X_raw_train: sparse.spmatrix | np.ndarray,
        gene_names: list[str],
    ) -> LeakageSafePreprocessor:
        """Fit preprocessing pipeline strictly on training cells.

        Args:
            X_raw_train: Raw count expression matrix for training cells (N_train x G).
            gene_names: Gene names/identifiers for columns.

        Returns:
            Fitted preprocessor instance.
        """
        n_cells, n_genes = X_raw_train.shape
        self.raw_gene_names_ = list(gene_names)

        logger.info(
            "Fitting LeakageSafePreprocessor on %d training cells x %d genes (target_sum=%.1f, n_hvgs=%d, n_pca=%d)",
            n_cells,
            n_genes,
            self.config.target_sum,
            self.config.n_top_genes,
            self.config.n_comps,
        )

        # 1. Normalise to target_sum and apply log1p
        X_norm_log1p_train = self._normalize_and_log1p(
            X_raw_train,
            target_sum=self.config.target_sum,
            apply_log1p=self.config.log1p,
        )

        # 2. Select Highly Variable Genes strictly on training log1p data
        self.hvg_indices_, self.hvg_gene_names_ = select_seurat_hvgs_train_only(
            X_norm_log1p_train=X_norm_log1p_train,
            gene_names=self.raw_gene_names_,
            n_top_genes=self.config.n_top_genes,
        )

        # 3. Extract training HVG slice and compute training mean and std
        if sparse.issparse(X_norm_log1p_train):
            X_train_hvg = X_norm_log1p_train[:, self.hvg_indices_].toarray()
        else:
            X_train_hvg = X_norm_log1p_train[:, self.hvg_indices_]

        if self.config.scale_data:
            self.train_means_ = np.mean(X_train_hvg, axis=0).astype(np.float32)
            self.train_stds_ = np.std(X_train_hvg, axis=0, ddof=0).astype(np.float32)
            # Guard against zero-variance genes
            self.train_stds_[self.train_stds_ < 1e-12] = 1.0

            Z_train = (X_train_hvg - self.train_means_) / self.train_stds_
            if self.config.clip_value is not None:
                Z_train = np.clip(Z_train, -self.config.clip_value, self.config.clip_value)
        else:
            self.train_means_ = np.zeros(len(self.hvg_indices_), dtype=np.float32)
            self.train_stds_ = np.ones(len(self.hvg_indices_), dtype=np.float32)
            Z_train = X_train_hvg

        # 4. Fit PCA strictly on scaled training HVG matrix
        n_comps = min(self.config.n_comps, n_cells - 1, len(self.hvg_indices_))
        logger.info("Fitting PCA with %d components on training scaled HVGs...", n_comps)
        self.pca_ = PCA(
            n_components=n_comps,
            svd_solver=self.config.pca_solver,
            random_state=self.config.random_state,
        )
        self.pca_.fit(Z_train)

        self.is_fitted = True
        logger.info(
            "Preprocessor fitting complete. Total explained variance ratio: %.4f",
            float(np.sum(self.pca_.explained_variance_ratio_)),
        )
        return self

    def transform(
        self,
        X_raw: sparse.spmatrix | np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Transform raw counts into fixed 50-dim PCA and scaled HVG feature representations.

        Args:
            X_raw: Raw counts matrix for an evaluation partition (N x G).

        Returns:
            tuple containing:
                - X_pca: np.ndarray (N x n_comps) projected into training PCA basis.
                - X_scaled_hvg: np.ndarray (N x n_hvgs) standard-scaled using training statistics.
        """
        if not self.is_fitted or self.pca_ is None:
            raise RuntimeError("Preprocessor has not been fitted. Call fit() first.")

        # 1. Normalise to target_sum and log1p
        X_norm_log1p = self._normalize_and_log1p(
            X_raw,
            target_sum=self.config.target_sum,
            apply_log1p=self.config.log1p,
        )

        # 2. Extract fitted HVG columns
        if sparse.issparse(X_norm_log1p):
            X_hvg = X_norm_log1p[:, self.hvg_indices_].toarray()
        else:
            X_hvg = X_norm_log1p[:, self.hvg_indices_]

        # 3. Standard-scale using training statistics
        if self.config.scale_data:
            Z = (X_hvg - self.train_means_) / self.train_stds_
            if self.config.clip_value is not None:
                Z = np.clip(Z, -self.config.clip_value, self.config.clip_value)
        else:
            Z = X_hvg

        # 4. Project into training PCA space (using fitted transform only, never fit)
        X_pca = self.pca_.transform(Z).astype(np.float32)
        return X_pca, Z.astype(np.float32)

    def fit_transform_split(
        self,
        adata: ad.AnnData,
        split_def: SplitDefinition,
        label_key: str = "cell_type",
    ) -> PreprocessedBundle:
        """Execute end-to-end training-fitted preprocessing on an AnnData object and frozen split.

        Args:
            adata: AnnData with raw counts in .X.
            split_def: Frozen SplitDefinition defining train, validation, and test cell IDs.
            label_key: Metadata column in adata.obs containing cell type labels.

        Returns:
            PreprocessedBundle containing fixed X_pca matrices, encoded labels, and metadata.
        """
        cell_id_to_idx = {str(cid): idx for idx, cid in enumerate(adata.obs_names)}

        train_indices = [cell_id_to_idx[cid] for cid in split_def.train_cell_ids]
        val_indices = [cell_id_to_idx[cid] for cid in split_def.val_cell_ids]
        test_indices = [cell_id_to_idx[cid] for cid in split_def.test_cell_ids]

        gene_names = list(adata.var_names.astype(str))

        # Extract partition raw counts
        X_train_raw = adata.X[train_indices]
        X_val_raw = adata.X[val_indices]
        X_test_raw = adata.X[test_indices]

        # 1. Fit strictly on training partition
        self.fit(X_raw_train=X_train_raw, gene_names=gene_names)

        # 2. Transform all partitions using training-fitted state
        X_pca_train, X_hvg_train = self.transform(X_train_raw)
        X_pca_val, X_hvg_val = self.transform(X_val_raw)
        X_pca_test, X_hvg_test = self.transform(X_test_raw)

        # 3. Deterministic integer label encoding
        unique_labels = sorted(adata.obs[label_key].unique().tolist())
        label_to_id = {lbl: idx for idx, lbl in enumerate(unique_labels)}

        train_labels = np.array(
            [label_to_id[lbl] for lbl in adata.obs.iloc[train_indices][label_key]], dtype=np.int64
        )
        val_labels = np.array(
            [label_to_id[lbl] for lbl in adata.obs.iloc[val_indices][label_key]], dtype=np.int64
        )
        test_labels = np.array(
            [label_to_id[lbl] for lbl in adata.obs.iloc[test_indices][label_key]], dtype=np.int64
        )

        # 4. Construct PreprocessorMetadata
        assert self.pca_ is not None
        pca_components_hash = hash_array(self.pca_.components_)
        pca_mean_hash = hash_array(self.pca_.mean_) if self.pca_.mean_ is not None else ""
        scaler_mean_hash = hash_array(self.train_means_)
        scaler_std_hash = hash_array(self.train_stds_)

        metadata = PreprocessorMetadata(
            dataset_name=split_def.dataset_name,
            split_id=split_def.split_id,
            config_hash=self.config.compute_hash(),
            n_train_cells=len(train_indices),
            n_val_cells=len(val_indices),
            n_test_cells=len(test_indices),
            n_genes_raw=len(gene_names),
            n_hvg_selected=len(self.hvg_indices_),
            hvg_gene_names=self.hvg_gene_names_,
            hvg_indices=self.hvg_indices_.tolist(),
            hvg_flavor=self.config.hvg_flavor.value,
            hvg_n_bins=20,
            scanpy_version=importlib.metadata.version("scanpy"),
            input_transformation="target_sum_1e4_log1p",
            n_pca_components=int(self.pca_.n_components_),
            pca_solver=self.config.pca_solver,
            pca_random_state=self.config.random_state,
            pca_explained_variance=[float(v) for v in self.pca_.explained_variance_],
            pca_explained_variance_ratio=[float(v) for v in self.pca_.explained_variance_ratio_],
            pca_singular_values=[float(v) for v in self.pca_.singular_values_],
            pca_components_hash=pca_components_hash,
            pca_mean_hash=pca_mean_hash,
            scaler_mean_hash=scaler_mean_hash,
            scaler_std_hash=scaler_std_hash,
            train_mean_summary={
                "min": float(np.min(self.train_means_)),
                "max": float(np.max(self.train_means_)),
                "mean": float(np.mean(self.train_means_)),
            },
            train_std_summary={
                "min": float(np.min(self.train_stds_)),
                "max": float(np.max(self.train_stds_)),
                "mean": float(np.mean(self.train_stds_)),
            },
        )

        # 5. Construct FeatureManifest
        manifest = FeatureManifest(
            dataset_name=split_def.dataset_name,
            split_id=split_def.split_id,
            split_config_hash=split_def.config_hash,
            preprocessing_config_hash=self.config.compute_hash(),
            n_train_cells=len(train_indices),
            n_val_cells=len(val_indices),
            n_test_cells=len(test_indices),
            feature_dim=X_pca_train.shape[1],
            train_pca_hash=hash_array(X_pca_train),
            val_pca_hash=hash_array(X_pca_val),
            test_pca_hash=hash_array(X_pca_test),
            train_cell_ids_hash=hash_string(json.dumps(split_def.train_cell_ids)),
            val_cell_ids_hash=hash_string(json.dumps(split_def.val_cell_ids)),
            test_cell_ids_hash=hash_string(json.dumps(split_def.test_cell_ids)),
            hvg_gene_names_hash=hash_string(json.dumps(self.hvg_gene_names_)),
            scaler_mean_hash=scaler_mean_hash,
            scaler_std_hash=scaler_std_hash,
            pca_components_hash=pca_components_hash,
            pca_mean_hash=pca_mean_hash,
            label_mapping_hash=hash_dict(label_to_id),
        )

        return PreprocessedBundle(
            X_pca_train=X_pca_train,
            X_pca_val=X_pca_val,
            X_pca_test=X_pca_test,
            train_cell_ids=split_def.train_cell_ids,
            val_cell_ids=split_def.val_cell_ids,
            test_cell_ids=split_def.test_cell_ids,
            train_labels=train_labels,
            val_labels=val_labels,
            test_labels=test_labels,
            label_to_id=label_to_id,
            metadata=metadata,
            manifest=manifest,
            X_hvg_train=X_hvg_train,
            X_hvg_val=X_hvg_val,
            X_hvg_test=X_hvg_test,
        )
