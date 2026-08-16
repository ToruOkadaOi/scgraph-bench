"""Schemas, feature manifests, and containers for preprocessed scRNA-seq feature representations."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from scgraph_bench.utils.hashing import hash_array, hash_dict, hash_string


class PreprocessorMetadata(BaseModel):
    """Metadata describing the training-fitted preprocessing pipeline and derived dimensions."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    dataset_name: str
    split_id: str
    config_hash: str
    n_train_cells: int
    n_val_cells: int
    n_test_cells: int
    n_genes_raw: int
    n_hvg_selected: int
    hvg_gene_names: list[str]
    hvg_indices: list[int]
    hvg_flavor: str
    hvg_n_bins: int
    scanpy_version: str
    input_transformation: str
    n_pca_components: int
    pca_solver: str
    pca_random_state: int
    pca_explained_variance: list[float]
    pca_explained_variance_ratio: list[float]
    pca_singular_values: list[float]
    pca_components_hash: str
    pca_mean_hash: str
    scaler_mean_hash: str
    scaler_std_hash: str
    train_mean_summary: dict[str, float]
    train_std_summary: dict[str, float]
    created_at_utc: str = Field(
        default_factory=lambda: datetime.now(UTC).isoformat(),
    )


class FeatureManifest(BaseModel):
    """Immutable manifest recording cryptographic hashes and provenance for preprocessed features."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    dataset_name: str
    split_id: str
    split_config_hash: str
    preprocessing_config_hash: str
    n_train_cells: int
    n_val_cells: int
    n_test_cells: int
    feature_dim: int
    train_pca_hash: str
    val_pca_hash: str
    test_pca_hash: str
    train_cell_ids_hash: str
    val_cell_ids_hash: str
    test_cell_ids_hash: str
    hvg_gene_names_hash: str
    scaler_mean_hash: str
    scaler_std_hash: str
    pca_components_hash: str
    pca_mean_hash: str
    label_mapping_hash: str
    created_at_utc: str = Field(
        default_factory=lambda: datetime.now(UTC).isoformat(),
    )

    def compute_manifest_hash(self) -> str:
        """Compute top-level hash of the entire feature manifest."""
        return hash_dict(self.model_dump(mode="json"))


class PreprocessedBundle:
    """Immutable in-memory and on-disk container for preprocessed feature matrices and split partitions."""

    def __init__(
        self,
        X_pca_train: np.ndarray,
        X_pca_val: np.ndarray,
        X_pca_test: np.ndarray,
        train_cell_ids: list[str],
        val_cell_ids: list[str],
        test_cell_ids: list[str],
        train_labels: np.ndarray,
        val_labels: np.ndarray,
        test_labels: np.ndarray,
        label_to_id: dict[str, int],
        metadata: PreprocessorMetadata,
        manifest: FeatureManifest,
        X_hvg_train: np.ndarray | None = None,
        X_hvg_val: np.ndarray | None = None,
        X_hvg_test: np.ndarray | None = None,
    ) -> None:
        self.X_pca_train = np.asarray(X_pca_train, dtype=np.float32)
        self.X_pca_val = np.asarray(X_pca_val, dtype=np.float32)
        self.X_pca_test = np.asarray(X_pca_test, dtype=np.float32)

        self.train_cell_ids = list(train_cell_ids)
        self.val_cell_ids = list(val_cell_ids)
        self.test_cell_ids = list(test_cell_ids)

        self.train_labels = np.asarray(train_labels, dtype=np.int64)
        self.val_labels = np.asarray(val_labels, dtype=np.int64)
        self.test_labels = np.asarray(test_labels, dtype=np.int64)
        self.label_to_id = dict(label_to_id)
        self.metadata = metadata
        self.manifest = manifest

        self.X_hvg_train = (
            np.asarray(X_hvg_train, dtype=np.float32) if X_hvg_train is not None else None
        )
        self.X_hvg_val = np.asarray(X_hvg_val, dtype=np.float32) if X_hvg_val is not None else None
        self.X_hvg_test = (
            np.asarray(X_hvg_test, dtype=np.float32) if X_hvg_test is not None else None
        )

        self._validate_shapes()

    def _validate_shapes(self) -> None:
        n_tr, n_va, n_te = len(self.train_cell_ids), len(self.val_cell_ids), len(self.test_cell_ids)
        if self.X_pca_train.shape[0] != n_tr:
            raise ValueError(
                f"X_pca_train shape {self.X_pca_train.shape} does not match train_cell_ids ({n_tr})"
            )
        if self.X_pca_val.shape[0] != n_va:
            raise ValueError(
                f"X_pca_val shape {self.X_pca_val.shape} does not match val_cell_ids ({n_va})"
            )
        if self.X_pca_test.shape[0] != n_te:
            raise ValueError(
                f"X_pca_test shape {self.X_pca_test.shape} does not match test_cell_ids ({n_te})"
            )
        if len(self.train_labels) != n_tr:
            raise ValueError(
                f"train_labels length {len(self.train_labels)} does not match train_cell_ids ({n_tr})"
            )
        if len(self.val_labels) != n_va:
            raise ValueError(
                f"val_labels length {len(self.val_labels)} does not match val_cell_ids ({n_va})"
            )
        if len(self.test_labels) != n_te:
            raise ValueError(
                f"test_labels length {len(self.test_labels)} does not match test_cell_ids ({n_te})"
            )

    def save(self, output_dir: Path | str, save_hvg: bool = False) -> None:
        """Persist unambiguous partition-specific feature files, labels, and manifests to disk."""
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)

        # 1. Save unambiguous partition-specific PCA arrays
        np.save(out / "train_pca.npy", self.X_pca_train)
        np.save(out / "val_pca.npy", self.X_pca_val)
        np.save(out / "test_pca.npy", self.X_pca_test)

        # 2. Save partition integer label arrays
        np.save(out / "train_labels.npy", self.train_labels)
        np.save(out / "val_labels.npy", self.val_labels)
        np.save(out / "test_labels.npy", self.test_labels)

        # 3. Save ordered cell ID lists
        (out / "train_cell_ids.json").write_text(
            json.dumps(self.train_cell_ids, indent=2), encoding="utf-8"
        )
        (out / "val_cell_ids.json").write_text(
            json.dumps(self.val_cell_ids, indent=2), encoding="utf-8"
        )
        (out / "test_cell_ids.json").write_text(
            json.dumps(self.test_cell_ids, indent=2), encoding="utf-8"
        )

        # 4. Save label encoding dictionary
        (out / "label_mapping.json").write_text(
            json.dumps(self.label_to_id, indent=2), encoding="utf-8"
        )

        # 5. Save detailed metadata and feature manifest
        (out / "preprocessor_metadata.json").write_text(
            self.metadata.model_dump_json(indent=2), encoding="utf-8"
        )
        (out / "feature_manifest.json").write_text(
            self.manifest.model_dump_json(indent=2), encoding="utf-8"
        )

        # 6. Legacy compatibility bundle
        np.savez_compressed(
            out / "pca_features.npz",
            X_pca_train=self.X_pca_train,
            X_pca_val=self.X_pca_val,
            X_pca_test=self.X_pca_test,
            train_labels=self.train_labels,
            val_labels=self.val_labels,
            test_labels=self.test_labels,
            train_cell_ids=np.array(self.train_cell_ids, dtype=object),
            val_cell_ids=np.array(self.val_cell_ids, dtype=object),
            test_cell_ids=np.array(self.test_cell_ids, dtype=object),
        )

        # 7. Optional HVG scaled matrices
        if (
            save_hvg
            and self.X_hvg_train is not None
            and self.X_hvg_val is not None
            and self.X_hvg_test is not None
        ):
            np.savez_compressed(
                out / "hvg_scaled_features.npz",
                X_hvg_train=self.X_hvg_train,
                X_hvg_val=self.X_hvg_val,
                X_hvg_test=self.X_hvg_test,
            )

    @classmethod
    def load(cls, output_dir: Path | str, load_hvg: bool = False) -> PreprocessedBundle:
        """Load preprocessed feature bundle and manifests from disk."""
        out = Path(output_dir)

        # Load partition-specific PCA arrays
        if (out / "train_pca.npy").is_file():
            X_pca_tr = np.load(out / "train_pca.npy")
            X_pca_va = np.load(out / "val_pca.npy")
            X_pca_te = np.load(out / "test_pca.npy")

            tr_labels = np.load(out / "train_labels.npy")
            va_labels = np.load(out / "val_labels.npy")
            te_labels = np.load(out / "test_labels.npy")

            tr_cids = json.loads((out / "train_cell_ids.json").read_text(encoding="utf-8"))
            va_cids = json.loads((out / "val_cell_ids.json").read_text(encoding="utf-8"))
            te_cids = json.loads((out / "test_cell_ids.json").read_text(encoding="utf-8"))
        else:
            # Fallback to npz
            pca_npz = np.load(out / "pca_features.npz", allow_pickle=True)
            X_pca_tr = pca_npz["X_pca_train"]
            X_pca_va = pca_npz["X_pca_val"]
            X_pca_te = pca_npz["X_pca_test"]
            tr_labels = pca_npz["train_labels"]
            va_labels = pca_npz["val_labels"]
            te_labels = pca_npz["test_labels"]
            tr_cids = pca_npz["train_cell_ids"].tolist()
            va_cids = pca_npz["val_cell_ids"].tolist()
            te_cids = pca_npz["test_cell_ids"].tolist()

        label_to_id = json.loads((out / "label_mapping.json").read_text(encoding="utf-8"))
        meta_dict = json.loads((out / "preprocessor_metadata.json").read_text(encoding="utf-8"))
        metadata = PreprocessorMetadata.model_validate(meta_dict)

        if (out / "feature_manifest.json").is_file():
            manifest_dict = json.loads((out / "feature_manifest.json").read_text(encoding="utf-8"))
            manifest = FeatureManifest.model_validate(manifest_dict)
        else:
            # Compute manifest on the fly if missing
            manifest = FeatureManifest(
                dataset_name=metadata.dataset_name,
                split_id=metadata.split_id,
                split_config_hash="",
                preprocessing_config_hash=metadata.config_hash,
                n_train_cells=len(tr_cids),
                n_val_cells=len(va_cids),
                n_test_cells=len(te_cids),
                feature_dim=X_pca_tr.shape[1],
                train_pca_hash=hash_array(X_pca_tr),
                val_pca_hash=hash_array(X_pca_va),
                test_pca_hash=hash_array(X_pca_te),
                train_cell_ids_hash=hash_string(json.dumps(tr_cids)),
                val_cell_ids_hash=hash_string(json.dumps(va_cids)),
                test_cell_ids_hash=hash_string(json.dumps(te_cids)),
                hvg_gene_names_hash=hash_string(json.dumps(metadata.hvg_gene_names)),
                scaler_mean_hash=metadata.scaler_mean_hash,
                scaler_std_hash=metadata.scaler_std_hash,
                pca_components_hash=metadata.pca_components_hash,
                pca_mean_hash=metadata.pca_mean_hash,
                label_mapping_hash=hash_dict(label_to_id),
            )

        X_hvg_tr = None
        X_hvg_va = None
        X_hvg_te = None
        hvg_file = out / "hvg_scaled_features.npz"
        if load_hvg and hvg_file.is_file():
            hvg_npz = np.load(hvg_file)
            X_hvg_tr = hvg_npz["X_hvg_train"]
            X_hvg_va = hvg_npz["X_hvg_val"]
            X_hvg_te = hvg_npz["X_hvg_test"]

        return cls(
            X_pca_train=X_pca_tr,
            X_pca_val=X_pca_va,
            X_pca_test=X_pca_te,
            train_cell_ids=tr_cids,
            val_cell_ids=va_cids,
            test_cell_ids=te_cids,
            train_labels=tr_labels,
            val_labels=va_labels,
            test_labels=te_labels,
            label_to_id=label_to_id,
            metadata=metadata,
            manifest=manifest,
            X_hvg_train=X_hvg_tr,
            X_hvg_val=X_hvg_va,
            X_hvg_test=X_hvg_te,
        )
