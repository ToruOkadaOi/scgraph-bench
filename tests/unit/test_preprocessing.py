"""Unit tests for leakage-safe, training-fitted preprocessing pipeline and feature manifest."""

import numpy as np
import scanpy as sc
from scipy import sparse

from scgraph_bench.config.preprocessing import PreprocessingConfig
from scgraph_bench.data.loaders import SyntheticFixtureLoader
from scgraph_bench.preprocessing.hvg import select_seurat_hvgs_train_only
from scgraph_bench.preprocessing.pipeline import LeakageSafePreprocessor
from scgraph_bench.preprocessing.schema import PreprocessedBundle
from scgraph_bench.splitting.group_split import create_site_stratified_donor_split


def test_hvg_parity_against_scanpy():
    """Verify select_seurat_hvgs_train_only matches scanpy.pp.highly_variable_genes exactly."""
    rng = np.random.default_rng(42)
    n_cells = 300
    n_genes = 200

    # Synthetic log1p normalized matrix
    X_norm = rng.poisson(lam=3.0, size=(n_cells, n_genes)).astype(np.float32)
    X_log1p = np.log1p(X_norm)
    gene_names = [f"ENSG_{i:05d}" for i in range(n_genes)]

    # 1. Custom wrapper
    hvg_idx, hvg_names = select_seurat_hvgs_train_only(
        X_norm_log1p_train=X_log1p,
        gene_names=gene_names,
        n_top_genes=50,
    )

    # 2. Direct Scanpy call
    import anndata as ad
    import pandas as pd

    adata_direct = ad.AnnData(X=X_log1p, var=pd.DataFrame(index=pd.Index(gene_names)))
    hvg_direct = sc.pp.highly_variable_genes(
        adata_direct,
        flavor="seurat",
        n_top_genes=50,
        n_bins=20,
        inplace=False,
    )
    direct_idx = np.where(hvg_direct["highly_variable"].to_numpy())[0]
    direct_names = [gene_names[i] for i in direct_idx]

    # Assert 100% exact parity
    assert np.array_equal(hvg_idx, direct_idx)
    assert hvg_names == direct_names


def test_leakage_invariance_on_test_cell_perturbation():
    """Verify that altering or adding test cells has zero effect on training-fitted artifacts."""
    rng = np.random.default_rng(42)
    n_train = 200
    n_test_1 = 50
    n_genes = 100

    # Fixed training matrix
    X_train = rng.poisson(lam=2.0, size=(n_train, n_genes)).astype(np.float32)
    gene_names = [f"gene_{i}" for i in range(n_genes)]

    # Test batch 1
    X_test_1 = rng.poisson(lam=1.5, size=(n_test_1, n_genes)).astype(np.float32)

    # Test batch 2 (perturbed, different size and expression profile)
    X_test_2 = rng.poisson(lam=10.0, size=(100, n_genes)).astype(np.float32)

    config = PreprocessingConfig(n_top_genes=30, n_comps=10, random_state=42)

    # Run 1: Fit with X_train
    prep_1 = LeakageSafePreprocessor(config)
    prep_1.fit(X_train, gene_names)
    X_pca_train_1, _ = prep_1.transform(X_train)
    X_pca_test_1, _ = prep_1.transform(X_test_1)

    # Run 2: Fit with X_train independently
    prep_2 = LeakageSafePreprocessor(config)
    prep_2.fit(X_train, gene_names)
    X_pca_train_2, _ = prep_2.transform(X_train)
    X_pca_test_2, _ = prep_2.transform(X_test_2)

    # Assert exact bit-for-bit invariance of training-derived parameters
    assert np.array_equal(prep_1.hvg_indices_, prep_2.hvg_indices_)
    assert np.allclose(prep_1.train_means_, prep_2.train_means_, atol=1e-7)
    assert np.allclose(prep_1.train_stds_, prep_2.train_stds_, atol=1e-7)
    assert prep_1.pca_ is not None and prep_2.pca_ is not None
    assert np.allclose(prep_1.pca_.components_, prep_2.pca_.components_, atol=1e-7)
    assert np.allclose(X_pca_train_1, X_pca_train_2, atol=1e-7)


def test_normalization_and_log1p_properties():
    """Verify target sum normalization scales rows correctly and log1p maps zeros to zeros."""
    counts = np.array(
        [
            [10.0, 20.0, 30.0],  # sum = 60
            [100.0, 200.0, 100.0],  # sum = 400
            [0.0, 0.0, 0.0],  # sum = 0
        ],
        dtype=np.float32,
    )
    target = 1000.0
    norm_log1p = LeakageSafePreprocessor._normalize_and_log1p(
        counts, target_sum=target, apply_log1p=True
    )

    # Row 0: scaled by 1000/60 = 16.6667
    expected_row0 = np.log1p(counts[0] * (target / 60.0))
    assert np.allclose(norm_log1p[0], expected_row0, atol=1e-5)

    # Row 2 (all zeros) remains all zeros
    assert np.allclose(norm_log1p[2], 0.0)

    # Sparse representation test
    sparse_counts = sparse.csr_matrix(counts)
    sparse_norm = LeakageSafePreprocessor._normalize_and_log1p(
        sparse_counts, target_sum=target, apply_log1p=True
    )
    assert np.allclose(sparse_norm.toarray(), norm_log1p, atol=1e-5)


def test_training_standardization_statistics():
    """Verify that training HVG matrix has zero mean and unit variance before clipping."""
    rng = np.random.default_rng(42)
    X_raw = rng.poisson(lam=5.0, size=(500, 50)).astype(np.float32)
    gene_names = [f"g_{i}" for i in range(50)]

    config = PreprocessingConfig(n_top_genes=30, scale_data=True, clip_value=None, n_comps=10)
    prep = LeakageSafePreprocessor(config)
    prep.fit(X_raw, gene_names)

    _, Z_train = prep.transform(X_raw)
    assert np.allclose(np.mean(Z_train, axis=0), 0.0, atol=1e-5)
    assert np.allclose(np.std(Z_train, axis=0), 1.0, atol=1e-4)


def test_pca_orthogonality():
    """Verify that fitted PCA components form an orthonormal basis."""
    rng = np.random.default_rng(42)
    X_raw = rng.poisson(lam=3.0, size=(200, 80)).astype(np.float32)
    gene_names = [f"g_{i}" for i in range(80)]

    config = PreprocessingConfig(n_top_genes=40, n_comps=15, random_state=42)
    prep = LeakageSafePreprocessor(config)
    prep.fit(X_raw, gene_names)

    assert prep.pca_ is not None
    components = prep.pca_.components_  # (n_comps, n_hvgs)
    identity_approx = components @ components.T
    assert np.allclose(identity_approx, np.eye(15), atol=1e-5)


def test_bundle_save_load_roundtrip_with_manifest(tmp_path):
    """Verify that PreprocessedBundle and FeatureManifest save and reload with bit-for-bit exactness."""
    loader = SyntheticFixtureLoader()
    adata = loader.load(seed=42)

    split_def = create_site_stratified_donor_split(
        adata=adata,
        donor_key="donor_id",
        site_key="condition",
        label_key="cell_type",
        split_id="test_split_prep",
        seed=42,
    )

    config = PreprocessingConfig(n_top_genes=50, n_comps=10, random_state=42)
    prep = LeakageSafePreprocessor(config)
    bundle = prep.fit_transform_split(adata, split_def, label_key="cell_type")

    out_dir = tmp_path / "preprocessed_bundle"
    bundle.save(out_dir, save_hvg=True)

    # Check files on disk
    assert (out_dir / "train_pca.npy").is_file()
    assert (out_dir / "val_pca.npy").is_file()
    assert (out_dir / "test_pca.npy").is_file()
    assert (out_dir / "feature_manifest.json").is_file()
    assert (out_dir / "preprocessor_metadata.json").is_file()

    reloaded = PreprocessedBundle.load(out_dir, load_hvg=True)

    assert np.array_equal(reloaded.X_pca_train, bundle.X_pca_train)
    assert np.array_equal(reloaded.X_pca_val, bundle.X_pca_val)
    assert np.array_equal(reloaded.X_pca_test, bundle.X_pca_test)
    assert np.array_equal(reloaded.train_labels, bundle.train_labels)
    assert np.array_equal(reloaded.val_labels, bundle.val_labels)
    assert np.array_equal(reloaded.test_labels, bundle.test_labels)
    assert reloaded.train_cell_ids == bundle.train_cell_ids
    assert reloaded.val_cell_ids == bundle.val_cell_ids
    assert reloaded.test_cell_ids == bundle.test_cell_ids
    assert reloaded.label_to_id == bundle.label_to_id
    assert reloaded.manifest.compute_manifest_hash() == bundle.manifest.compute_manifest_hash()
