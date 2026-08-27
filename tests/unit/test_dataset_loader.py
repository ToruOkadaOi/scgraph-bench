"""Unit tests for dataset loaders and schema validation."""

import anndata as ad
import numpy as np
import pandas as pd
import pytest
from scipy import sparse

from scgraph_bench.config.dataset import DatasetConfig
from scgraph_bench.data.loaders import (
    PRIMARY_V0_LABELS_STEPHENSON,
    StephensonHealthyPBMCLoader,
    SyntheticFixtureLoader,
)
from scgraph_bench.data.registry import get_dataset_loader, list_registered_datasets
from scgraph_bench.data.validation import DatasetValidationError, validate_anndata_schema


def test_registry_discovery():
    """Verify registered datasets are discoverable."""
    registered = list_registered_datasets()
    assert "stephenson_2021_healthy_pbmc" in registered
    assert "synthetic_fixture" in registered

    loader = get_dataset_loader("synthetic_fixture")
    assert isinstance(loader, SyntheticFixtureLoader)


def test_synthetic_fixture_loader():
    """Verify synthetic fixture loader returns compliant AnnData."""
    loader = SyntheticFixtureLoader()
    adata = loader.load()
    assert adata.n_obs == 600
    assert adata.n_vars == 100
    assert "donor_id" in adata.obs.columns
    assert "cell_type" in adata.obs.columns
    assert "cell_id" in adata.obs.columns


def test_schema_validation_catches_negative_counts(synthetic_adata):
    """Verify validator flags negative count values."""
    config = DatasetConfig(name="synthetic_fixture", donor_key="donor_id", label_key="cell_type")
    # Make a copy and inject negative value
    adata_bad = synthetic_adata.copy()
    dense_x = adata_bad.X.toarray() if hasattr(adata_bad.X, "toarray") else adata_bad.X.copy()
    dense_x[0, 0] = -5.0
    adata_bad.X = dense_x

    with pytest.raises(DatasetValidationError, match="negative count"):
        validate_anndata_schema(adata_bad, config)


def test_schema_validation_catches_missing_columns(synthetic_adata):
    """Verify validator flags missing required columns."""
    config = DatasetConfig(
        name="synthetic_fixture", donor_key="nonexistent_donor", label_key="cell_type"
    )
    with pytest.raises(DatasetValidationError, match="Missing required metadata"):
        validate_anndata_schema(synthetic_adata, config)


def test_stephenson_loader_dev_subsampling(tmp_path):
    """Verify development subsample produces deterministic reduced dataset on cached Stephenson fixture."""
    # 1. Create a synthetic AnnData matching Stephenson 23-donor structure
    manifest_path = tmp_path / "stephenson_manifest.csv"
    donors_cambridge = [f"CAM_{i:02d}" for i in range(12)]
    donors_newcastle = [f"NCL_{i:02d}" for i in range(11)]
    all_donors = donors_cambridge + donors_newcastle

    manifest_df = pd.DataFrame(
        {
            "donor_id": all_donors,
            "site": ["Cambridge"] * 12 + ["Newcastle"] * 11,
            "inclusion_status": ["included"] * 23,
        }
    )
    manifest_df.to_csv(manifest_path, index=False)

    # 2. Build mock cache .h5ad with 100 cells per donor across primary labels
    cells_per_donor = 100
    n_cells = len(all_donors) * cells_per_donor
    cell_ids = [f"cell_{i:05d}" for i in range(n_cells)]
    cell_donors = [d for d in all_donors for _ in range(cells_per_donor)]
    rng = np.random.default_rng(42)
    cell_labels = rng.choice(PRIMARY_V0_LABELS_STEPHENSON, size=n_cells)

    mock_x = sparse.csr_matrix(rng.poisson(lam=2.0, size=(n_cells, 50)).astype(np.float32))
    mock_obs = pd.DataFrame(
        {
            "donor_id": cell_donors,
            "cell_type": cell_labels,
        },
        index=cell_ids,
    )
    mock_var = pd.DataFrame(
        {"feature_id": [f"ENSG{j:08d}" for j in range(50)]},
        index=[f"ENSG{j:08d}" for j in range(50)],
    )
    mock_adata = ad.AnnData(X=mock_x, obs=mock_obs, var=mock_var)

    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    cache_file = cache_dir / "stephenson_2021_healthy_pbmc.h5ad"
    mock_adata.write_h5ad(cache_file)

    # 3. Test loader with subsampling
    loader = StephensonHealthyPBMCLoader(cache_dir=cache_dir, manifest_path=manifest_path)
    adata_sub = loader.load(dev_subsample_per_donor=50, seed=42)

    assert adata_sub.n_obs == 23 * 50  # 23 donors * 50 cells = 1150
    assert (adata_sub.obs["donor_id"].value_counts() == 50).all()
    assert adata_sub.obs["cell_type"].isin(PRIMARY_V0_LABELS_STEPHENSON).all()
    assert "site" in adata_sub.obs.columns


def test_stephenson_loader_missing_census_error(tmp_path, monkeypatch):
    """Verify that missing cache triggers clear RuntimeError if cellxgene_census is unavailable."""
    import sys

    # Simulate cellxgene_census missing from sys.modules
    monkeypatch.setitem(sys.modules, "cellxgene_census", None)

    empty_cache = tmp_path / "empty_cache"
    empty_cache.mkdir()
    loader = StephensonHealthyPBMCLoader(cache_dir=empty_cache)

    with pytest.raises(RuntimeError, match="cellxgene_census is required"):
        loader.load()


def test_gse164690_loader_dev_subsampling(tmp_path):
    """Verify GSE164690 loader dev subsampling on mock fixture."""
    from scgraph_bench.data.loaders import GSE164690HNSCCLoader

    manifest_path = tmp_path / "gse164690_manifest.csv"
    donors = [f"HN{i:02d}" for i in range(1, 19)]
    manifest_df = pd.DataFrame(
        {
            "donor_id": donors,
            "hpv_status": ["HPV_positive"] * 6 + ["HPV_negative"] * 12,
            "inclusion_status": ["included"] * 18,
        }
    )
    manifest_df.to_csv(manifest_path, index=False)

    cells_per_donor = 60
    n_cells = len(donors) * cells_per_donor
    cell_ids = [f"cell_{i:05d}" for i in range(n_cells)]
    cell_donors = [d for d in donors for _ in range(cells_per_donor)]
    rng = np.random.default_rng(42)
    cell_labels = rng.choice(GSE164690HNSCCLoader.PRIMARY_V0_LABELS, size=n_cells)

    mock_x = sparse.csr_matrix(rng.poisson(lam=2.0, size=(n_cells, 50)).astype(np.float32))
    mock_obs = pd.DataFrame(
        {
            "donor_id": cell_donors,
            "cell_type": cell_labels,
            "compartment": ["tumor"] * n_cells,
        },
        index=cell_ids,
    )
    mock_var = pd.DataFrame(
        {"gene_id": [f"ENSG{j:08d}" for j in range(50)]},
        index=[f"GENE{j:04d}" for j in range(50)],
    )
    mock_adata = ad.AnnData(X=mock_x, obs=mock_obs, var=mock_var)

    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    cache_file = cache_dir / "gse164690_annotated.h5ad"
    mock_adata.write_h5ad(cache_file)

    loader = GSE164690HNSCCLoader(cache_dir=cache_dir, manifest_path=manifest_path)
    adata_sub = loader.load(dev_subsample_per_donor=20, seed=42)

    assert adata_sub.n_obs == 18 * 20
    assert (adata_sub.obs["donor_id"].value_counts() == 20).all()
    assert adata_sub.obs["cell_type"].isin(GSE164690HNSCCLoader.PRIMARY_V0_LABELS).all()
