"""Unit tests for dataset loaders and schema validation."""

import pytest

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


def test_stephenson_loader_dev_subsampling():
    """Verify development subsample produces deterministic reduced dataset."""
    loader = StephensonHealthyPBMCLoader()
    # Test dev subsampling with 50 cells per donor
    adata = loader.load(dev_subsample_per_donor=50, seed=42)
    assert adata.n_obs == 23 * 50  # 23 donors * 50 cells
    assert (adata.obs["donor_id"].value_counts() == 50).all()
    assert adata.obs["cell_type"].isin(PRIMARY_V0_LABELS_STEPHENSON).all()
