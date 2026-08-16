"""Unit tests for site-stratified donor splitting and frozen split serialization."""

import anndata as ad
import numpy as np
import pandas as pd
import pytest
from scipy import sparse

from scgraph_bench.config.split import SplitType
from scgraph_bench.splitting.group_split import create_site_stratified_donor_split
from scgraph_bench.splitting.random_split import create_random_cell_split
from scgraph_bench.splitting.schema import SplitDefinition


@pytest.fixture
def multi_site_adata() -> ad.AnnData:
    """Deterministic in-memory multi-donor, multi-site single-cell fixture."""
    rng = np.random.default_rng(42)
    donors_cambridge = [f"CAM_{i:02d}" for i in range(12)]
    donors_newcastle = [f"NCL_{i:02d}" for i in range(11)]
    all_donors = donors_cambridge + donors_newcastle
    donor_to_site = {d: ("Cambridge" if d.startswith("CAM") else "Newcastle") for d in all_donors}

    cells_per_donor = 40
    n_cells = len(all_donors) * cells_per_donor
    cell_ids = [f"cell_{i:05d}" for i in range(n_cells)]
    cell_donors = [d for d in all_donors for _ in range(cells_per_donor)]
    cell_sites = [donor_to_site[d] for d in cell_donors]
    cell_types = [f"cell_type_{i}" for i in range(12)]
    cell_labels = rng.choice(cell_types, size=n_cells)

    mock_x = sparse.csr_matrix(rng.poisson(lam=2.0, size=(n_cells, 30)).astype(np.float32))
    mock_obs = pd.DataFrame(
        {
            "cell_id": cell_ids,
            "donor_id": cell_donors,
            "site": cell_sites,
            "cell_type": cell_labels,
        },
        index=cell_ids,
    )
    mock_var = pd.DataFrame(
        {"gene_id": [f"gene_{j}" for j in range(30)]},
        index=[f"gene_{j}" for j in range(30)],
    )
    return ad.AnnData(X=mock_x, obs=mock_obs, var=mock_var)


def test_site_stratified_split_disjointness(multi_site_adata):
    """Verify site-stratified donor splitting yields strictly disjoint partitions."""
    split_def = create_site_stratified_donor_split(
        adata=multi_site_adata,
        donor_key="donor_id",
        site_key="site",
        label_key="cell_type",
        split_id="test_site_stratified",
        seed=42,
    )

    # Disjointness checks
    train_d = set(split_def.train_donors)
    val_d = set(split_def.val_donors)
    test_d = set(split_def.test_donors)

    assert train_d.isdisjoint(val_d)
    assert train_d.isdisjoint(test_d)
    assert val_d.isdisjoint(test_d)

    train_c = set(split_def.train_cell_ids)
    val_c = set(split_def.val_cell_ids)
    test_c = set(split_def.test_cell_ids)

    assert train_c.isdisjoint(val_c)
    assert train_c.isdisjoint(test_c)
    assert val_c.isdisjoint(test_c)

    assert len(train_c) + len(val_c) + len(test_c) == multi_site_adata.n_obs


def test_frozen_split_json_roundtrip(multi_site_adata, tmp_path):
    """Verify SplitDefinition serializes to JSON and reloads with 100% cell ID alignment."""
    split_def = create_site_stratified_donor_split(
        adata=multi_site_adata,
        split_id="test_roundtrip",
        seed=100,
    )

    json_file = tmp_path / "test_split.json"
    split_def.save_json(json_file)

    reloaded = SplitDefinition.load_json(json_file)
    assert reloaded.split_id == split_def.split_id
    assert reloaded.train_donors == split_def.train_donors
    assert reloaded.val_donors == split_def.val_donors
    assert reloaded.test_donors == split_def.test_donors
    assert reloaded.train_cell_ids == split_def.train_cell_ids
    assert reloaded.val_cell_ids == split_def.val_cell_ids
    assert reloaded.test_cell_ids == split_def.test_cell_ids
    assert reloaded.compute_artifact_hash() == split_def.compute_artifact_hash()


def test_random_cell_split_marked_as_debug(multi_site_adata):
    """Verify random-cell splitter returns SplitType.RANDOM_CELL_DEBUG."""
    debug_split = create_random_cell_split(adata=multi_site_adata, split_id="debug_random", seed=42)
    assert debug_split.split_type == SplitType.RANDOM_CELL_DEBUG
