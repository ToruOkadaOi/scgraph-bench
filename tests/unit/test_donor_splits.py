"""Unit tests for site-stratified donor splitting and frozen split serialization."""

from scgraph_bench.config.split import SplitType
from scgraph_bench.data.loaders import StephensonHealthyPBMCLoader
from scgraph_bench.splitting.group_split import create_site_stratified_donor_split
from scgraph_bench.splitting.random_split import create_random_cell_split
from scgraph_bench.splitting.schema import SplitDefinition


def test_site_stratified_split_disjointness():
    """Verify site-stratified donor splitting yields strictly disjoint partitions."""
    loader = StephensonHealthyPBMCLoader()
    adata = loader.load(dev_subsample_per_donor=100, seed=42)

    split_def = create_site_stratified_donor_split(
        adata=adata,
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

    assert len(train_c) + len(val_c) + len(test_c) == adata.n_obs


def test_frozen_split_json_roundtrip(tmp_path):
    """Verify SplitDefinition serializes to JSON and reloads with 100% cell ID alignment."""
    loader = StephensonHealthyPBMCLoader()
    adata = loader.load(dev_subsample_per_donor=50, seed=42)

    split_def = create_site_stratified_donor_split(
        adata=adata,
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


def test_random_cell_split_marked_as_debug():
    """Verify random-cell splitter returns SplitType.RANDOM_CELL_DEBUG."""
    loader = StephensonHealthyPBMCLoader()
    adata = loader.load(dev_subsample_per_donor=30, seed=42)

    debug_split = create_random_cell_split(adata=adata, split_id="debug_random", seed=42)
    assert debug_split.split_type == SplitType.RANDOM_CELL_DEBUG
