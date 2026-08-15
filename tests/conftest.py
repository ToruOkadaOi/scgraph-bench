"""Pytest configuration and shared fixtures for CPU test suite."""

import pytest

from scgraph_bench.utils.paths import ArtifactPaths
from scgraph_bench.utils.seed import set_seed
from tests.fixtures.synthetic_adata import generate_synthetic_scrna_adata


@pytest.fixture(autouse=True)
def reset_random_seeds():
    """Ensure consistent random seed for every test."""
    set_seed(42, deterministic_torch=False)


@pytest.fixture
def synthetic_adata():
    """Provide a standard synthetic multi-donor AnnData fixture."""
    return generate_synthetic_scrna_adata(
        n_cells=200,
        n_genes=60,
        n_donors=6,
        n_classes=4,
        seed=42,
    )


@pytest.fixture
def temp_artifact_paths(tmp_path):
    """Provide an isolated ArtifactPaths pointing to a temporary test directory."""
    paths = ArtifactPaths(root_dir=tmp_path)
    paths.ensure_directories()
    return paths
