"""Unit tests for path management and artifact path resolvers."""

from pathlib import Path

from scgraph_bench.utils.paths import ArtifactPaths, get_project_root


def test_get_project_root():
    """Verify project root contains expected anchor files like pyproject.toml."""
    root = get_project_root()
    assert isinstance(root, Path)
    assert (root / "pyproject.toml").is_file()


def test_artifact_paths_resolver(tmp_path):
    """Verify path derivation and directory creation."""
    paths = ArtifactPaths(root_dir=tmp_path)
    paths.ensure_directories()

    assert paths.data_dir.is_dir()
    assert paths.raw_data_dir.is_dir()
    assert paths.splits_dir.is_dir()
    assert paths.artifacts_dir.is_dir()
    assert paths.configs_dir.is_dir()
    assert paths.results_dir.is_dir()

    split_file = paths.dataset_split_file("kang_pbmc", "split_01")
    assert split_file == tmp_path / "splits" / "kang_pbmc" / "split_01.json"

    artifact_dir = paths.dataset_artifact_dir("kang_pbmc", "split_01")
    assert artifact_dir == tmp_path / "artifacts" / "kang_pbmc" / "split_01"
