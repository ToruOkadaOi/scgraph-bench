"""Path management and standard directory resolution."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


def get_project_root() -> Path:
    """Find the root directory of the scgraph-bench repository."""
    # Check if explicitly configured via environment variable
    if env_root := os.getenv("SCGRAPH_BENCH_ROOT"):
        return Path(env_root).resolve()

    # Locate relative to this file (src/scgraph_bench/utils/paths.py -> root)
    current = Path(__file__).resolve()
    # Go up 4 levels: utils -> scgraph_bench -> src -> repo root
    return current.parents[3]


@dataclass(frozen=True)
class ArtifactPaths:
    """Structured path resolver for project artifacts."""

    root_dir: Path

    @classmethod
    def default(cls) -> ArtifactPaths:
        return cls(root_dir=get_project_root())

    @property
    def data_dir(self) -> Path:
        return self.root_dir / "data"

    @property
    def raw_data_dir(self) -> Path:
        return self.data_dir / "raw"

    @property
    def splits_dir(self) -> Path:
        return self.root_dir / "splits"

    @property
    def artifacts_dir(self) -> Path:
        return self.root_dir / "artifacts"

    @property
    def configs_dir(self) -> Path:
        return self.root_dir / "configs"

    @property
    def results_dir(self) -> Path:
        return self.root_dir / "results"

    def dataset_split_file(self, dataset_name: str, split_id: str) -> Path:
        """Resolve path to a frozen split JSON file."""
        return self.splits_dir / dataset_name / f"{split_id}.json"

    def dataset_artifact_dir(self, dataset_name: str, split_id: str) -> Path:
        """Resolve directory for preprocessed features and serialized graphs."""
        return self.artifacts_dir / dataset_name / split_id

    def ensure_directories(self) -> None:
        """Create standard directories if they do not exist."""
        for path in [
            self.data_dir,
            self.raw_data_dir,
            self.splits_dir,
            self.artifacts_dir,
            self.configs_dir,
            self.results_dir,
        ]:
            path.mkdir(parents=True, exist_ok=True)
