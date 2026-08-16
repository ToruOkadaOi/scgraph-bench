"""Abstract base class for scRNA-seq dataset loaders."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

import anndata as ad

from scgraph_bench.config.dataset import DatasetConfig


class BaseDatasetLoader(ABC):
    """Abstract dataset loader interface."""

    def __init__(self, cache_dir: Path | str | None = None) -> None:
        self.cache_dir = Path(cache_dir) if cache_dir else Path("data/raw")
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    @abstractmethod
    def load(
        self,
        config: DatasetConfig,
        dev_subsample_per_donor: int | None = None,
        seed: int = 42,
    ) -> ad.AnnData:
        """Load, validate, and return the AnnData object.

        Args:
            config: Dataset configuration specification.
            dev_subsample_per_donor: Optional max cells per donor for fast development smoke testing.
            seed: Seed for deterministic subsampling.

        Returns:
            Validated AnnData instance with raw counts in .X, standardized .obs columns.
        """
        raise NotImplementedError
