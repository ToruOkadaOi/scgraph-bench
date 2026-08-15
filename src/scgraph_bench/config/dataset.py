"""Dataset configuration schemas."""

from __future__ import annotations

from scgraph_bench.config.base import BaseBenchConfig


class MetadataConstraintConfig(BaseBenchConfig):
    """Quality and representation constraints on dataset metadata."""

    min_cells_per_donor: int = 50
    min_donors_per_class: int = 2
    allow_missing_donors: bool = False


class DatasetConfig(BaseBenchConfig):
    """Configuration for dataset loading and validation."""

    name: str = "kang_pbmc"
    description: str = "Kang et al. 10x PBMC dataset (Development reference)"
    cell_id_key: str = "cell_id"
    label_key: str = "cell_type"
    donor_key: str = "donor_id"
    batch_key: str | None = "condition"
    constraints: MetadataConstraintConfig = MetadataConstraintConfig()
