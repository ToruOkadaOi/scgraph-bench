"""Data loading, dataset registry, and schema validation."""

from scgraph_bench.data.base import BaseDatasetLoader
from scgraph_bench.data.loaders import (
    PRIMARY_V0_LABELS_STEPHENSON,
    StephensonHealthyPBMCLoader,
    SyntheticFixtureLoader,
)
from scgraph_bench.data.registry import (
    get_dataset_loader,
    list_registered_datasets,
    register_dataset,
)
from scgraph_bench.data.validation import DatasetValidationError, validate_anndata_schema

__all__ = [
    "BaseDatasetLoader",
    "StephensonHealthyPBMCLoader",
    "SyntheticFixtureLoader",
    "PRIMARY_V0_LABELS_STEPHENSON",
    "register_dataset",
    "get_dataset_loader",
    "list_registered_datasets",
    "validate_anndata_schema",
    "DatasetValidationError",
]
