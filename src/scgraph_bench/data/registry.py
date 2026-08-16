"""Dataset registry for discovery and instantiation of loaders."""

from __future__ import annotations

from collections.abc import Callable
from typing import TypeVar

from scgraph_bench.data.base import BaseDatasetLoader

T = TypeVar("T", bound=type[BaseDatasetLoader])

_DATASET_REGISTRY: dict[str, type[BaseDatasetLoader]] = {}


def register_dataset(name: str) -> Callable[[T], T]:
    """Decorator to register a dataset loader class by unique name."""

    def decorator(cls: T) -> T:
        if name in _DATASET_REGISTRY:
            raise KeyError(f"Dataset loader '{name}' is already registered.")
        _DATASET_REGISTRY[name] = cls
        return cls

    return decorator


def get_dataset_loader(name: str, **kwargs) -> BaseDatasetLoader:
    """Retrieve and instantiate a registered dataset loader by name.

    Args:
        name: Registered identifier of the dataset loader.
        **kwargs: Arguments forwarded to the loader constructor.

    Returns:
        Instantiated BaseDatasetLoader.
    """
    if name not in _DATASET_REGISTRY:
        available = ", ".join(sorted(_DATASET_REGISTRY.keys()))
        raise KeyError(f"Dataset '{name}' not found in registry. Available datasets: {available}")
    return _DATASET_REGISTRY[name](**kwargs)


def list_registered_datasets() -> list[str]:
    """List all registered dataset loader identifiers."""
    return sorted(_DATASET_REGISTRY.keys())
