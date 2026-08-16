"""Registry for discovering and instantiating graph builders."""

from __future__ import annotations

from collections.abc import Callable
from typing import TypeVar

from scgraph_bench.graph.base import BaseGraphBuilder

T = TypeVar("T", bound=type[BaseGraphBuilder])

_GRAPH_BUILDER_REGISTRY: dict[str, type[BaseGraphBuilder]] = {}


def register_graph_builder(name: str) -> Callable[[T], T]:
    """Decorator to register a graph builder class by unique algorithm name."""

    def decorator(cls: T) -> T:
        if name in _GRAPH_BUILDER_REGISTRY:
            raise KeyError(f"Graph builder '{name}' is already registered.")
        _GRAPH_BUILDER_REGISTRY[name] = cls
        return cls

    return decorator


def get_graph_builder(name: str, **kwargs) -> BaseGraphBuilder:
    """Retrieve and instantiate a registered graph builder by name."""
    if name not in _GRAPH_BUILDER_REGISTRY:
        available = ", ".join(sorted(_GRAPH_BUILDER_REGISTRY.keys()))
        raise KeyError(
            f"Graph builder '{name}' not found in registry. Available builders: {available}"
        )
    return _GRAPH_BUILDER_REGISTRY[name](**kwargs)


def list_registered_graph_builders() -> list[str]:
    """List all registered graph builder identifiers."""
    return sorted(_GRAPH_BUILDER_REGISTRY.keys())
