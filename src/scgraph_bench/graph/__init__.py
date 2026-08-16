"""Graph construction and PyG container serialization interfaces."""

from scgraph_bench.graph.base import BaseGraphBuilder
from scgraph_bench.graph.bbknn import StrictInductiveBBKNNGraphBuilder
from scgraph_bench.graph.mutual_knn import MutualKNNGraphBuilder
from scgraph_bench.graph.pca_knn import PCAkNNGraphBuilder
from scgraph_bench.graph.registry import (
    get_graph_builder,
    list_registered_graph_builders,
    register_graph_builder,
)
from scgraph_bench.graph.rewired_control import RewiredControlGraphBuilder
from scgraph_bench.graph.schema import GraphBundle, GraphManifest

__all__ = [
    "BaseGraphBuilder",
    "GraphBundle",
    "GraphManifest",
    "MutualKNNGraphBuilder",
    "PCAkNNGraphBuilder",
    "RewiredControlGraphBuilder",
    "StrictInductiveBBKNNGraphBuilder",
    "get_graph_builder",
    "list_registered_graph_builders",
    "register_graph_builder",
]
