"""Baseline models and classification interfaces."""

from scgraph_bench.models.base import BaseClassifier
from scgraph_bench.models.gcn import GCNClassifier, GCNConfig, GCNNet
from scgraph_bench.models.graphsage import (
    GraphSAGEClassifier,
    GraphSAGEConfig,
    GraphSAGENet,
)
from scgraph_bench.models.logistic_regression import (
    LogisticRegressionBaseline,
    LogisticRegressionConfig,
)
from scgraph_bench.models.mlp import MLPBaseline, MLPConfig, PyTorchMLP

__all__ = [
    "BaseClassifier",
    "GCNClassifier",
    "GCNConfig",
    "GCNNet",
    "GraphSAGEClassifier",
    "GraphSAGEConfig",
    "GraphSAGENet",
    "LogisticRegressionBaseline",
    "LogisticRegressionConfig",
    "MLPBaseline",
    "MLPConfig",
    "PyTorchMLP",
]
