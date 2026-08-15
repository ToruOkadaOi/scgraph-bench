"""Master benchmark configuration assembling all pipeline components."""

from __future__ import annotations

from scgraph_bench.config.base import BaseBenchConfig
from scgraph_bench.config.dataset import DatasetConfig
from scgraph_bench.config.graph import GraphConfig
from scgraph_bench.config.model import ModelConfig
from scgraph_bench.config.preprocessing import PreprocessingConfig
from scgraph_bench.config.split import SplitConfig


class TrackingConfig(BaseBenchConfig):
    """Experiment and metrics tracking configuration."""

    experiment_name: str = "scgraph_benchmark_v0"
    tracking_uri: str | None = None
    save_csv: bool = True
    save_parquet: bool = True


class BenchmarkConfig(BaseBenchConfig):
    """Top-level benchmark configuration."""

    experiment_id: str = "benchmark_run"
    seed: int = 42
    dataset: DatasetConfig = DatasetConfig()
    split: SplitConfig = SplitConfig()
    preprocessing: PreprocessingConfig = PreprocessingConfig()
    graph: GraphConfig = GraphConfig()
    model: ModelConfig = ModelConfig()
    tracking: TrackingConfig = TrackingConfig()
