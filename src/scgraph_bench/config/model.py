"""Model configurations for baselines and smoke testing."""

from __future__ import annotations

from enum import StrEnum

from pydantic import Field

from scgraph_bench.config.base import BaseBenchConfig


class ModelType(StrEnum):
    """Supported CPU baseline models."""

    LOGISTIC_REGRESSION = "logistic_regression"
    MLP = "mlp"


class LogisticRegressionConfig(BaseBenchConfig):
    """Configuration for Scikit-learn Logistic Regression baseline."""

    penalty: str = "l2"
    solver: str = "lbfgs"
    c_grid: list[float] = Field(
        default_factory=lambda: [1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0],
        description="Regularization C values evaluated on validation set.",
    )
    max_iter: int = 1000
    class_weight: str | None = "balanced"
    random_state: int = 42


class MLPConfig(BaseBenchConfig):
    """Configuration for PyTorch MLP baseline (CPU-compatible smoke test architecture)."""

    hidden_dims: list[int] = Field(default_factory=lambda: [128, 128])
    activation: str = "relu"
    dropout: float = 0.3
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    batch_size: int = 256
    max_epochs: int = 100
    patience: int = 10
    device: str = "cpu"
    seed: int = 42


class ModelConfig(BaseBenchConfig):
    """Composite model configuration."""

    model_type: ModelType = ModelType.LOGISTIC_REGRESSION
    logistic_regression: LogisticRegressionConfig | None = LogisticRegressionConfig()
    mlp: MLPConfig | None = None
