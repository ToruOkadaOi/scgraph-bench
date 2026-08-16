"""Model configurations for classical baselines and CPU MLP smoke tests."""

from __future__ import annotations

from enum import StrEnum

from pydantic import Field

from scgraph_bench.config.base import BaseBenchConfig


class ModelType(StrEnum):
    """Supported baseline model architectures."""

    LOGISTIC_REGRESSION = "logistic_regression"
    MLP = "mlp"


class LogisticRegressionConfig(BaseBenchConfig):
    """Configuration for CPU-only Scikit-learn Logistic Regression baseline."""

    penalty: str = "l2"
    solver: str = "lbfgs"
    c_grid: list[float] = Field(
        default_factory=lambda: [0.01, 0.1, 1.0, 10.0],
        description="Regularization C candidate grid evaluated strictly on validation macro-F1.",
    )
    class_weight: str | None = "balanced"
    class_weight_grid: list[str | None] = Field(
        default_factory=lambda: [None, "balanced"],
        description="Candidate class weighting schemes evaluated on validation set.",
    )
    max_iter: int = 1000
    random_state: int = 42


class MLPConfig(BaseBenchConfig):
    """Configuration for PyTorch feature-only MLP baseline."""

    input_dim: int = 50
    hidden_dims: list[int] = Field(default_factory=lambda: [128, 128])
    num_classes: int = 12
    activation: str = "relu"
    dropout: float = 0.3
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    batch_size: int = 256
    max_epochs: int = 500
    patience: int = 50
    device: str = "cpu"
    seed: int = 42


class ModelConfig(BaseBenchConfig):
    """Composite baseline model configuration."""

    model_type: ModelType = ModelType.LOGISTIC_REGRESSION
    logistic_regression: LogisticRegressionConfig | None = LogisticRegressionConfig()
    mlp: MLPConfig | None = None
