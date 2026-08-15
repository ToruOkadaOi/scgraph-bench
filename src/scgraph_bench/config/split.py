"""Split configuration schemas and constraints."""

from __future__ import annotations

from enum import StrEnum

from pydantic import Field, model_validator

from scgraph_bench.config.base import BaseBenchConfig


class SplitType(StrEnum):
    """Supported dataset partition strategies."""

    DONOR_HELD_OUT = "donor_held_out"
    RANDOM_CELL_DEBUG = "random_cell_debug"


class RareClassAction(StrEnum):
    """Behavior when rare cell types are encountered."""

    WARN_AND_RETAIN = "warn_and_retain"
    EXCLUDE = "exclude"
    FAIL = "fail"


class RareClassConfig(BaseBenchConfig):
    """Configuration for handling rare or unbalanced cell types across splits."""

    min_train_cells_per_class: int = Field(
        default=10,
        description="Minimum cells required in training partition to consider class supported.",
    )
    min_test_cells_per_class: int = Field(
        default=5,
        description="Minimum cells required in test partition for valid evaluation.",
    )
    action: RareClassAction = Field(
        default=RareClassAction.WARN_AND_RETAIN,
        description="Action to take when a cell type is missing or below threshold in training.",
    )


class SplitConfig(BaseBenchConfig):
    """Configuration for frozen dataset splitting."""

    split_id: str = "donor_split_seed42"
    split_type: SplitType = SplitType.DONOR_HELD_OUT
    train_fraction: float = Field(default=0.6, ge=0.1, le=0.9)
    val_fraction: float = Field(default=0.2, ge=0.05, le=0.5)
    test_fraction: float = Field(default=0.2, ge=0.05, le=0.5)
    seed: int = 42
    rare_class_config: RareClassConfig = RareClassConfig()
    custom_train_donors: list[str] | None = None
    custom_val_donors: list[str] | None = None
    custom_test_donors: list[str] | None = None

    @model_validator(mode="after")
    def validate_fractions(self) -> SplitConfig:
        total = self.train_fraction + self.val_fraction + self.test_fraction
        if not abs(total - 1.0) < 1e-5:
            raise ValueError(f"Split fractions must sum to 1.0, got sum={total:.4f}")
        return self
