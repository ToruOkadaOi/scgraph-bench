"""Split definition schema and serialization."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from scgraph_bench.config.split import SplitType
from scgraph_bench.utils.hashing import hash_dict


class SplitDefinition(BaseModel):
    """Frozen dataset split specification.

    Persisted to splits/{dataset_name}/{split_id}.json and committed to Git.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    dataset_name: str
    split_id: str
    split_type: SplitType
    seed: int
    train_donors: list[str]
    val_donors: list[str]
    test_donors: list[str]
    train_cell_ids: list[str]
    val_cell_ids: list[str]
    test_cell_ids: list[str]
    site_composition: dict[str, dict[str, int]] = Field(
        default_factory=dict,
        description="Donor counts per site across partitions (train, val, test).",
    )
    label_support: dict[str, dict[str, int]] = Field(
        default_factory=dict,
        description="Cell counts per label across partitions.",
    )
    total_cells: int
    total_donors: int
    config_hash: str
    created_at_utc: str = Field(
        default_factory=lambda: datetime.now(UTC).isoformat(),
    )

    def validate_disjointness(self) -> None:
        """Assert that train, validation, and test donor and cell sets are strictly disjoint."""
        train_d, val_d, test_d = set(self.train_donors), set(self.val_donors), set(self.test_donors)
        if not train_d.isdisjoint(val_d):
            raise ValueError(f"Train and Val donors overlap: {train_d & val_d}")
        if not train_d.isdisjoint(test_d):
            raise ValueError(f"Train and Test donors overlap: {train_d & test_d}")
        if not val_d.isdisjoint(test_d):
            raise ValueError(f"Val and Test donors overlap: {val_d & test_d}")

        train_c, val_c, test_c = (
            set(self.train_cell_ids),
            set(self.val_cell_ids),
            set(self.test_cell_ids),
        )
        if not train_c.isdisjoint(val_c):
            raise ValueError("Train and Val cell IDs overlap!")
        if not train_c.isdisjoint(test_c):
            raise ValueError("Train and Test cell IDs overlap!")
        if not val_c.isdisjoint(test_c):
            raise ValueError("Val and Test cell IDs overlap!")

    def compute_artifact_hash(self) -> str:
        """Compute deterministic hash of the split assignments."""
        payload = {
            "dataset_name": self.dataset_name,
            "split_id": self.split_id,
            "seed": self.seed,
            "train_cell_ids": self.train_cell_ids,
            "val_cell_ids": self.val_cell_ids,
            "test_cell_ids": self.test_cell_ids,
            "train_donors": self.train_donors,
            "val_donors": self.val_donors,
            "test_donors": self.test_donors,
        }
        return hash_dict(payload)

    def save_json(self, path: Path | str) -> None:
        """Save split definition to JSON file."""
        self.validate_disjointness()
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        data = self.model_dump(mode="json")
        p.write_text(json.dumps(data, indent=2), encoding="utf-8")

    @classmethod
    def load_json(cls, path: Path | str) -> SplitDefinition:
        """Load split definition from JSON file."""
        p = Path(path)
        if not p.is_file():
            raise FileNotFoundError(f"Split file not found: {path}")
        data: dict[str, Any] = json.loads(p.read_text(encoding="utf-8"))
        instance = cls.model_validate(data)
        instance.validate_disjointness()
        return instance
