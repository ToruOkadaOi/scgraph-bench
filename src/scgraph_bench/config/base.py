"""Base configuration model with YAML serialization and hashing."""

from __future__ import annotations

from pathlib import Path
from typing import Any, TypeVar

import yaml
from pydantic import BaseModel, ConfigDict

from scgraph_bench.utils.hashing import hash_config

T = TypeVar("T", bound="BaseBenchConfig")


class BaseBenchConfig(BaseModel):
    """Base Pydantic model for benchmark configuration components."""

    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True,
        frozen=True,
    )

    def compute_hash(self) -> str:
        """Compute deterministic SHA-256 hash of this configuration."""
        return hash_config(self)

    def to_yaml(self, path: str | Path | None = None) -> str:
        """Export configuration to YAML string or save to file.

        Args:
            path: Optional file path to save YAML.

        Returns:
            YAML formatted string.
        """
        data = self.model_dump(mode="json")
        yaml_str = yaml.safe_dump(data, sort_keys=False)
        if path:
            p = Path(path)
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(yaml_str, encoding="utf-8")
        return yaml_str

    @classmethod
    def from_yaml(cls: type[T], path: str | Path) -> T:
        """Load configuration from a YAML file.

        Args:
            path: Path to YAML file.

        Returns:
            Instantiated configuration object.
        """
        p = Path(path)
        if not p.is_file():
            raise FileNotFoundError(f"Configuration file not found: {path}")
        content = p.read_text(encoding="utf-8")
        data: dict[str, Any] = yaml.safe_load(content) or {}
        return cls.model_validate(data)
