"""Cryptographic hashing utilities for artifact verification and config tracking."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
from pydantic import BaseModel


def hash_bytes(data: bytes) -> str:
    """Calculate SHA-256 hash of bytes.

    Args:
        data: Raw byte string.

    Returns:
        Hexadecimal SHA-256 digest string.
    """
    hasher = hashlib.sha256()
    hasher.update(data)
    return hasher.hexdigest()


def hash_string(text: str) -> str:
    """Calculate SHA-256 hash of a string."""
    return hash_bytes(text.encode("utf-8"))


def hash_dict(data: dict[str, Any]) -> str:
    """Calculate deterministic SHA-256 hash of a dictionary.

    Keys are sorted and values serialized to standard JSON.
    """
    serialized = json.dumps(data, sort_keys=True, default=str)
    return hash_string(serialized)


def hash_config(model: BaseModel) -> str:
    """Calculate deterministic SHA-256 hash of a Pydantic model."""
    serialized = model.model_dump_json()
    # Parse back and dump sorted to ensure key ordering invariance
    parsed = json.loads(serialized)
    return hash_dict(parsed)


def hash_file(file_path: str | Path, chunk_size: int = 65536) -> str:
    """Calculate SHA-256 hash of a file on disk.

    Args:
        file_path: Path to the target file.
        chunk_size: Block size for chunked reading.

    Returns:
        Hexadecimal SHA-256 digest string.
    """
    path = Path(file_path)
    if not path.is_file():
        raise FileNotFoundError(f"File not found for hashing: {file_path}")

    hasher = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(chunk_size):
            hasher.update(chunk)
    return hasher.hexdigest()


def hash_array(array: np.ndarray) -> str:
    """Calculate deterministic SHA-256 hash of a NumPy array.

    Includes array shape, dtype, and contiguous byte data.
    """
    hasher = hashlib.sha256()
    hasher.update(str(array.shape).encode("utf-8"))
    hasher.update(str(array.dtype).encode("utf-8"))
    hasher.update(np.ascontiguousarray(array).tobytes())
    return hasher.hexdigest()
