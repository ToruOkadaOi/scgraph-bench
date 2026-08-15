"""Unit tests for cryptographic artifact and config hashing."""

import numpy as np
import pytest

from scgraph_bench.config.preprocessing import PreprocessingConfig
from scgraph_bench.utils.hashing import (
    hash_array,
    hash_config,
    hash_dict,
    hash_file,
    hash_string,
)


def test_hash_string():
    """Verify hash_string determinism and length."""
    h1 = hash_string("test_string")
    h2 = hash_string("test_string")
    h3 = hash_string("different_string")

    assert h1 == h2
    assert len(h1) == 64
    assert h1 != h3


def test_hash_dict_key_order_invariance():
    """Verify hash_dict produces identical hashes regardless of dictionary key insertion order."""
    d1 = {"b": 2, "a": 1, "c": [1, 2, 3]}
    d2 = {"a": 1, "c": [1, 2, 3], "b": 2}

    assert hash_dict(d1) == hash_dict(d2)


def test_hash_config_determinism():
    """Verify Pydantic model configuration hashing."""
    c1 = PreprocessingConfig(n_top_genes=2000, n_comps=50)
    c2 = PreprocessingConfig(n_top_genes=2000, n_comps=50)
    c3 = PreprocessingConfig(n_top_genes=1000, n_comps=50)

    assert hash_config(c1) == hash_config(c2)
    assert hash_config(c1) != hash_config(c3)
    assert c1.compute_hash() == hash_config(c1)


def test_hash_array():
    """Verify hash_array captures values, shapes, and dtypes."""
    a1 = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    a2 = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    a3 = np.array([1.0, 2.0, 3.0], dtype=np.float64)
    a4 = np.array([[1.0], [2.0], [3.0]], dtype=np.float32)

    assert hash_array(a1) == hash_array(a2)
    assert hash_array(a1) != hash_array(a3)  # different dtype
    assert hash_array(a1) != hash_array(a4)  # different shape


def test_hash_file(tmp_path):
    """Verify file hashing and file-not-found error handling."""
    f = tmp_path / "sample.txt"
    f.write_text("benchmark data content", encoding="utf-8")

    h = hash_file(f)
    assert len(h) == 64

    # Different content produces different hash
    f.write_text("modified data content", encoding="utf-8")
    assert hash_file(f) != h

    with pytest.raises(FileNotFoundError):
        hash_file(tmp_path / "nonexistent.txt")
