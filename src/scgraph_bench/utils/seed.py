"""Centralized seed management utility for reproducible benchmarks."""

from __future__ import annotations

import os
import random
from collections.abc import Generator
from contextlib import contextmanager

import numpy as np
import torch


def set_seed(seed: int, deterministic_torch: bool = True) -> None:
    """Set random seed across Python random, NumPy, and PyTorch.

    Args:
        seed: Integer seed value.
        deterministic_torch: Whether to configure PyTorch for deterministic algorithms.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    os.environ["PYTHONHASHSEED"] = str(seed)

    if deterministic_torch:
        torch.use_deterministic_algorithms(True, warn_only=True)


def get_rng(seed: int | None = None) -> np.random.Generator:
    """Get a NumPy random Generator instance for isolated stochastic operations.

    Args:
        seed: Optional integer seed for the generator.

    Returns:
        np.random.Generator instance.
    """
    return np.random.default_rng(seed)


@contextmanager
def SeedContext(seed: int) -> Generator[None, None, None]:
    """Context manager for temporary localized seed scoping.

    Restores previous random states for random, numpy, and torch on exit.

    Args:
        seed: Seed to apply within the context.
    """
    # Capture states
    py_state = random.getstate()
    np_state = np.random.get_state()
    torch_state = torch.get_rng_state()

    try:
        set_seed(seed, deterministic_torch=False)
        yield
    finally:
        # Restore states
        random.setstate(py_state)
        np.random.set_state(np_state)
        torch.set_rng_state(torch_state)
