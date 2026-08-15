"""Unit tests for seed management and determinism."""

import random

import numpy as np
import torch

from scgraph_bench.utils.seed import SeedContext, get_rng, set_seed


def test_set_seed_determinism():
    """Verify set_seed produces identical random sequences across random, numpy, and torch."""
    set_seed(12345)
    py_r1 = [random.random() for _ in range(5)]
    np_r1 = np.random.uniform(size=5).tolist()
    th_r1 = torch.rand(5).tolist()

    set_seed(12345)
    py_r2 = [random.random() for _ in range(5)]
    np_r2 = np.random.uniform(size=5).tolist()
    th_r2 = torch.rand(5).tolist()

    assert py_r1 == py_r2
    assert np.allclose(np_r1, np_r2)
    assert np.allclose(th_r1, th_r2)


def test_get_rng_isolation():
    """Verify get_rng creates independent deterministic generators."""
    rng1 = get_rng(42)
    rng2 = get_rng(42)

    samples1 = rng1.standard_normal(10)
    samples2 = rng2.standard_normal(10)

    assert np.allclose(samples1, samples2)


def test_seed_context_restores_state():
    """Verify SeedContext isolates inner seed without corrupting outer RNG state."""
    set_seed(100)
    outer_before = random.random()

    with SeedContext(999):
        _ = random.random()
        _ = np.random.rand()
        _ = torch.rand(1)

    outer_after = random.random()

    # Replay sequence with seed 100
    set_seed(100)
    ref_before = random.random()
    ref_after = random.random()

    assert outer_before == ref_before
    assert outer_after == ref_after
