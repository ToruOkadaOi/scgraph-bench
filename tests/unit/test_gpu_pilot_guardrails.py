"""Unit tests for GPU pilot execution guardrails, CUDA preflight check, and --confirm-paid-gpu-run."""

from unittest.mock import MagicMock

import pytest
import torch
from scripts.run_gpu_pilot import (
    perform_cuda_hardware_check,
    run_gpu_pilot,
    validate_device_argument,
)


def test_cuda_unavailable_exits_nonzero_before_data_or_model(monkeypatch):
    """Verify that perform_cuda_hardware_check exits with SystemExit(1) when CUDA is unavailable."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    with pytest.raises(SystemExit) as exc_info:
        perform_cuda_hardware_check()

    assert exc_info.value.code == 1


@pytest.mark.parametrize("invalid_device", ["cpu", "mps", "auto", "cuda:1"])
def test_invalid_device_argument_rejected_immediately(invalid_device):
    """Verify that any device argument other than exactly 'cuda' raises SystemExit(1)."""
    with pytest.raises(SystemExit) as exc_info:
        validate_device_argument(requested_device=invalid_device)

    assert exc_info.value.code == 1


def test_valid_device_argument_accepted():
    """Verify that exactly 'cuda' passes device validation."""
    validate_device_argument(requested_device="cuda")


def test_missing_confirm_paid_gpu_run_performs_dry_run_zero_artifacts(monkeypatch):
    """Verify that without --confirm-paid-gpu-run, execution completes dry-run without requiring CUDA or loading datasets."""
    # Ensure CUDA is False to prove dry-run does NOT require CUDA hardware
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    # Mock data loader to verify it is NEVER called during dry run
    loader_mock = MagicMock()
    monkeypatch.setattr("scripts.run_gpu_pilot.StephensonHealthyPBMCLoader", loader_mock)

    # Run without confirmation flag
    run_gpu_pilot(
        dataset_name="stephenson_2021_healthy_pbmc",
        split_id="site_stratified_seed42",
        seed=42,
        device="cuda",
        confirm_paid_gpu_run=False,
    )

    # Assert data loader was not called and no training occurred
    loader_mock.assert_not_called()


def test_cuda_device_plus_confirmation_permits_execution(monkeypatch):
    """Verify that with CUDA hardware present and confirmation flag True, the execution path proceeds to data loading."""
    monkeypatch.setattr(
        "scripts.run_gpu_pilot.perform_cuda_hardware_check",
        lambda *_args, **_kwargs: torch.device("cpu"),
    )
    loader_mock = MagicMock()
    loader_mock.return_value.load.side_effect = RuntimeError("Reached data load successfully")
    monkeypatch.setattr("scripts.run_gpu_pilot.StephensonHealthyPBMCLoader", loader_mock)

    with pytest.raises(RuntimeError, match="Reached data load successfully"):
        run_gpu_pilot(
            dataset_name="stephenson_2021_healthy_pbmc",
            split_id="site_stratified_seed42",
            seed=42,
            device="cuda",
            confirm_paid_gpu_run=True,
        )
