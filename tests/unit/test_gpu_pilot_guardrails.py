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
    """Verify that without --confirm-paid-gpu-run, execution completes dry-run without requiring CUDA or calling execute_pilot."""
    # Ensure CUDA is False to prove dry-run does NOT require CUDA hardware
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    # Spy on execute_pilot to verify it is NEVER called during dry run
    mock_execute = MagicMock()
    monkeypatch.setattr("scripts.run_gpu_pilot.execute_pilot", mock_execute)

    # Run without confirmation flag
    run_gpu_pilot(
        dataset_name="stephenson_2021_healthy_pbmc",
        split_id="site_stratified_seed42",
        seed=42,
        device="cuda",
        confirm_paid_gpu_run=False,
    )

    # Assert execute_pilot was not called and no training occurred
    mock_execute.assert_not_called()


def test_cuda_device_plus_confirmation_permits_execution(monkeypatch):
    """Verify that with CUDA hardware present and confirmation flag True, execute_pilot is called with torch.device('cuda:0')."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda _idx: "NVIDIA RTX 4090")
    monkeypatch.setattr(
        torch.cuda,
        "get_device_properties",
        lambda _idx: MagicMock(total_memory=25769803776, multi_processor_count=128),
    )

    mock_execute = MagicMock()
    monkeypatch.setattr("scripts.run_gpu_pilot.execute_pilot", mock_execute)

    run_gpu_pilot(
        dataset_name="stephenson_2021_healthy_pbmc",
        split_id="site_stratified_seed42",
        seed=42,
        device="cuda",
        confirm_paid_gpu_run=True,
    )

    mock_execute.assert_called_once_with(
        dataset_name="stephenson_2021_healthy_pbmc",
        split_id="site_stratified_seed42",
        seed=42,
        target_device=torch.device("cuda:0"),
    )


def test_cuda_unavailable_with_confirmation_exits_nonzero_zero_artifact_access(monkeypatch):
    """Verify that when confirmation is present but CUDA is unavailable, execution halts with exit 1 before execute_pilot."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    mock_execute = MagicMock()
    monkeypatch.setattr("scripts.run_gpu_pilot.execute_pilot", mock_execute)

    with pytest.raises(SystemExit) as exc_info:
        run_gpu_pilot(
            dataset_name="stephenson_2021_healthy_pbmc",
            split_id="site_stratified_seed42",
            seed=42,
            device="cuda",
            confirm_paid_gpu_run=True,
        )

    assert exc_info.value.code == 1
    mock_execute.assert_not_called()


@pytest.mark.parametrize("invalid_device", ["cpu", "mps", "auto"])
def test_invalid_device_with_confirmation_exits_nonzero(invalid_device, monkeypatch):
    """Verify that invalid device argument with confirmation flag halts with exit 1 before execute_pilot."""
    mock_execute = MagicMock()
    monkeypatch.setattr("scripts.run_gpu_pilot.execute_pilot", mock_execute)

    with pytest.raises(SystemExit) as exc_info:
        run_gpu_pilot(
            dataset_name="stephenson_2021_healthy_pbmc",
            split_id="site_stratified_seed42",
            seed=42,
            device=invalid_device,
            confirm_paid_gpu_run=True,
        )

    assert exc_info.value.code == 1
    mock_execute.assert_not_called()
