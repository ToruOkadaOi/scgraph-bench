"""Validation utilities for single-cell AnnData objects."""

from __future__ import annotations

import anndata as ad
import numpy as np
from scipy import sparse

from scgraph_bench.config.dataset import DatasetConfig


class DatasetValidationError(ValueError):
    """Raised when an AnnData object fails benchmark data validation rules."""


def validate_anndata_schema(
    adata: ad.AnnData,
    config: DatasetConfig,
    require_raw_integer_counts: bool = True,
) -> None:
    """Validate that an AnnData object strictly complies with benchmark requirements.

    Args:
        adata: AnnData object to validate.
        config: DatasetConfig specifying expected metadata keys and constraints.
        require_raw_integer_counts: Whether to enforce raw non-negative integer counts.

    Raises:
        DatasetValidationError: When schema or data integrity invariants are violated.
    """
    # 1. Non-empty check
    if adata.n_obs == 0 or adata.n_vars == 0:
        raise DatasetValidationError(f"AnnData is empty: shape={adata.shape}")

    # 2. Required columns in obs
    required_keys = [config.cell_id_key, config.donor_key, config.label_key]
    if config.batch_key:
        required_keys.append(config.batch_key)

    missing_keys = [k for k in required_keys if k not in adata.obs.columns]
    if missing_keys:
        raise DatasetValidationError(
            f"Missing required metadata columns in adata.obs: {missing_keys}. "
            f"Present columns: {list(adata.obs.columns)}"
        )

    # 3. Unique cell identifiers
    cell_ids = adata.obs[config.cell_id_key].values
    if len(cell_ids) != len(set(cell_ids)):
        raise DatasetValidationError("Duplicate cell IDs found in metadata.")
    if not (adata.obs_names == cell_ids).all():
        raise DatasetValidationError("adata.obs_names does not match cell_id column exactly.")

    # 4. Null checks in donor and label
    null_donors = adata.obs[config.donor_key].isna().sum()
    if null_donors > 0:
        raise DatasetValidationError(f"Found {null_donors} cells with null/NaN donor IDs.")

    null_labels = adata.obs[config.label_key].isna().sum()
    if null_labels > 0:
        raise DatasetValidationError(f"Found {null_labels} cells with null/NaN labels.")

    # 5. Raw counts verification
    if require_raw_integer_counts:
        x_mat = adata.X
        if sparse.issparse(x_mat):
            data_arr = np.asarray(x_mat.data)
            if not np.all(data_arr >= 0):
                raise DatasetValidationError("Expression matrix contains negative count values.")
            if not np.all(data_arr % 1 == 0):
                raise DatasetValidationError(
                    "Expression matrix contains non-integer values; expected raw unnormalized counts."
                )
        elif x_mat is not None:
            data_arr = np.asarray(x_mat)
            if not np.all(data_arr >= 0):
                raise DatasetValidationError("Expression matrix contains negative count values.")
            if not np.all(data_arr % 1 == 0):
                raise DatasetValidationError(
                    "Expression matrix contains non-integer values; expected raw unnormalized counts."
                )

    # 6. Minimum cells per donor constraint
    donor_counts = adata.obs[config.donor_key].value_counts()
    min_observed = donor_counts.min()
    if min_observed < config.constraints.min_cells_per_donor:
        violating_donors = donor_counts[
            donor_counts < config.constraints.min_cells_per_donor
        ].to_dict()
        raise DatasetValidationError(
            f"Donors below minimum threshold ({config.constraints.min_cells_per_donor} cells): "
            f"{violating_donors}"
        )
