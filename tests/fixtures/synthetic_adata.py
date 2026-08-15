"""Synthetic multi-donor scRNA-seq AnnData generator for CPU tests."""

from __future__ import annotations

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse


def generate_synthetic_scrna_adata(
    n_cells: int = 300,
    n_genes: int = 100,
    n_donors: int = 6,
    n_classes: int = 4,
    n_batches: int = 2,
    seed: int = 42,
    sparse_format: bool = True,
) -> ad.AnnData:
    """Generate a lightweight synthetic multi-donor scRNA-seq AnnData object.

    Args:
        n_cells: Number of cells.
        n_genes: Number of genes.
        n_donors: Number of donors.
        n_classes: Number of distinct cell-type classes.
        n_batches: Number of distinct conditions/batches.
        seed: Random seed for reproducibility.
        sparse_format: Whether to store X as scipy.sparse.csr_matrix.

    Returns:
        AnnData object with valid counts, obs metadata, and var index.
    """
    rng = np.random.default_rng(seed)

    donors = [f"donor_{i}" for i in range(n_donors)]
    cell_types = [f"cell_type_{i}" for i in range(n_classes)]
    conditions = [f"condition_{i}" for i in range(n_batches)]

    # Assign metadata to cells
    cell_donors = rng.choice(donors, size=n_cells)
    cell_labels = rng.choice(cell_types, size=n_cells)
    cell_conditions = rng.choice(conditions, size=n_cells)
    cell_ids = [f"cell_{i:04d}" for i in range(n_cells)]

    # Generate synthetic count matrix with cluster-specific gene expression signals
    base_expr = rng.negative_binomial(n=5, p=0.3, size=(n_cells, n_genes)).astype(np.float32)

    # Add class-specific marker shifts
    genes_per_class = max(2, n_genes // (n_classes + 1))
    for c_idx, c_label in enumerate(cell_types):
        mask = cell_labels == c_label
        start_g = c_idx * genes_per_class
        end_g = min(n_genes, (c_idx + 1) * genes_per_class)
        base_expr[mask, start_g:end_g] += rng.poisson(
            lam=15.0, size=(np.sum(mask), end_g - start_g)
        ).astype(np.float32)

    # Add donor batch effect
    for _d_idx, d_name in enumerate(donors):
        mask = cell_donors == d_name
        donor_shift = rng.poisson(lam=1.0, size=(1, n_genes)).astype(np.float32)
        base_expr[mask] = np.maximum(0, base_expr[mask] + donor_shift)

    base_expr = np.round(base_expr).astype(np.float32)
    counts = sparse.csr_matrix(base_expr) if sparse_format else base_expr

    obs_df = pd.DataFrame(
        {
            "cell_id": cell_ids,
            "donor_id": cell_donors,
            "cell_type": cell_labels,
            "condition": cell_conditions,
        },
        index=cell_ids,
    )

    var_df = pd.DataFrame(
        {
            "gene_name": [f"gene_{j:04d}" for j in range(n_genes)],
        },
        index=[f"gene_{j:04d}" for j in range(n_genes)],
    )

    adata = ad.AnnData(X=counts, obs=obs_df, var=var_df)
    return adata
