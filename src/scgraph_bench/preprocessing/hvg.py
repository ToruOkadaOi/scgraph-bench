"""Highly Variable Gene (HVG) selection methods fitted strictly on training cells."""

from __future__ import annotations

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
from scipy import sparse

from scgraph_bench.utils.logging import get_logger

logger = get_logger("preprocessing.hvg")


def select_seurat_hvgs_train_only(
    X_norm_log1p_train: sparse.spmatrix | np.ndarray,
    gene_names: list[str],
    n_top_genes: int = 2000,
) -> tuple[np.ndarray, list[str]]:
    """Select highly variable genes using the explicit 'seurat' flavor fitted strictly on training cells.

    This function computes gene means, dispersions, and bin-normalized dispersion z-scores
    exclusively on the training cell population. Validation and test cell profiles are
    never accessed or observed during this calculation.

    Args:
        X_norm_log1p_train: Training expression matrix after library size normalisation and log1p.
        gene_names: Complete list of raw gene identifiers or symbols corresponding to columns.
        n_top_genes: Target number of HVGs to select (default: 2,000).

    Returns:
        tuple containing:
            - hvg_indices: np.ndarray of integer column indices of selected genes.
            - hvg_gene_names: list of string names of selected genes.
    """
    n_cells, n_genes = X_norm_log1p_train.shape
    if n_genes <= n_top_genes:
        logger.warning(
            "Requested %d HVGs, but dataset has only %d genes. Selecting all genes.",
            n_top_genes,
            n_genes,
        )
        indices = np.arange(n_genes)
        return indices, list(gene_names)

    # Construct temporary AnnData for training slice only
    adata_train = ad.AnnData(
        X=X_norm_log1p_train,
        var=pd.DataFrame(index=pd.Index(gene_names, name="gene_name")),
    )

    # Run scanpy Seurat HVG calculation on normalized log1p data
    hvg_df = sc.pp.highly_variable_genes(
        adata_train,
        flavor="seurat",
        n_top_genes=n_top_genes,
        inplace=False,
    )

    # Extract boolean mask of highly variable genes
    mask = hvg_df["highly_variable"].to_numpy()
    hvg_indices = np.where(mask)[0]
    hvg_gene_names = [gene_names[i] for i in hvg_indices]

    logger.info(
        "Fitted Seurat HVG on %d training cells: selected %d genes from %d raw genes",
        n_cells,
        len(hvg_indices),
        n_genes,
    )
    return hvg_indices, hvg_gene_names
