"""Preprocessing pipeline configuration schema."""

from __future__ import annotations

from enum import StrEnum

from pydantic import Field

from scgraph_bench.config.base import BaseBenchConfig


class HVGFlavor(StrEnum):
    """Explicit HVG selection flavor.

    In v0, 'seurat' (computed on log1p normalised counts) is the sole explicit method.
    """

    SEURAT = "seurat"


class PreprocessingConfig(BaseBenchConfig):
    """Configuration for training-fitted scRNA-seq preprocessing pipeline."""

    target_sum: float = Field(
        default=1e4,
        description="Target library size for total-count normalisation.",
    )
    log1p: bool = Field(
        default=True,
        description="Whether to apply log(1 + x) transformation.",
    )
    hvg_flavor: HVGFlavor = Field(
        default=HVGFlavor.SEURAT,
        description="Explicit HVG selection method (fixed to 'seurat' in v0 protocol).",
    )
    n_top_genes: int = Field(
        default=2000,
        description="Number of highly variable genes to select.",
    )
    scale_data: bool = Field(
        default=True,
        description="Whether to standard-scale (zero mean, unit variance) genes using training stats.",
    )
    clip_value: float | None = Field(
        default=10.0,
        description="Maximum absolute value for scaling clip (None for no clipping).",
    )
    n_comps: int = Field(
        default=50,
        description="Number of principal components to compute on training cells.",
    )
    pca_solver: str = Field(
        default="arpack",
        description="SVD solver algorithm for PCA.",
    )
    random_state: int = Field(
        default=42,
        description="Random seed for PCA solver.",
    )
