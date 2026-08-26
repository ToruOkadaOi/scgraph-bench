"""Dataset loader implementations for scgraph-bench."""

from __future__ import annotations

from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
from rich.console import Console

from scgraph_bench.config.dataset import DatasetConfig, MetadataConstraintConfig
from scgraph_bench.data.base import BaseDatasetLoader
from scgraph_bench.data.registry import register_dataset
from scgraph_bench.data.validation import validate_anndata_schema
from scgraph_bench.utils.logging import get_logger

logger = get_logger("data.loaders")
console = Console()

# 12 Initial Primary v0 Flat Classes
PRIMARY_V0_LABELS_STEPHENSON = [
    "naive thymus-derived CD4-positive, alpha-beta T cell",
    "central memory CD4-positive, alpha-beta T cell",
    "naive thymus-derived CD8-positive, alpha-beta T cell",
    "effector CD8-positive, alpha-beta T cell",
    "effector memory CD8-positive, alpha-beta T cell",
    "CD14-positive monocyte",
    "CD16-positive, CD56-dim natural killer cell, human",
    "CD16-negative, CD56-bright natural killer cell, human",
    "naive B cell",
    "gamma-delta T cell",
    "mucosal invariant T cell",
    "platelet",
]


@register_dataset("stephenson_2021_healthy_pbmc")
class StephensonHealthyPBMCLoader(BaseDatasetLoader):
    """Production loader for Stephenson et al. (2021) Healthy PBMC dataset."""

    def __init__(
        self,
        cache_dir: Path | str | None = None,
        manifest_path: Path | str | None = None,
    ) -> None:
        super().__init__(cache_dir)
        self.manifest_path = (
            Path(manifest_path)
            if manifest_path
            else Path(__file__).parents[3] / "configs" / "dataset" / "stephenson_donor_manifest.csv"
        )

    def load(
        self,
        config: DatasetConfig | None = None,
        dev_subsample_per_donor: int | None = None,
        seed: int = 42,
        primary_only: bool = True,
    ) -> ad.AnnData:
        """Load and standardize Stephenson healthy PBMC dataset."""
        if config is None:
            config = DatasetConfig(
                name="stephenson_2021_healthy_pbmc",
                description="Stephenson et al. (2021) Nature Medicine Healthy PBMC Cohort",
                cell_id_key="cell_id",
                label_key="cell_type",
                donor_key="donor_id",
                batch_key="site",
            )

        cache_file = self.cache_dir / "stephenson_2021_healthy_pbmc.h5ad"

        if cache_file.is_file():
            logger.info("Loading Stephenson healthy PBMC from local cache %s", cache_file)
            adata = ad.read_h5ad(cache_file)
        else:
            logger.info("Local cache not found. Extracting via cellxgene_census...")
            try:
                import cellxgene_census
            except ImportError as err:
                raise RuntimeError(
                    f"cellxgene_census is required to extract the Stephenson dataset from Census to {cache_file}. "
                    "Please install it via `pip install cellxgene-census` or place the pre-cached .h5ad in cache_dir."
                ) from err

            with cellxgene_census.open_soma(census_version="2025-11-08") as census:
                adata = cellxgene_census.get_anndata(
                    census,
                    "homo_sapiens",
                    measurement_name="RNA",
                    X_name="raw",
                    obs_value_filter="dataset_id == 'c7775e88-49bf-4ba2-a03b-93f00447c958' and disease == 'normal'",
                )
            logger.info("Caching raw AnnData to %s", cache_file)
            adata.write_h5ad(cache_file)

        # 1. Apply versioned donor manifest to filter to unperturbed healthy donors
        if not self.manifest_path.is_file():
            raise FileNotFoundError(f"Stephenson donor manifest not found: {self.manifest_path}")

        manifest_df = pd.read_csv(self.manifest_path)
        included_donors = manifest_df[manifest_df["inclusion_status"] == "included"][
            "donor_id"
        ].tolist()
        donor_to_site = dict(zip(manifest_df["donor_id"], manifest_df["site"], strict=False))

        mask_included = adata.obs["donor_id"].isin(included_donors)
        adata = adata[mask_included].copy()

        # 2. Standardize obs and var metadata fields
        adata.obs["site"] = adata.obs["donor_id"].map(donor_to_site)
        adata.obs["cell_id"] = adata.obs_names.astype(str)
        if "feature_id" in adata.var.columns:
            adata.var_names = adata.var["feature_id"].astype(str)

        # 3. Label policy: Mark and filter 12 primary v0 classes
        adata.obs["is_primary_v0"] = adata.obs["cell_type"].isin(PRIMARY_V0_LABELS_STEPHENSON)

        if primary_only:
            adata = adata[adata.obs["is_primary_v0"]].copy()

        # 4. Optional deterministic development subsampling per donor (for CPU smoke tests)
        if dev_subsample_per_donor is not None and dev_subsample_per_donor > 0:
            logger.info(
                "Applying development subsample cap: max %d cells per donor (seed=%d)",
                dev_subsample_per_donor,
                seed,
            )
            rng = np.random.default_rng(seed)
            sampled_indices: list[int] = []
            for donor in adata.obs["donor_id"].unique():
                donor_cell_idx = np.where(adata.obs["donor_id"] == donor)[0]
                if len(donor_cell_idx) > dev_subsample_per_donor:
                    chosen = rng.choice(donor_cell_idx, size=dev_subsample_per_donor, replace=False)
                    sampled_indices.extend(chosen)
                else:
                    sampled_indices.extend(donor_cell_idx)
            sampled_indices.sort()
            adata = adata[sampled_indices].copy()

        # 5. Validate schema
        if dev_subsample_per_donor is not None and dev_subsample_per_donor > 0:
            config = config.model_copy(
                update={
                    "constraints": config.constraints.model_copy(
                        update={
                            "min_cells_per_donor": min(
                                config.constraints.min_cells_per_donor, dev_subsample_per_donor
                            )
                        }
                    )
                }
            )
        validate_anndata_schema(adata, config)
        logger.info(
            "Stephenson healthy PBMC loaded successfully: shape=%s, donors=%d, labels=%d",
            adata.shape,
            adata.obs["donor_id"].nunique(),
            adata.obs["cell_type"].nunique(),
        )
        return adata


@register_dataset("gse164690_hnscc")
class GSE164690HNSCCLoader(BaseDatasetLoader):
    """Production loader for GSE164690 Head and Neck Squamous Cell Carcinoma (HNSCC) cohort."""

    PRIMARY_V0_LABELS = [
        "B cell",
        "CD4-positive, alpha-beta T cell",
        "CD8-positive, alpha-beta T cell",
        "classical monocyte",
        "dendritic cell",
        "endothelial cell",
        "fibroblast",
        "macrophage",
        "malignant epithelial cell",
        "mast cell",
        "natural killer cell",
        "non-classical monocyte",
        "plasma cell",
        "regulatory T cell",
    ]

    def __init__(
        self,
        cache_dir: Path | str | None = None,
        manifest_path: Path | str | None = None,
    ) -> None:
        super().__init__(cache_dir)
        self.manifest_path = (
            Path(manifest_path)
            if manifest_path
            else Path(__file__).parents[3] / "configs" / "dataset" / "gse164690_donor_manifest.csv"
        )

    def load(
        self,
        config: DatasetConfig | None = None,
        dev_subsample_per_donor: int | None = None,
        seed: int = 42,
        primary_only: bool = True,
    ) -> ad.AnnData:
        """Load and standardize GSE164690 HNSCC dataset."""
        if config is None:
            config = DatasetConfig(
                name="gse164690_hnscc",
                description="Kürten et al. (2021) Nature Communications HNSCC Cohort",
                cell_id_key="cell_id",
                label_key="cell_type",
                donor_key="donor_id",
                batch_key="compartment",
            )

        cache_file = self.cache_dir / "gse164690_annotated.h5ad"

        if cache_file.is_file():
            logger.info("Loading GSE164690 from local cache %s", cache_file)
            adata = ad.read_h5ad(cache_file)
        else:
            logger.info("Local cache not found. Running audit loader...")
            from scripts.audit_gse164690_hnscc import run_audit
            project_root = Path(__file__).parents[3]
            run_audit(self.cache_dir.parent, project_root / "audits" / "gse164690_hnscc")
            adata = ad.read_h5ad(cache_file)

        # 1. Apply versioned donor manifest to filter to included donors
        if self.manifest_path.is_file():
            manifest_df = pd.read_csv(self.manifest_path)
            included_donors = manifest_df[manifest_df["inclusion_status"] == "included"][
                "donor_id"
            ].tolist()
            mask_included = adata.obs["donor_id"].isin(included_donors)
            adata = adata[mask_included].copy()

        # 2. Standardize obs and var metadata fields
        adata.obs["cell_id"] = adata.obs_names.astype(str)
        if "feature_id" in adata.var.columns:
            adata.var_names = adata.var["feature_id"].astype(str)

        # 3. Label policy: Mark and filter primary classes
        adata.obs["is_primary_v0"] = adata.obs["cell_type"].isin(self.PRIMARY_V0_LABELS)

        if primary_only:
            adata = adata[adata.obs["is_primary_v0"]].copy()

        # 4. Optional deterministic development subsampling per donor
        if dev_subsample_per_donor is not None and dev_subsample_per_donor > 0:
            logger.info(
                "Applying development subsample cap: max %d cells per donor (seed=%d)",
                dev_subsample_per_donor,
                seed,
            )
            rng = np.random.default_rng(seed)
            sampled_indices: list[int] = []
            for donor in adata.obs["donor_id"].unique():
                donor_cell_idx = np.where(adata.obs["donor_id"] == donor)[0]
                if len(donor_cell_idx) > dev_subsample_per_donor:
                    chosen = rng.choice(donor_cell_idx, size=dev_subsample_per_donor, replace=False)
                    sampled_indices.extend(chosen)
                else:
                    sampled_indices.extend(donor_cell_idx)
            sampled_indices.sort()
            adata = adata[sampled_indices].copy()

        # 5. Validate schema
        if dev_subsample_per_donor is not None and dev_subsample_per_donor > 0:
            config = config.model_copy(
                update={
                    "constraints": config.constraints.model_copy(
                        update={
                            "min_cells_per_donor": min(
                                config.constraints.min_cells_per_donor, dev_subsample_per_donor
                            )
                        }
                    )
                }
            )
        validate_anndata_schema(adata, config)
        logger.info(
            "GSE164690 HNSCC loaded successfully: shape=%s, donors=%d, labels=%d",
            adata.shape,
            adata.obs["donor_id"].nunique(),
            adata.obs["cell_type"].nunique(),
        )
        return adata


@register_dataset("synthetic_fixture")
class SyntheticFixtureLoader(BaseDatasetLoader):
    """Synthetic multi-donor scRNA-seq loader for unit and integration testing."""

    def load(
        self,
        config: DatasetConfig | None = None,
        _dev_subsample_per_donor: int | None = None,
        seed: int = 42,
    ) -> ad.AnnData:
        from tests.fixtures.synthetic_adata import generate_synthetic_scrna_adata

        if config is None:
            config = DatasetConfig(
                name="synthetic_fixture",
                cell_id_key="cell_id",
                label_key="cell_type",
                donor_key="donor_id",
                batch_key="condition",
                constraints=MetadataConstraintConfig(min_cells_per_donor=10),
            )

        adata = generate_synthetic_scrna_adata(
            n_cells=600,
            n_genes=100,
            n_donors=6,
            n_classes=4,
            seed=seed,
        )
        validate_anndata_schema(adata, config)
        return adata
