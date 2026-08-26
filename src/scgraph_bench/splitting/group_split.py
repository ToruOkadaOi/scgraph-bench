"""Site-stratified and group-held-out split partitioners."""

from __future__ import annotations

import anndata as ad
import numpy as np
import pandas as pd

from scgraph_bench.config.split import RareClassAction, SplitConfig, SplitType
from scgraph_bench.splitting.schema import SplitDefinition
from scgraph_bench.utils.logging import get_logger

logger = get_logger("splitting.group_split")


def create_site_stratified_donor_split(
    adata: ad.AnnData,
    donor_key: str = "donor_id",
    site_key: str = "site",
    label_key: str = "cell_type",
    split_id: str = "site_stratified_seed42",
    dataset_name: str | None = None,
    config: SplitConfig | None = None,
    seed: int = 42,
) -> SplitDefinition:
    """Create site-stratified donor-held-out train/val/test split.

    Guarantees that donors from each sequencing center/site (e.g. Cambridge and Newcastle)
    are evenly stratified into train, validation, and test partitions, while ensuring
    zero donor overlap between partitions.

    Default target donor allocation for 23 donors (12 Cambridge + 11 Newcastle):
    - Train: 6 Cambridge + 6 Newcastle (12 donors)
    - Validation: 3 Cambridge + 3 Newcastle (6 donors)
    - Test: 3 Cambridge + 2 Newcastle (5 donors)

    Args:
        adata: Standardized AnnData object.
        donor_key: Column name for donor IDs.
        site_key: Column name for site/batch IDs.
        label_key: Column name for cell-type labels.
        split_id: Unique identifier for this frozen split.
        config: Optional SplitConfig with rare-class constraints.
        seed: Random seed for deterministic donor assignment.

    Returns:
        SplitDefinition with disjoint donor and cell ID sets.
    """
    if config is None:
        config = SplitConfig(split_id=split_id, seed=seed)

    rng = np.random.default_rng(seed)

    # Get unique donor-to-site mapping and per-donor cell counts
    donor_site_series = adata.obs.groupby(donor_key, observed=True)[site_key].first()
    donor_cell_counts = adata.obs[donor_key].value_counts()
    donor_df = pd.DataFrame({site_key: donor_site_series, "cell_count": donor_cell_counts})

    sites = sorted(donor_df[site_key].unique().tolist())
    train_donors: list[str] = []
    val_donors: list[str] = []
    test_donors: list[str] = []
    site_composition: dict[str, dict[str, int]] = {"train": {}, "val": {}, "test": {}}

    for site in sites:
        site_donors = donor_df[donor_df[site_key] == site].index.tolist()
        # Sort site donors by cell count for balanced partition weights
        site_donors_sorted = sorted(site_donors, key=lambda d: donor_cell_counts[d], reverse=True)

        n_site = len(site_donors_sorted)
        if n_site == 12:  # Cambridge
            n_tr, n_va, n_te = 6, 3, 3
        elif n_site == 11:  # Newcastle
            n_tr, n_va, n_te = 6, 3, 2
        else:
            n_tr = max(1, int(round(n_site * config.train_fraction)))
            n_va = max(1, int(round(n_site * config.val_fraction)))
            n_te = n_site - n_tr - n_va
            if n_te <= 0:
                n_te = 1
                n_tr = n_site - n_va - n_te

        # Permute within count-ranked pairs for balanced stochastic selection
        indices = np.arange(n_site)
        rng.shuffle(indices)
        permuted = [site_donors_sorted[i] for i in indices]

        s_tr = permuted[:n_tr]
        s_va = permuted[n_tr : n_tr + n_va]
        s_te = permuted[n_tr + n_va :]

        train_donors.extend(s_tr)
        val_donors.extend(s_va)
        test_donors.extend(s_te)

        site_composition["train"][site] = len(s_tr)
        site_composition["val"][site] = len(s_va)
        site_composition["test"][site] = len(s_te)

    # Sort donor lists
    train_donors = sorted(train_donors)
    val_donors = sorted(val_donors)
    test_donors = sorted(test_donors)

    # Extract cell IDs
    train_mask = adata.obs[donor_key].isin(train_donors)
    val_mask = adata.obs[donor_key].isin(val_donors)
    test_mask = adata.obs[donor_key].isin(test_donors)

    train_cells = adata.obs_names[train_mask].astype(str).tolist()
    val_cells = adata.obs_names[val_mask].astype(str).tolist()
    test_cells = adata.obs_names[test_mask].astype(str).tolist()

    # Calculate label support per partition
    adata.obs["_split_temp"] = "unassigned"
    adata.obs.loc[train_mask, "_split_temp"] = "train"
    adata.obs.loc[val_mask, "_split_temp"] = "val"
    adata.obs.loc[test_mask, "_split_temp"] = "test"

    support_df = pd.crosstab(adata.obs[label_key], adata.obs["_split_temp"])
    for col in ["train", "val", "test"]:
        if col not in support_df.columns:
            support_df[col] = 0

    label_support: dict[str, dict[str, int]] = {}
    for lbl in support_df.index:
        label_support[str(lbl)] = {
            "train": int(support_df.loc[lbl, "train"]),
            "val": int(support_df.loc[lbl, "val"]),
            "test": int(support_df.loc[lbl, "test"]),
            "total": int(support_df.loc[lbl].sum()),
        }

    # Clean up temp column
    adata.obs.drop(columns=["_split_temp"], inplace=True)

    # Rare class check
    rc_cfg = config.rare_class_config
    for lbl, counts in label_support.items():
        if counts["train"] < rc_cfg.min_train_cells_per_class:
            msg = (
                f"Class '{lbl}' has {counts['train']} train cells (below minimum threshold "
                f"{rc_cfg.min_train_cells_per_class})."
            )
            if rc_cfg.action == RareClassAction.FAIL:
                raise ValueError(msg)
            logger.warning(msg)

    split_def = SplitDefinition(
        dataset_name=dataset_name
        or (
            adata.obs["dataset_name"].iloc[0]
            if "dataset_name" in adata.obs.columns
            else "stephenson_2021_healthy_pbmc"
        ),
        split_id=split_id,
        split_type=SplitType.DONOR_HELD_OUT,
        seed=seed,
        train_donors=train_donors,
        val_donors=val_donors,
        test_donors=test_donors,
        train_cell_ids=train_cells,
        val_cell_ids=val_cells,
        test_cell_ids=test_cells,
        site_composition=site_composition,
        label_support=label_support,
        total_cells=adata.n_obs,
        total_donors=len(train_donors) + len(val_donors) + len(test_donors),
        config_hash=config.compute_hash(),
    )
    split_def.validate_disjointness()
    return split_def
