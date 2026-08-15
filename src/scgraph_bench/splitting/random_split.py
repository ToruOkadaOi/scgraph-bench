"""Random cell-level partitioner (FOR DEBUGGING ONLY - NON-PRIMARY)."""

from __future__ import annotations

import anndata as ad
import numpy as np
import pandas as pd

from scgraph_bench.config.split import SplitConfig, SplitType
from scgraph_bench.splitting.schema import SplitDefinition
from scgraph_bench.utils.logging import get_logger

logger = get_logger("splitting.random_split")


def create_random_cell_split(
    adata: ad.AnnData,
    donor_key: str = "donor_id",
    label_key: str = "cell_type",
    split_id: str = "random_cell_debug_seed42",
    config: SplitConfig | None = None,
    seed: int = 42,
) -> SplitDefinition:
    """Create a random cell-level train/val/test split.

    WARNING: THIS METHOD IS FOR DEBUGGING AND CODE VERIFICATION ONLY.
    Random cell splitting leaks donor background variation across partitions
    and violates the primary benchmark protocol (STUDY_PROTOCOL.md).
    It must never be used as the primary benchmark evaluation setting.
    """
    logger.warning(
        "[NON-PRIMARY / DEBUG ONLY] Generating random cell-level split '%s'. "
        "This split violates donor-held-out evaluation and must NOT be used for benchmark reporting.",
        split_id,
    )

    if config is None:
        config = SplitConfig(
            split_id=split_id,
            split_type=SplitType.RANDOM_CELL_DEBUG,
            seed=seed,
        )

    rng = np.random.default_rng(seed)
    n_cells = adata.n_obs

    indices = np.arange(n_cells)
    rng.shuffle(indices)

    n_tr = int(round(n_cells * config.train_fraction))
    n_va = int(round(n_cells * config.val_fraction))

    tr_idx = indices[:n_tr]
    va_idx = indices[n_tr : n_tr + n_va]
    te_idx = indices[n_tr + n_va :]

    train_cells = adata.obs_names[tr_idx].astype(str).tolist()
    val_cells = adata.obs_names[va_idx].astype(str).tolist()
    test_cells = adata.obs_names[te_idx].astype(str).tolist()

    train_donors = sorted(adata.obs.iloc[tr_idx][donor_key].unique().tolist())
    val_donors = sorted(adata.obs.iloc[va_idx][donor_key].unique().tolist())
    test_donors = sorted(adata.obs.iloc[te_idx][donor_key].unique().tolist())

    # Label support
    adata.obs["_split_temp"] = "unassigned"
    adata.obs.iloc[tr_idx, adata.obs.columns.get_loc("_split_temp")] = "train"
    adata.obs.iloc[va_idx, adata.obs.columns.get_loc("_split_temp")] = "val"
    adata.obs.iloc[te_idx, adata.obs.columns.get_loc("_split_temp")] = "test"

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

    adata.obs.drop(columns=["_split_temp"], inplace=True)

    return SplitDefinition(
        dataset_name=adata.obs.get(
            "dataset_name", pd.Series(["stephenson_2021_healthy_pbmc"])
        ).iloc[0],
        split_id=split_id,
        split_type=SplitType.RANDOM_CELL_DEBUG,
        seed=seed,
        train_donors=train_donors,
        val_donors=val_donors,
        test_donors=test_donors,
        train_cell_ids=train_cells,
        val_cell_ids=val_cells,
        test_cell_ids=test_cells,
        site_composition={"debug_random_split": {"all_sites": len(train_donors)}},
        label_support=label_support,
        total_cells=n_cells,
        total_donors=adata.obs[donor_key].nunique(),
        config_hash=config.compute_hash(),
    )
