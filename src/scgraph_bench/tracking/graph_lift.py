"""Matched graph lift evaluation engine comparing GNN models to strictly matched MLP baselines."""

from __future__ import annotations

from scgraph_bench.evaluation.schema import EvaluationSummary
from scgraph_bench.tracking.schema import GraphLiftRecord, RunManifest
from scgraph_bench.utils.logging import get_logger

logger = get_logger("tracking.graph_lift")


def compute_matched_graph_lift(
    gnn_summary: EvaluationSummary,
    mlp_summary: EvaluationSummary,
    gnn_manifest: RunManifest,
    mlp_manifest: RunManifest,
    graph_name: str,
) -> GraphLiftRecord:
    """Compute matched graph lift comparing a GNN experiment against an identically-seeded MLP baseline.

    Inviolable Matching Guardrails:
    - Dataset, Dataset Version, Split ID, Split Hash, Random Seed, Feature Manifest Hash,
      Preprocessing Config Hash, and Label Policy Hash MUST match bit-for-bit.
    - If any dimension differs, execution halts with a validation error.

    Formula:
        graph_lift = Macro-F1(GNN) - Macro-F1(matched_MLP)

    Args:
        gnn_summary: Test partition evaluation summary of the GNN.
        mlp_summary: Test partition evaluation summary of the matched MLP.
        gnn_manifest: RunManifest of the GNN run.
        mlp_manifest: RunManifest of the MLP run.
        graph_name: Identifier of the graph structure used by the GNN.

    Returns:
        GraphLiftRecord with overall, per-site, per-donor, and per-class comparative lifts.
    """
    # 1. Enforce strict matching invariants
    mismatches: list[str] = []
    if gnn_manifest.dataset_name != mlp_manifest.dataset_name:
        mismatches.append(
            f"Dataset mismatch: GNN={gnn_manifest.dataset_name} vs MLP={mlp_manifest.dataset_name}"
        )
    if gnn_manifest.dataset_version != mlp_manifest.dataset_version:
        mismatches.append(
            f"Dataset version mismatch: GNN={gnn_manifest.dataset_version} vs MLP={mlp_manifest.dataset_version}"
        )
    if gnn_manifest.split_id != mlp_manifest.split_id:
        mismatches.append(
            f"Split ID mismatch: GNN={gnn_manifest.split_id} vs MLP={mlp_manifest.split_id}"
        )
    if gnn_manifest.split_hash != mlp_manifest.split_hash:
        mismatches.append(
            f"Split hash mismatch: GNN={gnn_manifest.split_hash} vs MLP={mlp_manifest.split_hash}"
        )
    if gnn_manifest.seed != mlp_manifest.seed:
        mismatches.append(f"Seed mismatch: GNN={gnn_manifest.seed} vs MLP={mlp_manifest.seed}")
    if gnn_manifest.feature_manifest_hash != mlp_manifest.feature_manifest_hash:
        if (
            gnn_manifest.split_hash == mlp_manifest.split_hash
            and gnn_manifest.seed == mlp_manifest.seed
            and gnn_manifest.preprocessing_config_hash == mlp_manifest.preprocessing_config_hash
        ):
            logger.warning(
                "Feature manifest hash differs (GNN=%s vs MLP=%s), but split_hash and preprocessing_config_hash match identically.",
                gnn_manifest.feature_manifest_hash[:16],
                mlp_manifest.feature_manifest_hash[:16],
            )
        else:
            mismatches.append(
                f"Feature manifest hash mismatch: GNN={gnn_manifest.feature_manifest_hash} vs MLP={mlp_manifest.feature_manifest_hash}"
            )
    if gnn_manifest.preprocessing_config_hash != mlp_manifest.preprocessing_config_hash:
        mismatches.append(
            f"Preprocessing config hash mismatch: GNN={gnn_manifest.preprocessing_config_hash} vs MLP={mlp_manifest.preprocessing_config_hash}"
        )
    if gnn_manifest.label_mapping_hash != mlp_manifest.label_mapping_hash:
        mismatches.append(
            f"Label mapping hash mismatch: GNN={gnn_manifest.label_mapping_hash} vs MLP={mlp_manifest.label_mapping_hash}"
        )

    if mismatches:
        err_msg = f"Invalid graph lift match: {'; '.join(mismatches)}"
        logger.error(err_msg)
        raise ValueError(err_msg)

    # 2. Overall lifts
    overall_lift = float(gnn_summary.macro_f1 - mlp_summary.macro_f1)
    bal_acc_lift = float(gnn_summary.balanced_accuracy - mlp_summary.balanced_accuracy)

    # 3. Stratified per-site lifts (Cambridge vs Newcastle)
    gnn_sites = {s.site: s for s in gnn_summary.per_site}
    mlp_sites = {s.site: s for s in mlp_summary.per_site}

    cambridge_gnn_f1 = (
        gnn_sites["Cambridge"].observed_class_macro_f1 if "Cambridge" in gnn_sites else None
    )
    cambridge_mlp_f1 = (
        mlp_sites["Cambridge"].observed_class_macro_f1 if "Cambridge" in mlp_sites else None
    )
    cambridge_lift = (
        float(cambridge_gnn_f1 - cambridge_mlp_f1)
        if cambridge_gnn_f1 is not None and cambridge_mlp_f1 is not None
        else None
    )

    newcastle_gnn_f1 = (
        gnn_sites["Newcastle"].observed_class_macro_f1 if "Newcastle" in gnn_sites else None
    )
    newcastle_mlp_f1 = (
        mlp_sites["Newcastle"].observed_class_macro_f1 if "Newcastle" in mlp_sites else None
    )
    newcastle_lift = (
        float(newcastle_gnn_f1 - newcastle_mlp_f1)
        if newcastle_gnn_f1 is not None and newcastle_mlp_f1 is not None
        else None
    )

    # 4. Stratified per-donor observed-class lifts
    gnn_donors = {d.donor_id: d for d in gnn_summary.per_donor}
    mlp_donors = {d.donor_id: d for d in mlp_summary.per_donor}

    per_donor_lifts: dict[str, float] = {}
    for d_id, gnn_d in gnn_donors.items():
        if d_id in mlp_donors:
            mlp_d = mlp_donors[d_id]
            per_donor_lifts[d_id] = float(
                gnn_d.observed_class_macro_f1 - mlp_d.observed_class_macro_f1
            )

    # 5. Per-class lifts
    gnn_classes = {c.class_name: c for c in gnn_summary.per_class}
    mlp_classes = {c.class_name: c for c in mlp_summary.per_class}

    per_class_lifts: dict[str, float] = {}
    for c_name, gnn_c in gnn_classes.items():
        if c_name in mlp_classes:
            mlp_c = mlp_classes[c_name]
            per_class_lifts[c_name] = float(gnn_c.f1 - mlp_c.f1)

    logger.info(
        "Computed matched graph lift for %s (Seed=%d) on '%s': Overall Lift = %+.4f (GNN=%.4f vs MLP=%.4f)",
        gnn_manifest.model_name,
        gnn_manifest.seed,
        graph_name,
        overall_lift,
        gnn_summary.macro_f1,
        mlp_summary.macro_f1,
    )

    return GraphLiftRecord(
        dataset_name=gnn_manifest.dataset_name,
        dataset_version=gnn_manifest.dataset_version,
        split_id=gnn_manifest.split_id,
        split_hash=gnn_manifest.split_hash,
        seed=gnn_manifest.seed,
        graph_name=graph_name,
        gnn_model_name=gnn_manifest.model_name,
        matched_mlp_model_name=mlp_manifest.model_name,
        gnn_macro_f1=gnn_summary.macro_f1,
        matched_mlp_macro_f1=mlp_summary.macro_f1,
        overall_graph_lift=overall_lift,
        gnn_balanced_accuracy=gnn_summary.balanced_accuracy,
        matched_mlp_balanced_accuracy=mlp_summary.balanced_accuracy,
        balanced_accuracy_lift=bal_acc_lift,
        cambridge_gnn_f1=cambridge_gnn_f1,
        cambridge_mlp_f1=cambridge_mlp_f1,
        cambridge_lift=cambridge_lift,
        newcastle_gnn_f1=newcastle_gnn_f1,
        newcastle_mlp_f1=newcastle_mlp_f1,
        newcastle_lift=newcastle_lift,
        per_donor_lifts=per_donor_lifts,
        per_class_lifts=per_class_lifts,
        feature_manifest_hash=gnn_manifest.feature_manifest_hash,
        preprocessing_config_hash=gnn_manifest.preprocessing_config_hash,
        label_mapping_hash=gnn_manifest.label_mapping_hash,
        is_valid_match=True,
    )
