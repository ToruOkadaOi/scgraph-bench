"""Schemas and manifest structures for graph diagnostics."""

from __future__ import annotations

from datetime import UTC, datetime

from pydantic import BaseModel, ConfigDict, Field

from scgraph_bench.utils.hashing import hash_dict


class TopologyDiagnostics(BaseModel):
    """Structural, degree, and connectivity metrics computed strictly on graph topology."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    num_nodes: int
    num_edges: int
    density: float
    in_degree_mean: float
    in_degree_median: float
    in_degree_std: float
    in_degree_min: int
    in_degree_max: int
    out_degree_mean: float
    out_degree_median: float
    out_degree_std: float
    out_degree_min: int
    out_degree_max: int
    isolated_node_count: int
    isolated_node_fraction: float
    num_connected_components: int
    largest_component_size: int
    largest_component_fraction: float
    partition_edge_counts: dict[str, int]
    edge_weight_summary: dict[str, float] | None = None


class LabelDiagnostics(BaseModel):
    """Post hoc label-aware metrics (homophily, class purity) computed separately from graph building."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    overall_edge_homophily: float
    overall_node_homophily: float
    train_train_edge_homophily: float
    train_train_node_homophily: float
    val_to_train_query_homophily: float
    test_to_train_query_homophily: float
    expected_random_homophily: float = Field(
        default=0.0,
        description="Global class-composition baseline: sum_c (p_c ** 2) across all nodes.",
    )
    homophily_lift_over_random: float = Field(
        default=0.0,
        description="overall_edge_homophily - expected_random_homophily.",
    )
    expected_train_train_homophily: float = Field(
        default=0.0,
        description="Partition-specific baseline: sum_c p_train(c)^2.",
    )
    train_train_homophily_lift: float = Field(
        default=0.0,
        description="train_train_edge_homophily - expected_train_train_homophily.",
    )
    expected_train_to_val_homophily: float = Field(
        default=0.0,
        description="Partition-specific query baseline: sum_c p_train(c) * p_val(c).",
    )
    val_to_train_query_homophily_lift: float = Field(
        default=0.0,
        description="val_to_train_query_homophily - expected_train_to_val_homophily.",
    )
    expected_train_to_test_homophily: float = Field(
        default=0.0,
        description="Partition-specific query baseline: sum_c p_train(c) * p_test(c).",
    )
    test_to_train_query_homophily_lift: float = Field(
        default=0.0,
        description="test_to_train_query_homophily - expected_train_to_test_homophily.",
    )
    per_class_neighborhood_purity: dict[str, float]
    macro_average_class_purity: float


class MetadataDiagnostics(BaseModel):
    """Donor and sequencing site mixing diagnostics."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    train_intra_donor_edge_fraction: float
    train_intra_site_edge_fraction: float
    val_to_train_site_match_fraction: float
    test_to_train_site_match_fraction: float
    mean_train_donor_entropy: float
    mean_train_site_entropy: float
    mean_val_query_donor_entropy: float
    mean_test_query_donor_entropy: float


class GraphDiagnosticsReport(BaseModel):
    """Comprehensive graph diagnostics report linking topology, homophily, and mixing."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    graph_name: str
    dataset_name: str
    split_id: str
    graph_manifest_hash: str
    edge_index_hash: str
    feature_manifest_hash: str
    label_policy_hash: str = ""
    topology: TopologyDiagnostics
    label_diagnostics: LabelDiagnostics | None = None
    metadata_diagnostics: MetadataDiagnostics | None = None
    created_at_utc: str = Field(
        default_factory=lambda: datetime.now(UTC).isoformat(),
    )

    def compute_report_hash(self) -> str:
        """Compute SHA-256 hash of diagnostics report."""
        return hash_dict(self.model_dump(mode="json"))
