"""Graph construction configuration schemas."""

from __future__ import annotations

from enum import StrEnum

from pydantic import Field

from scgraph_bench.config.base import BaseBenchConfig


class GraphBuilderType(StrEnum):
    """Supported graph builder algorithms."""

    PCA_KNN = "pca_knn"
    MUTUAL_KNN = "mutual_knn"
    BBKNN = "bbknn"
    REWIRED_CONTROL = "rewired_control"


class EdgeWeightingMode(StrEnum):
    """Edge weighting schemes."""

    UNWEIGHTED = "unweighted"
    RBF_WEIGHTED = "rbf_weighted"


class InductiveMode(StrEnum):
    """Inductive graph connectivity semantics.

    In v0, 'strict_bipartite' enforces that test/val nodes connect only to
    training-reference nodes; test-to-test and val-to-val edges are disabled.
    """

    STRICT_BIPARTITE = "strict_bipartite"


class PCAkNNConfig(BaseBenchConfig):
    """Configuration for standard PCA k-nearest neighbor graphs."""

    k: int = Field(default=20, ge=2, le=100, description="Number of nearest neighbors.")
    metric: str = Field(default="euclidean", description="Distance metric in PCA space.")
    symmetrize: bool = Field(default=True, description="Whether to symmetrize adjacency matrix.")
    weighting: EdgeWeightingMode = Field(
        default=EdgeWeightingMode.UNWEIGHTED,
        description="Edge weighting method.",
    )


class MutualkNNConfig(BaseBenchConfig):
    """Configuration for Mutual PCA k-nearest neighbor graphs."""

    k: int = Field(default=20, ge=2, le=100, description="Number of candidate nearest neighbors.")
    metric: str = Field(default="euclidean", description="Distance metric in PCA space.")
    weighting: EdgeWeightingMode = Field(
        default=EdgeWeightingMode.UNWEIGHTED,
        description="Edge weighting method.",
    )


class BBKNNConfig(BaseBenchConfig):
    """Configuration for Batch-Balanced k-Nearest Neighbors (BBKNN).

    Requires explicit strict-inductive inference across training batches.
    """

    k_per_batch: int = Field(
        default=3,
        ge=1,
        description="Number of neighbors selected per donor/batch.",
    )
    approx_total_k: int = Field(
        default=20,
        description="Target total degree matching standard PCA-kNN.",
    )
    batch_key: str = Field(
        default="donor_id",
        description="Metadata key identifying batch/donor partitions.",
    )
    metric: str = Field(default="euclidean", description="Distance metric in PCA space.")


class RewiredControlConfig(BaseBenchConfig):
    """Configuration for degree-preserving randomized rewiring negative control.

    Topology is swapped preserving node degrees; realized homophily is recorded post hoc.
    """

    reference_graph_name: str = Field(
        default="pca_knn",
        description="Base graph to rewire.",
    )
    n_swaps_factor: float = Field(
        default=10.0,
        description="Multiplier of total edge count for number of edge swaps.",
    )
    seed: int = Field(default=42, description="Random seed for rewiring permutation.")


class GraphConfig(BaseBenchConfig):
    """Composite graph construction configuration."""

    builder_type: GraphBuilderType = GraphBuilderType.PCA_KNN
    inductive_mode: InductiveMode = InductiveMode.STRICT_BIPARTITE
    pca_knn: PCAkNNConfig | None = PCAkNNConfig()
    mutual_knn: MutualkNNConfig | None = None
    bbknn: BBKNNConfig | None = None
    rewired_control: RewiredControlConfig | None = None
