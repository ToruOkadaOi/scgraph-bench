"""Embedding quality metrics for evaluating learned representation geometry.

Compares GNN hidden-layer embeddings against raw input PCA features and MLP
penultimate representations to isolate whether message passing improves class
separability beyond what the fixed features already provide.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field
from sklearn.metrics import silhouette_score
from sklearn.neighbors import KNeighborsClassifier

from scgraph_bench.utils.logging import get_logger

logger = get_logger("analysis.embedding_quality")


class EmbeddingQualityReport(BaseModel):
    """Geometry and separability metrics for one representation of one partition."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    representation_name: str
    partition: str = "test"
    n_samples: int = 0
    n_dims: int = 0
    silhouette_euclidean: float | None = Field(
        default=None,
        description="Silhouette coefficient with Euclidean distance (-1 to 1); None if <3 classes.",
    )
    knn_accuracy: float | None = Field(
        default=None,
        description="Leave-partition-out kNN classification accuracy in this space.",
    )
    centroid_separation: float | None = Field(
        default=None,
        description="Mean between-class centroid distance / mean within-class scatter radius.",
    )
    mean_class_radius: float | None = None


def _centroid_separation(emb: np.ndarray, y: np.ndarray) -> tuple[float, float] | tuple[None, None]:
    classes = np.unique(y)
    if len(classes) < 2:
        return None, None
    centroids = np.stack([emb[y == c].mean(axis=0) for c in classes])
    radii = [
        float(np.linalg.norm(emb[y == c] - centroids[i], axis=1).mean())
        for i, c in enumerate(classes)
    ]
    mean_radius = float(np.mean(radii))
    dists = []
    for i in range(len(classes)):
        for j in range(i + 1, len(classes)):
            dists.append(float(np.linalg.norm(centroids[i] - centroids[j])))
    mean_between = float(np.mean(dists))
    if mean_radius <= 0:
        return None, mean_radius
    return mean_between / mean_radius, mean_radius


def compute_embedding_quality(
    emb: np.ndarray,
    y: np.ndarray,
    representation_name: str,
    partition: str = "test",
    knn_neighbors: int = 15,
    reference_emb: np.ndarray | None = None,
    reference_y_train: np.ndarray | None = None,
    y_train: np.ndarray | None = None,
) -> EmbeddingQualityReport:
    """Compute geometry metrics; optionally kNN accuracy trained on a train split.

    Args:
        emb: Representation matrix for the evaluated partition (N x D).
        y: Integer labels aligned with rows of emb.
        representation_name: Label for the representation being scored.
        partition: Partition name.
        knn_neighbors: Neighbors used by the kNN separability probe.
        reference_emb: Optional training-split embeddings for fitting the kNN probe.
        reference_y_train: Labels aligned with reference_emb.
        y_train: Ignored placeholder for API symmetry; must be None or equal-length alias.
    """
    emb = np.asarray(emb, dtype=np.float64)
    y = np.asarray(y, dtype=np.int64)
    if len(emb) != len(y):
        raise ValueError(f"emb rows ({len(emb)}) != labels ({len(y)})")

    classes = np.unique(y)
    silhouette: float | None = None
    if 2 <= len(classes) < len(y):
        silhouette = float(silhouette_score(emb, y, metric="euclidean"))

    knn_acc: float | None = None
    if reference_emb is not None and reference_y_train is not None:
        ref = np.asarray(reference_emb, dtype=np.float64)
        ref_y = np.asarray(reference_y_train, dtype=np.int64)
        k = min(knn_neighbors, len(ref))
        probe = KNeighborsClassifier(n_neighbors=k)
        probe.fit(ref, ref_y)
        knn_acc = float(probe.score(emb, y))
    elif y_train is None and len(classes) >= 2:
        k = min(knn_neighbors, len(y) // 2)
        probe = KNeighborsClassifier(n_neighbors=k, weights="distance")
        rng = np.random.default_rng(42)
        idx = rng.permutation(len(y))
        half = len(y) // 2
        probe.fit(emb[idx[:half]], y[idx[:half]])
        knn_acc = float(probe.score(emb[idx[half:]], y[idx[half:]]))

    separation, radius = _centroid_separation(emb, y)

    report = EmbeddingQualityReport(
        representation_name=representation_name,
        partition=partition,
        n_samples=int(len(emb)),
        n_dims=int(emb.shape[1]),
        silhouette_euclidean=silhouette,
        knn_accuracy=knn_acc,
        centroid_separation=separation,
        mean_class_radius=radius,
    )
    logger.debug(
        "Embedding quality (%s/%s): silhouette=%s knn=%s sep=%s",
        representation_name,
        partition,
        silhouette,
        knn_acc,
        separation,
    )
    return report


def compare_representations(
    reports: list[EmbeddingQualityReport],
) -> pd.DataFrame:
    """Tabulate multiple representation-quality reports side by side."""
    rows = [r.model_dump() for r in reports]
    df = pd.DataFrame(rows)
    order = [
        "representation_name",
        "partition",
        "silhouette_euclidean",
        "knn_accuracy",
        "centroid_separation",
        "mean_class_radius",
        "n_samples",
        "n_dims",
    ]
    cols = [c for c in order if c in df.columns]
    return df[cols]
