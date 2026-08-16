"""Schemas, manifests, and PyG containers for benchmark graph bundles."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
from pydantic import BaseModel, ConfigDict, Field
from torch_geometric.data import Data

from scgraph_bench.utils.hashing import hash_dict


class GraphManifest(BaseModel):
    """Cryptographic and topological manifest for a serialized graph bundle."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    graph_name: str
    builder_type: str
    dataset_name: str
    split_id: str
    k: int
    nominal_query_k: int | None = None
    nominal_train_k: int | None = None
    reference_edge_policy: str = Field(
        default="standard_knn",
        description="Algorithm policy used for reference train->train edges.",
    )
    query_edge_policy: str = Field(
        default="standard_knn",
        description="Algorithm policy used for query train->val and train->test edges.",
    )
    metric: str
    weighting: str
    sigma_k: float | None = None
    edge_index_convention: str = Field(
        default="source_to_target",
        description="Edge index orientation: row 0 is source node, row 1 is target node.",
    )
    message_flow_train: str = Field(default="train_to_train")
    message_flow_validation: str = Field(default="train_to_validation")
    message_flow_test: str = Field(default="train_to_test")
    query_nodes_affect_training_representations: bool = Field(default=False)
    num_nodes: int
    num_edges: int
    num_train_nodes: int
    num_val_nodes: int
    num_test_nodes: int
    num_train_train_edges: int
    num_train_to_val_edges: int
    num_train_to_test_edges: int
    num_disallowed_edges: int = Field(
        default=0,
        description="Number of forbidden edges (val-val, test-test, test-val, val->train, test->train). Must be 0.",
    )
    edge_index_hash: str
    edge_weight_hash: str | None = None
    feature_manifest_hash: str
    created_at_utc: str = Field(
        default_factory=lambda: datetime.now(UTC).isoformat(),
    )

    def compute_manifest_hash(self) -> str:
        """Compute top-level hash of graph manifest."""
        return hash_dict(self.model_dump(mode="json"))


class GraphBundle:
    """Immutable graph container holding PyG connectivity tensors, partition masks, and manifests.

    Enforces strict label isolation: GraphBundle does not store full cell-type labels.
    """

    def __init__(
        self,
        edge_index: torch.Tensor,
        num_nodes: int,
        train_mask: torch.Tensor,
        val_mask: torch.Tensor,
        test_mask: torch.Tensor,
        node_cell_ids: list[str],
        manifest: GraphManifest,
        edge_weight: torch.Tensor | None = None,
    ) -> None:
        self.edge_index = edge_index.to(dtype=torch.long)
        self.num_nodes = int(num_nodes)
        self.train_mask = train_mask.to(dtype=torch.bool)
        self.val_mask = val_mask.to(dtype=torch.bool)
        self.test_mask = test_mask.to(dtype=torch.bool)
        self.node_cell_ids = list(node_cell_ids)
        self.manifest = manifest
        self.edge_weight = edge_weight.to(dtype=torch.float32) if edge_weight is not None else None

        self._validate()

    def _validate(self) -> None:
        if self.edge_index.dim() != 2 or self.edge_index.size(0) != 2:
            raise ValueError(f"edge_index must have shape [2, E], got {self.edge_index.shape}")
        if len(self.node_cell_ids) != self.num_nodes:
            raise ValueError(
                f"node_cell_ids count ({len(self.node_cell_ids)}) does not match num_nodes ({self.num_nodes})"
            )
        if (
            self.train_mask.size(0) != self.num_nodes
            or self.val_mask.size(0) != self.num_nodes
            or self.test_mask.size(0) != self.num_nodes
        ):
            raise ValueError("Mask sizes do not match num_nodes")
        if self.edge_weight is not None and self.edge_weight.size(0) != self.edge_index.size(1):
            raise ValueError(
                f"edge_weight size ({self.edge_weight.size(0)}) != num edges ({self.edge_index.size(1)})"
            )

    def to_pyg_data(
        self,
        x: torch.Tensor | np.ndarray,
        y_train_only: torch.Tensor | np.ndarray | None = None,
    ) -> Data:
        """Convert GraphBundle into standard PyG Data object.

        Strict Label Isolation:
        Validation and test labels are never stored in the PyG graph container.
        If y_train_only is provided, it is populated exclusively for training nodes.

        Args:
            x: Node feature matrix (N x D).
            y_train_only: Optional training-only labels (length N_train or full array masked).

        Returns:
            torch_geometric.data.Data instance.
        """
        x_tensor = torch.as_tensor(x, dtype=torch.float32)

        data = Data(
            x=x_tensor,
            edge_index=self.edge_index,
            edge_attr=self.edge_weight,
            train_mask=self.train_mask,
            val_mask=self.val_mask,
            test_mask=self.test_mask,
            num_nodes=self.num_nodes,
        )

        if y_train_only is not None:
            y_arr = np.asarray(y_train_only)
            if len(y_arr) == self.manifest.num_train_nodes:
                y_full = torch.full((self.num_nodes,), fill_value=-1, dtype=torch.long)
                y_full[self.train_mask] = torch.as_tensor(y_arr, dtype=torch.long)
                data.y = y_full
            else:
                data.y = torch.as_tensor(y_train_only, dtype=torch.long)

        return data

    def save(self, output_dir: Path | str) -> None:
        """Serialize graph tensors, masks, and manifest to disk."""
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)

        payload: dict[str, Any] = {
            "edge_index": self.edge_index.cpu().numpy(),
            "train_mask": self.train_mask.cpu().numpy(),
            "val_mask": self.val_mask.cpu().numpy(),
            "test_mask": self.test_mask.cpu().numpy(),
            "num_nodes": self.num_nodes,
            "node_cell_ids": np.array(self.node_cell_ids, dtype=object),
        }
        if self.edge_weight is not None:
            payload["edge_weight"] = self.edge_weight.cpu().numpy()

        np.savez_compressed(out / "graph_tensors.npz", **payload)
        (out / "graph_manifest.json").write_text(
            self.manifest.model_dump_json(indent=2), encoding="utf-8"
        )

    @classmethod
    def load(cls, output_dir: Path | str) -> GraphBundle:
        """Load serialized GraphBundle from disk."""
        out = Path(output_dir)
        tensors = np.load(out / "graph_tensors.npz", allow_pickle=True)
        manifest_dict = json.loads((out / "graph_manifest.json").read_text(encoding="utf-8"))
        manifest = GraphManifest.model_validate(manifest_dict)

        edge_index = torch.from_numpy(tensors["edge_index"]).to(torch.long)
        train_mask = torch.from_numpy(tensors["train_mask"]).to(torch.bool)
        val_mask = torch.from_numpy(tensors["val_mask"]).to(torch.bool)
        test_mask = torch.from_numpy(tensors["test_mask"]).to(torch.bool)
        num_nodes = int(tensors["num_nodes"])
        node_cell_ids = tensors["node_cell_ids"].tolist()

        edge_weight = None
        if "edge_weight" in tensors:
            edge_weight = torch.from_numpy(tensors["edge_weight"]).to(torch.float32)

        return cls(
            edge_index=edge_index,
            num_nodes=num_nodes,
            train_mask=train_mask,
            val_mask=val_mask,
            test_mask=test_mask,
            node_cell_ids=node_cell_ids,
            manifest=manifest,
            edge_weight=edge_weight,
        )
