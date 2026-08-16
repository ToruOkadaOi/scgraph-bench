"""PyTorch Geometric GCN Classifier for strict-inductive cell-type annotation."""

from __future__ import annotations

import copy
import time
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pydantic import BaseModel, ConfigDict, Field
from sklearn.metrics import f1_score
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv

from scgraph_bench.utils.hashing import hash_dict
from scgraph_bench.utils.logging import get_logger
from scgraph_bench.utils.seed import set_seed

logger = get_logger("models.gcn")


class GCNConfig(BaseModel):
    """Configuration for 2-layer Graph Convolutional Network."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    in_features: int = 50
    hidden_dim: int = 128
    num_classes: int = 12
    dropout: float = 0.2
    lr: float = 0.001
    weight_decay: float = 1e-4
    max_epochs: int = 500
    patience: int = 50
    seed: int = 42
    device: str = Field(
        default="auto",
        description="Target device: 'cuda', 'mps', 'cpu', or 'auto' (detects fastest available).",
    )

    def compute_hash(self) -> str:
        """Compute cryptographic hash of model hyperparameters."""
        return hash_dict(self.model_dump(mode="json"))


class GCNNet(nn.Module):
    """2-layer Graph Convolutional Network architecture."""

    def __init__(
        self,
        in_features: int,
        hidden_dim: int,
        num_classes: int,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.conv1 = GCNConv(in_features, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.conv2 = GCNConv(hidden_dim, num_classes)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """Forward pass applying graph convolutions, batch norm, and dropout."""
        h = self.conv1(x, edge_index)
        h = self.bn1(h)
        h = F.relu(h)
        h = self.dropout(h)
        out = self.conv2(h, edge_index)
        return out


class GCNClassifier:
    """Strict-inductive GCN Classifier managing training, early stopping, and evaluation."""

    def __init__(self, config: GCNConfig | None = None) -> None:
        self.config = config or GCNConfig()
        self.model: GCNNet | None = None
        self.device_: torch.device = self._resolve_device(self.config.device)
        self.device_info_: dict[str, Any] = self._get_device_telemetry()
        self.best_epoch_: int | None = None
        self.best_val_macro_f1_: float | None = None
        self.training_time_seconds_: float = 0.0
        self.parameter_count_: int | None = None
        self.peak_memory_mb_: float = 0.0

    def _resolve_device(self, target: str) -> torch.device:
        if target == "auto":
            if torch.cuda.is_available():
                return torch.device("cuda")
            if torch.backends.mps.is_available():
                return torch.device("mps")
            return torch.device("cpu")
        return torch.device(target)

    def _get_device_telemetry(self) -> dict[str, Any]:
        info: dict[str, Any] = {
            "device_type": self.device_.type,
            "torch_version": torch.__version__,
        }
        if self.device_.type == "cuda":
            info["device_name"] = torch.cuda.get_device_name(0)
            info["cuda_version"] = torch.version.cuda
            info["device_count"] = torch.cuda.device_count()
        elif self.device_.type == "mps":
            info["device_name"] = "Apple Silicon GPU (MPS)"
        else:
            info["device_name"] = "CPU"
        return info

    def fit(
        self,
        pyg_data: Data,
        val_labels: np.ndarray,
    ) -> GCNClassifier:
        """Fit GCN on PyG Data with early stopping on validation partition macro-F1.

        Args:
            pyg_data: PyTorch Geometric Data object containing x, edge_index, train_mask, val_mask.
            val_labels: Ground-truth validation label array of length N_val for early stopping.

        Returns:
            self.
        """
        set_seed(self.config.seed)
        start_time = time.perf_counter()

        self.model = GCNNet(
            in_features=self.config.in_features,
            hidden_dim=self.config.hidden_dim,
            num_classes=self.config.num_classes,
            dropout=self.config.dropout,
        ).to(self.device_)

        self.parameter_count_ = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config.lr,
            weight_decay=self.config.weight_decay,
        )
        criterion = nn.CrossEntropyLoss()

        data = pyg_data.to(self.device_)
        train_mask = data.train_mask
        val_mask = data.val_mask
        y_train = data.y[train_mask]

        best_val_f1 = -1.0
        best_state: dict[str, Any] | None = None
        best_epoch = -1
        patience_counter = 0

        # Memory tracking
        if self.device_.type == "cuda":
            torch.cuda.reset_peak_memory_stats()

        logger.info(
            "Starting GCN training on device: %s (%d params, seed: %d)",
            self.device_,
            self.parameter_count_,
            self.config.seed,
        )

        for epoch in range(1, self.config.max_epochs + 1):
            self.model.train()
            optimizer.zero_grad()

            logits = self.model(data.x, data.edge_index)
            loss = criterion(logits[train_mask], y_train)
            loss.backward()
            optimizer.step()

            # Validation evaluation
            self.model.eval()
            with torch.no_grad():
                val_logits = logits[val_mask].cpu().numpy()
                val_preds = np.argmax(val_logits, axis=1)
                val_f1 = float(f1_score(val_labels, val_preds, average="macro", zero_division=0.0))

            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                best_state = copy.deepcopy(self.model.state_dict())
                best_epoch = epoch
                patience_counter = 0
            else:
                patience_counter += 1

            if epoch % 25 == 0 or epoch == 1 or patience_counter == 0:
                logger.debug(
                    "Epoch %03d | Train Loss: %.4f | Val Macro-F1: %.4f (Best: %.4f @ epoch %d)",
                    epoch,
                    loss.item(),
                    val_f1,
                    best_val_f1,
                    best_epoch,
                )

            if patience_counter >= self.config.patience:
                logger.info(
                    "Early stopping triggered at epoch %d (Best Epoch: %d)", epoch, best_epoch
                )
                break

        if best_state is not None:
            self.model.load_state_dict(best_state)

        self.best_epoch_ = best_epoch
        self.best_val_macro_f1_ = best_val_f1
        self.training_time_seconds_ = time.perf_counter() - start_time

        if self.device_.type == "cuda":
            self.peak_memory_mb_ = float(torch.cuda.max_memory_allocated() / (1024 * 1024))
        else:
            self.peak_memory_mb_ = 0.0

        logger.info(
            "GCN training completed in %.2fs. Best Val Macro-F1: %.4f at epoch %d",
            self.training_time_seconds_,
            self.best_val_macro_f1_,
            self.best_epoch_,
        )
        return self

    def predict_all(self, pyg_data: Data) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Generate class predictions for train, validation, and test partitions simultaneously.

        Args:
            pyg_data: PyG Data container.

        Returns:
            Tuple of (y_pred_train, y_pred_val, y_pred_test).
        """
        if self.model is None:
            raise RuntimeError("Model has not been fitted.")

        self.model.eval()
        data = pyg_data.to(self.device_)
        with torch.no_grad():
            logits = self.model(data.x, data.edge_index)
            preds = torch.argmax(logits, dim=1).cpu().numpy()

        train_preds = preds[data.train_mask.cpu().numpy()]
        val_preds = preds[data.val_mask.cpu().numpy()]
        test_preds = preds[data.test_mask.cpu().numpy()]

        return train_preds, val_preds, test_preds

    def predict_proba_all(self, pyg_data: Data) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Generate class probabilities for train, validation, and test partitions.

        Args:
            pyg_data: PyG Data container.

        Returns:
            Tuple of (proba_train, proba_val, proba_test).
        """
        if self.model is None:
            raise RuntimeError("Model has not been fitted.")

        self.model.eval()
        data = pyg_data.to(self.device_)
        with torch.no_grad():
            logits = self.model(data.x, data.edge_index)
            probs = F.softmax(logits, dim=1).cpu().numpy()

        train_probs = probs[data.train_mask.cpu().numpy()]
        val_probs = probs[data.val_mask.cpu().numpy()]
        test_probs = probs[data.test_mask.cpu().numpy()]

        return train_probs, val_probs, test_probs
