"""CPU-compatible PyTorch MLP baseline with early stopping on validation macro-F1."""

from __future__ import annotations

import copy
import platform
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader, TensorDataset

from scgraph_bench.config.model import MLPConfig
from scgraph_bench.models.base import BaseClassifier
from scgraph_bench.utils.logging import get_logger

logger = get_logger("models.mlp")


class PyTorchMLP(nn.Module):
    """Feedforward Multi-Layer Perceptron for single-cell gene expression representations."""

    def __init__(
        self,
        input_dim: int = 50,
        hidden_dims: list[int] | None = None,
        num_classes: int = 12,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        hidden_dims = hidden_dims or [128, 128]
        layers: list[nn.Module] = []

        prev_dim = input_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, h_dim))
            layers.append(nn.ReLU())
            if dropout > 0.0:
                layers.append(nn.Dropout(dropout))
            prev_dim = h_dim

        layers.append(nn.Linear(prev_dim, num_classes))
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class MLPBaseline(BaseClassifier):
    """Feature-only PyTorch MLP baseline.

    Consumes fixed PCA features only, without graph adjacency or metadata.
    Early stopping and checkpoint selection are governed strictly by validation macro-F1.
    """

    def __init__(self, config: MLPConfig | None = None) -> None:
        self.config = config or MLPConfig()
        self.model: PyTorchMLP | None = None
        self.best_epoch_: int = 0
        self.best_val_macro_f1_: float = 0.0
        self.parameter_count_: int = 0
        self.training_time_seconds_: float = 0.0
        self.training_history_: pd.DataFrame = pd.DataFrame()
        self.device_info_: dict[str, str] = {}
        self.is_fitted: bool = False

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray | None = None,
        y_val: np.ndarray | None = None,
    ) -> MLPBaseline:
        """Train MLP using AdamW and validation-based early stopping.

        Args:
            X_train: Training feature matrix (N_train x D).
            y_train: Training integer labels (N_train).
            X_val: Validation feature matrix for early stopping (N_val x D).
            y_val: Validation integer labels (N_val).

        Returns:
            Fitted baseline instance.
        """
        start_time = time.perf_counter()
        torch.manual_seed(self.config.seed)
        np.random.seed(self.config.seed)

        device = torch.device(self.config.device)
        self.device_info_ = {
            "device": str(device),
            "processor": platform.processor(),
            "machine": platform.machine(),
            "system": platform.system(),
        }

        X_tr = torch.tensor(X_train, dtype=torch.float32)
        y_tr = torch.tensor(y_train, dtype=torch.long)

        train_dataset = TensorDataset(X_tr, y_tr)
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
        )

        in_dim = X_tr.shape[1]
        n_classes = (
            self.config.num_classes
            if self.config.num_classes is not None
            else int(torch.max(y_tr).item() + 1)
        )

        self.model = PyTorchMLP(
            input_dim=in_dim,
            hidden_dims=self.config.hidden_dims,
            num_classes=n_classes,
            dropout=self.config.dropout,
        ).to(device)

        self.parameter_count_ = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        logger.info(
            "Initialized PyTorchMLP (%d trainable parameters) on %s. Max epochs=%d, patience=%d",
            self.parameter_count_,
            device,
            self.config.max_epochs,
            self.config.patience,
        )

        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )
        criterion = nn.CrossEntropyLoss()

        best_val_f1 = -1.0
        best_epoch = 0
        best_state_dict = copy.deepcopy(self.model.state_dict())
        epochs_no_improve = 0
        history_records = []

        has_val = X_val is not None and y_val is not None
        if has_val:
            assert X_val is not None and y_val is not None
            X_va = torch.tensor(X_val, dtype=torch.float32).to(device)
            y_va = np.asarray(y_val, dtype=np.int64)

        for epoch in range(1, self.config.max_epochs + 1):
            self.model.train()
            total_train_loss = 0.0

            for batch_x, batch_y in train_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                optimizer.zero_grad()
                logits = self.model(batch_x)
                loss = criterion(logits, batch_y)
                loss.backward()
                optimizer.step()
                total_train_loss += loss.item() * batch_x.size(0)

            avg_train_loss = total_train_loss / len(train_dataset)

            # Evaluate on validation partition
            if has_val:
                self.model.eval()
                with torch.no_grad():
                    val_logits = self.model(X_va)
                    val_preds = torch.argmax(val_logits, dim=-1).cpu().numpy()
                    val_f1 = float(f1_score(y_va, val_preds, average="macro", zero_division=0.0))

                history_records.append(
                    {
                        "epoch": epoch,
                        "train_loss": avg_train_loss,
                        "val_macro_f1": val_f1,
                    }
                )

                if val_f1 > best_val_f1:
                    best_val_f1 = val_f1
                    best_epoch = epoch
                    best_state_dict = copy.deepcopy(self.model.state_dict())
                    epochs_no_improve = 0
                else:
                    epochs_no_improve += 1

                if epoch % 20 == 0 or epochs_no_improve == 0:
                    logger.info(
                        "Epoch %3d/%3d: Train Loss: %.4f | Val Macro-F1: %.4f (Best: %.4f @ epoch %d)",
                        epoch,
                        self.config.max_epochs,
                        avg_train_loss,
                        val_f1,
                        best_val_f1,
                        best_epoch,
                    )

                if epochs_no_improve >= self.config.patience:
                    logger.info(
                        "Early stopping triggered at epoch %d (patience=%d without improvement).",
                        epoch,
                        self.config.patience,
                    )
                    break
            else:
                best_epoch = epoch
                best_state_dict = copy.deepcopy(self.model.state_dict())

        # Restore best checkpoint
        if has_val:
            logger.info(
                "Restoring best model checkpoint from epoch %d (Val Macro-F1: %.4f)...",
                best_epoch,
                best_val_f1,
            )
            self.model.load_state_dict(best_state_dict)
            self.best_val_macro_f1_ = best_val_f1
        self.best_epoch_ = best_epoch
        self.training_history_ = pd.DataFrame(history_records)
        self.training_time_seconds_ = time.perf_counter() - start_time
        self.is_fitted = True
        logger.info("MLP training completed in %.2f seconds.", self.training_time_seconds_)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels."""
        if not self.is_fitted or self.model is None:
            raise RuntimeError("Model is not fitted. Call fit() first.")
        self.model.eval()
        device = next(self.model.parameters()).device
        with torch.no_grad():
            x_tensor = torch.tensor(np.asarray(X, dtype=np.float32)).to(device)
            logits = self.model(x_tensor)
            preds = torch.argmax(logits, dim=-1).cpu().numpy()
        return preds

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities."""
        if not self.is_fitted or self.model is None:
            raise RuntimeError("Model is not fitted. Call fit() first.")
        self.model.eval()
        device = next(self.model.parameters()).device
        with torch.no_grad():
            x_tensor = torch.tensor(np.asarray(X, dtype=np.float32)).to(device)
            logits = self.model(x_tensor)
            probs = torch.softmax(logits, dim=-1).cpu().numpy()
        return probs
