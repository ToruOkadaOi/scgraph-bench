"""CPU-only Logistic Regression baseline with validation-based regularization tuning."""

from __future__ import annotations

import time
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score

from scgraph_bench.config.model import LogisticRegressionConfig
from scgraph_bench.models.base import BaseClassifier
from scgraph_bench.utils.logging import get_logger

logger = get_logger("models.logistic_regression")


class LogisticRegressionBaseline(BaseClassifier):
    """Multiclass Logistic Regression baseline evaluated on fixed PCA features.

    Strict Hyperparameter Tuning Protocol:
    Regularization parameter C and class_weight are tuned strictly on validation macro-F1.
    The selected configuration is refit exclusively on training cells.
    Test labels are never touched during fitting or selection.
    """

    def __init__(self, config: LogisticRegressionConfig | None = None) -> None:
        self.config = config or LogisticRegressionConfig()
        self.best_params_: dict[str, Any] = {}
        self.best_val_macro_f1_: float = 0.0
        self.selection_table_: pd.DataFrame = pd.DataFrame()
        self.model_: LogisticRegression | None = None
        self.training_time_seconds_: float = 0.0
        self.is_fitted: bool = False

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray | None = None,
        y_val: np.ndarray | None = None,
    ) -> LogisticRegressionBaseline:
        """Tune hyperparameters on validation macro-F1 and fit final model on X_train.

        Args:
            X_train: Training feature matrix (N_train x D).
            y_train: Training integer labels (N_train).
            X_val: Optional validation feature matrix (N_val x D).
            y_val: Optional validation integer labels (N_val).

        Returns:
            Fitted baseline instance.
        """
        start_time = time.perf_counter()
        X_tr = np.asarray(X_train, dtype=np.float32)
        y_tr = np.asarray(y_train, dtype=np.int64)

        grid_records = []

        if X_val is not None and y_val is not None:
            X_va = np.asarray(X_val, dtype=np.float32)
            y_va = np.asarray(y_val, dtype=np.int64)

            logger.info("Evaluating Logistic Regression hyperparameter grid on validation set...")
            for c_val in self.config.c_grid:
                for cw in self.config.class_weight_grid:
                    clf = LogisticRegression(
                        C=c_val,
                        class_weight=cw,
                        penalty=self.config.penalty,
                        solver=self.config.solver,
                        max_iter=self.config.max_iter,
                        random_state=self.config.random_state,
                    )
                    clf.fit(X_tr, y_tr)
                    val_preds = clf.predict(X_va)
                    val_macro_f1 = float(
                        f1_score(y_va, val_preds, average="macro", zero_division=0.0)
                    )
                    val_weighted_f1 = float(
                        f1_score(y_va, val_preds, average="weighted", zero_division=0.0)
                    )

                    logger.info(
                        "Candidate (C=%.4f, class_weight=%s) -> Val Macro-F1: %.4f, Weighted-F1: %.4f",
                        c_val,
                        str(cw),
                        val_macro_f1,
                        val_weighted_f1,
                    )

                    grid_records.append(
                        {
                            "C": c_val,
                            "class_weight": str(cw),
                            "val_macro_f1": val_macro_f1,
                            "val_weighted_f1": val_weighted_f1,
                        }
                    )

            self.selection_table_ = pd.DataFrame(grid_records)
            best_row = self.selection_table_.sort_values(by="val_macro_f1", ascending=False).iloc[0]
            best_C = float(best_row["C"])
            best_cw = None if best_row["class_weight"] == "None" else best_row["class_weight"]
            self.best_val_macro_f1_ = float(best_row["val_macro_f1"])
            self.best_params_ = {"C": best_C, "class_weight": best_cw}

            logger.info(
                "Selected best hyperparameters: C=%.4f, class_weight=%s (Val Macro-F1: %.4f)",
                best_C,
                str(best_cw),
                self.best_val_macro_f1_,
            )
        else:
            best_C = 1.0
            best_cw = None
            self.best_params_ = {"C": best_C, "class_weight": best_cw}

        # Refit final model strictly on training data
        logger.info(
            "Fitting final Logistic Regression model on training partition (%d cells)...", len(X_tr)
        )
        self.model_ = LogisticRegression(
            C=self.best_params_["C"],
            class_weight=self.best_params_["class_weight"],
            penalty=self.config.penalty,
            solver=self.config.solver,
            max_iter=self.config.max_iter,
            random_state=self.config.random_state,
        )
        self.model_.fit(X_tr, y_tr)

        self.training_time_seconds_ = time.perf_counter() - start_time
        self.is_fitted = True
        logger.info(
            "Logistic Regression training completed in %.2f seconds.", self.training_time_seconds_
        )
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels."""
        if not self.is_fitted or self.model_ is None:
            raise RuntimeError("Model is not fitted. Call fit() first.")
        return self.model_.predict(np.asarray(X, dtype=np.float32))

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities."""
        if not self.is_fitted or self.model_ is None:
            raise RuntimeError("Model is not fitted. Call fit() first.")
        return self.model_.predict_proba(np.asarray(X, dtype=np.float32))
