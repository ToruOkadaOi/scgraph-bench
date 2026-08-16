"""Abstract base class for baseline classifiers."""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class BaseClassifier(ABC):
    """Abstract base class for cell-type classification baselines.

    Inviolable Isolation Guards:
    - Baseline models receive feature arrays (X) and training labels (y_train) only.
    - Baseline models must never receive GraphBundle or edge_index objects.
    - Test labels must never be accessed during fitting or hyperparameter selection.
    """

    @abstractmethod
    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray | None = None,
        y_val: np.ndarray | None = None,
    ) -> BaseClassifier:
        """Fit model strictly on training data with optional validation-based tuning."""
        raise NotImplementedError

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict discrete class indices for input feature matrix."""
        raise NotImplementedError

    @abstractmethod
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probability distribution for input feature matrix."""
        raise NotImplementedError
