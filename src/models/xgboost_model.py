"""
XGBoost Model Module
====================
Gradient boosting classifier for gold price direction prediction.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from loguru import logger
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import TimeSeriesSplit

try:
    import xgboost as xgb
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False
    logger.warning("XGBoost not available")


class GoldXGBoost:
    """
    XGBoost classifier for predicting gold price direction.

    Predicts:
    - 0: Down (price will decrease)
    - 1: Neutral (price will stay within threshold)
    - 2: Up (price will increase)
    """

    DEFAULT_PARAMS = {
        "objective": "multi:softprob",
        "num_class": 3,
        "max_depth": 6,
        "learning_rate": 0.1,
        "n_estimators": 200,
        "min_child_weight": 1,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "gamma": 0,
        "reg_alpha": 0.1,
        "reg_lambda": 1,
        "random_state": 42,
        "n_jobs": -1,
        "tree_method": "hist",  # Faster training
    }

    def __init__(
        self,
        params: Optional[Dict] = None,
        feature_names: Optional[List[str]] = None,
    ):
        """
        Initialize XGBoost model.

        Args:
            params: XGBoost parameters (merged with defaults)
            feature_names: List of feature names for importance tracking
        """
        if not XGB_AVAILABLE:
            raise ImportError("XGBoost is required but not installed")

        self.params = {**self.DEFAULT_PARAMS, **(params or {})}
        self.feature_names = feature_names
        self.model = None
        self._fitted = False

    def prepare_features(self, X: np.ndarray) -> np.ndarray:
        """
        Flatten sequences for XGBoost.

        Converts (samples, timesteps, features) to (samples, timesteps * features)
        by taking statistical features from the sequence.

        Args:
            X: Input array of shape (samples, timesteps, features)

        Returns:
            Flattened features of shape (samples, new_features)
        """
        if len(X.shape) == 2:
            return X  # Already 2D

        samples, timesteps, features = X.shape

        # Extract statistical features from sequences
        flat_features = []

        # Last timestep values
        flat_features.append(X[:, -1, :])

        # Mean of sequence
        flat_features.append(np.mean(X, axis=1))

        # Standard deviation
        flat_features.append(np.std(X, axis=1))

        # Min/Max
        flat_features.append(np.min(X, axis=1))
        flat_features.append(np.max(X, axis=1))

        # Change from first to last
        flat_features.append(X[:, -1, :] - X[:, 0, :])

        # Rolling means (recent vs older)
        recent_mean = np.mean(X[:, -10:, :], axis=1)
        older_mean = np.mean(X[:, :-10, :], axis=1)
        flat_features.append(recent_mean - older_mean)

        return np.hstack(flat_features)

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        early_stopping_rounds: int = 10,
    ) -> Dict:
        """
        Train the XGBoost model.

        Args:
            X: Training features
            y: Training labels (0, 1, 2 for down, neutral, up)
            X_val: Validation features
            y_val: Validation labels
            early_stopping_rounds: Early stopping patience

        Returns:
            Training results dict
        """
        # Prepare features
        X_flat = self.prepare_features(X)
        logger.info(f"Training XGBoost with {X_flat.shape[1]} features")

        # Create model
        self.model = xgb.XGBClassifier(**self.params)

        # Prepare validation data
        eval_set = None
        if X_val is not None and y_val is not None:
            X_val_flat = self.prepare_features(X_val)
            eval_set = [(X_val_flat, y_val)]

        # Train
        self.model.fit(
            X_flat,
            y,
            eval_set=eval_set,
            verbose=True,
        )

        self._fitted = True

        # Evaluate
        y_pred = self.model.predict(X_flat)
        train_accuracy = accuracy_score(y, y_pred)
        train_f1 = f1_score(y, y_pred, average="weighted")

        results = {
            "train_accuracy": train_accuracy,
            "train_f1": train_f1,
        }

        if X_val is not None and y_val is not None:
            y_val_pred = self.model.predict(X_val_flat)
            results["val_accuracy"] = accuracy_score(y_val, y_val_pred)
            results["val_f1"] = f1_score(y_val, y_val_pred, average="weighted")
            logger.info(f"Validation Accuracy: {results['val_accuracy']:.4f}")

        logger.info(f"Training Accuracy: {train_accuracy:.4f}")

        return results

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class labels.

        Args:
            X: Input features

        Returns:
            Predicted labels (0, 1, 2)
        """
        if not self._fitted:
            raise ValueError("Model not fitted. Call fit() first.")

        X_flat = self.prepare_features(X)
        return self.model.predict(X_flat)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class probabilities.

        Args:
            X: Input features

        Returns:
            Probability array of shape (samples, 3)
        """
        if not self._fitted:
            raise ValueError("Model not fitted. Call fit() first.")

        X_flat = self.prepare_features(X)
        return self.model.predict_proba(X_flat)

    def get_feature_importance(self, importance_type: str = "gain") -> Dict[str, float]:
        """
        Get feature importance scores.

        Args:
            importance_type: Type of importance ('gain', 'weight', 'cover')

        Returns:
            Dict mapping feature names to importance scores
        """
        if not self._fitted:
            raise ValueError("Model not fitted.")

        importance = self.model.get_booster().get_score(importance_type=importance_type)

        # Map to feature names if available
        if self.feature_names:
            # Note: Feature names need to be extended for flattened features
            pass

        return importance

    def cross_validate(
        self,
        X: np.ndarray,
        y: np.ndarray,
        n_splits: int = 5,
    ) -> Dict:
        """
        Perform time series cross-validation.

        Args:
            X: Features
            y: Labels
            n_splits: Number of CV splits

        Returns:
            Cross-validation results
        """
        X_flat = self.prepare_features(X)
        tscv = TimeSeriesSplit(n_splits=n_splits)

        scores = []
        for fold, (train_idx, val_idx) in enumerate(tscv.split(X_flat)):
            X_train, X_val = X_flat[train_idx], X_flat[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]

            model = xgb.XGBClassifier(**self.params)
            model.fit(X_train, y_train, verbose=False)

            y_pred = model.predict(X_val)
            score = accuracy_score(y_val, y_pred)
            scores.append(score)

            logger.info(f"Fold {fold + 1}: Accuracy = {score:.4f}")

        return {
            "scores": scores,
            "mean": np.mean(scores),
            "std": np.std(scores),
        }

    def save(self, path: str) -> None:
        """Save model to file."""
        if not self._fitted:
            raise ValueError("Model not fitted. Nothing to save.")

        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.model.save_model(path)
        logger.info(f"Model saved to {path}")

    @classmethod
    def load(cls, path: str) -> "GoldXGBoost":
        """
        Load model from file.

        Args:
            path: Path to model file

        Returns:
            Loaded GoldXGBoost instance
        """
        instance = cls()
        instance.model = xgb.XGBClassifier()
        instance.model.load_model(path)
        instance._fitted = True
        logger.info(f"Model loaded from {path}")
        return instance


def calculate_direction_labels(
    close_prices: np.ndarray,
    horizon: int = 1,
    threshold: float = 0.05,
) -> np.ndarray:
    """
    Calculate direction labels from close prices.

    Args:
        close_prices: Array of close prices
        horizon: Prediction horizon (bars ahead)
        threshold: Percentage threshold for neutral zone

    Returns:
        Array of labels (0=down, 1=neutral, 2=up)
    """
    future_prices = np.roll(close_prices, -horizon)
    returns = (future_prices - close_prices) / close_prices * 100

    labels = np.ones(len(returns), dtype=int)  # Default neutral
    labels[returns > threshold] = 2  # Up
    labels[returns < -threshold] = 0  # Down

    # Remove last horizon values (no future data)
    return labels[:-horizon]
