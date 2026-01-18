"""
Feature Selection Module
========================
Tools for feature correlation analysis, pruning, and importance ranking.
"""

from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from loguru import logger


class FeatureSelector:
    """
    Feature selector for removing redundant and uninformative features.

    Handles:
    - Correlation-based pruning (removes highly correlated features)
    - Feature importance ranking
    - Automatic selection of top-N features
    """

    def __init__(
        self,
        correlation_threshold: float = 0.85,
        min_variance_threshold: float = 0.01,
    ):
        """
        Initialize feature selector.

        Args:
            correlation_threshold: Remove features with |correlation| > threshold
            min_variance_threshold: Remove features with variance < threshold
        """
        self.correlation_threshold = correlation_threshold
        self.min_variance_threshold = min_variance_threshold

        self.correlation_matrix_: Optional[pd.DataFrame] = None
        self.removed_features_: Dict[str, str] = {}
        self.selected_features_: List[str] = []
        self.feature_importance_: Optional[Dict[str, float]] = None

    def compute_correlation_matrix(
        self,
        df: pd.DataFrame,
        features: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """
        Compute correlation matrix for features.

        Args:
            df: DataFrame with features
            features: List of feature columns (uses all numeric if None)

        Returns:
            Correlation matrix as DataFrame
        """
        if features is None:
            features = df.select_dtypes(include=[np.number]).columns.tolist()
            # Exclude price columns
            features = [f for f in features if f not in ["open", "high", "low", "close", "tick_volume"]]

        self.correlation_matrix_ = df[features].corr()
        return self.correlation_matrix_

    def find_correlated_pairs(
        self,
        threshold: Optional[float] = None,
    ) -> List[Tuple[str, str, float]]:
        """
        Find all pairs of features with correlation above threshold.

        Args:
            threshold: Correlation threshold (uses default if None)

        Returns:
            List of (feature1, feature2, correlation) tuples
        """
        if self.correlation_matrix_ is None:
            raise ValueError("Call compute_correlation_matrix first")

        threshold = threshold or self.correlation_threshold
        corr = self.correlation_matrix_

        pairs = []
        for i, col1 in enumerate(corr.columns):
            for col2 in corr.columns[i + 1:]:
                corr_value = corr.loc[col1, col2]
                if abs(corr_value) > threshold:
                    pairs.append((col1, col2, corr_value))

        # Sort by absolute correlation descending
        pairs.sort(key=lambda x: abs(x[2]), reverse=True)

        logger.info(f"Found {len(pairs)} highly correlated pairs (|r| > {threshold})")
        return pairs

    def prune_correlated_features(
        self,
        df: pd.DataFrame,
        features: Optional[List[str]] = None,
        keep_highest_variance: bool = True,
    ) -> List[str]:
        """
        Remove highly correlated features.

        When two features are correlated above threshold, keeps the one
        with higher variance (more information) by default.

        Args:
            df: DataFrame with features
            features: List of feature columns to consider
            keep_highest_variance: If True, keep feature with higher variance

        Returns:
            List of selected (non-redundant) feature names
        """
        if features is None:
            features = df.select_dtypes(include=[np.number]).columns.tolist()
            features = [f for f in features if f not in ["open", "high", "low", "close", "tick_volume"]]

        # Compute correlation matrix
        self.compute_correlation_matrix(df, features)

        # Track features to remove
        features_to_remove = set()
        self.removed_features_ = {}

        # Get variance of each feature
        variances = df[features].var()

        # Iterate through correlation matrix
        corr = self.correlation_matrix_
        for i, col1 in enumerate(corr.columns):
            if col1 in features_to_remove:
                continue

            for col2 in corr.columns[i + 1:]:
                if col2 in features_to_remove:
                    continue

                corr_value = abs(corr.loc[col1, col2])
                if corr_value > self.correlation_threshold:
                    # Decide which to remove
                    if keep_highest_variance:
                        if variances[col1] >= variances[col2]:
                            to_remove = col2
                            to_keep = col1
                        else:
                            to_remove = col1
                            to_keep = col2
                    else:
                        # Keep first alphabetically
                        if col1 < col2:
                            to_remove = col2
                            to_keep = col1
                        else:
                            to_remove = col1
                            to_keep = col2

                    features_to_remove.add(to_remove)
                    self.removed_features_[to_remove] = (
                        f"Correlated with {to_keep} (r={corr.loc[col1, col2]:.3f})"
                    )

        # Selected features
        self.selected_features_ = [f for f in features if f not in features_to_remove]

        logger.info(
            f"Feature pruning: {len(features)} -> {len(self.selected_features_)} "
            f"(removed {len(features_to_remove)})"
        )

        # Log removed features
        for feature, reason in self.removed_features_.items():
            logger.debug(f"  Removed '{feature}': {reason}")

        return self.selected_features_

    def filter_low_variance(
        self,
        df: pd.DataFrame,
        features: Optional[List[str]] = None,
    ) -> List[str]:
        """
        Remove features with variance below threshold.

        Args:
            df: DataFrame with features
            features: List of feature columns

        Returns:
            List of features with sufficient variance
        """
        if features is None:
            features = self.selected_features_ or df.select_dtypes(include=[np.number]).columns.tolist()

        variances = df[features].var()
        low_variance = variances[variances < self.min_variance_threshold].index.tolist()

        for feature in low_variance:
            self.removed_features_[feature] = f"Low variance ({variances[feature]:.6f})"

        selected = [f for f in features if f not in low_variance]

        if low_variance:
            logger.info(f"Removed {len(low_variance)} low-variance features")

        self.selected_features_ = selected
        return selected

    def select_features(
        self,
        df: pd.DataFrame,
        features: Optional[List[str]] = None,
    ) -> List[str]:
        """
        Run full feature selection pipeline.

        1. Remove low variance features
        2. Remove highly correlated features

        Args:
            df: DataFrame with features
            features: Initial feature list

        Returns:
            List of selected features
        """
        if features is None:
            features = df.select_dtypes(include=[np.number]).columns.tolist()
            features = [f for f in features if f not in ["open", "high", "low", "close", "tick_volume"]]

        logger.info(f"Starting feature selection with {len(features)} features")

        # Step 1: Filter low variance
        features = self.filter_low_variance(df, features)

        # Step 2: Prune correlated features
        features = self.prune_correlated_features(df, features)

        logger.info(f"Final selected features: {len(features)}")
        return features

    def get_removal_report(self) -> pd.DataFrame:
        """
        Get report of removed features and reasons.

        Returns:
            DataFrame with 'feature' and 'reason' columns
        """
        if not self.removed_features_:
            return pd.DataFrame(columns=["feature", "reason"])

        return pd.DataFrame([
            {"feature": f, "reason": r}
            for f, r in self.removed_features_.items()
        ])


class FeatureImportanceAnalyzer:
    """
    Analyze feature importance from trained models.

    Supports:
    - XGBoost native importance (gain, weight, cover)
    - Permutation importance for any model
    - SHAP values (if shap is installed)
    """

    def __init__(self):
        self.importance_scores_: Dict[str, Dict[str, float]] = {}

    def from_xgboost(
        self,
        model,
        feature_names: List[str],
        importance_type: str = "gain",
    ) -> Dict[str, float]:
        """
        Extract feature importance from XGBoost model.

        Args:
            model: Trained XGBoost model
            feature_names: List of feature names
            importance_type: 'gain', 'weight', 'cover', 'total_gain', 'total_cover'

        Returns:
            Dict mapping feature name to importance score
        """
        try:
            # Get importance dict from model
            importance = model.get_booster().get_score(importance_type=importance_type)

            # Map to feature names (XGBoost uses f0, f1, etc.)
            result = {}
            for feat_id, score in importance.items():
                if feat_id.startswith("f"):
                    idx = int(feat_id[1:])
                    if idx < len(feature_names):
                        result[feature_names[idx]] = score
                else:
                    result[feat_id] = score

            # Normalize scores
            total = sum(result.values())
            if total > 0:
                result = {k: v / total for k, v in result.items()}

            # Sort by importance
            result = dict(sorted(result.items(), key=lambda x: x[1], reverse=True))

            self.importance_scores_["xgboost_" + importance_type] = result
            logger.info(f"Extracted XGBoost {importance_type} importance for {len(result)} features")

            return result

        except Exception as e:
            logger.error(f"Failed to extract XGBoost importance: {e}")
            return {}

    def permutation_importance(
        self,
        model,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: List[str],
        n_repeats: int = 10,
        metric_fn=None,
    ) -> Dict[str, float]:
        """
        Calculate permutation importance for any model.

        Permutation importance measures how much the model's performance
        decreases when a feature's values are randomly shuffled.

        Args:
            model: Trained model with predict() method
            X: Feature array
            y: Target array
            feature_names: List of feature names
            n_repeats: Number of times to shuffle each feature
            metric_fn: Metric function (default: MSE for regression)

        Returns:
            Dict mapping feature name to importance score
        """
        from sklearn.metrics import mean_squared_error

        if metric_fn is None:
            metric_fn = lambda y_true, y_pred: -mean_squared_error(y_true, y_pred)

        # Baseline score
        y_pred = model.predict(X)
        baseline = metric_fn(y, y_pred)

        importances = {}

        for i, name in enumerate(feature_names):
            scores = []
            for _ in range(n_repeats):
                # Copy and shuffle feature
                X_shuffled = X.copy()
                np.random.shuffle(X_shuffled[:, :, i] if X.ndim == 3 else X_shuffled[:, i])

                # Score with shuffled feature
                y_pred = model.predict(X_shuffled)
                score = metric_fn(y, y_pred)
                scores.append(baseline - score)

            importances[name] = np.mean(scores)

        # Normalize
        total = sum(abs(v) for v in importances.values())
        if total > 0:
            importances = {k: abs(v) / total for k, v in importances.items()}

        # Sort by importance
        importances = dict(sorted(importances.items(), key=lambda x: x[1], reverse=True))

        self.importance_scores_["permutation"] = importances
        logger.info(f"Calculated permutation importance for {len(importances)} features")

        return importances

    def get_top_features(
        self,
        n: int = 20,
        method: Optional[str] = None,
    ) -> List[str]:
        """
        Get top N most important features.

        Args:
            n: Number of features to return
            method: Importance method to use (uses most recent if None)

        Returns:
            List of top feature names
        """
        if not self.importance_scores_:
            raise ValueError("No importance scores calculated. Call an importance method first.")

        if method is None:
            method = list(self.importance_scores_.keys())[-1]

        scores = self.importance_scores_[method]
        return list(scores.keys())[:n]

    def get_importance_report(
        self,
        method: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Get importance report as DataFrame.

        Args:
            method: Importance method (uses most recent if None)

        Returns:
            DataFrame with 'feature', 'importance', 'rank' columns
        """
        if not self.importance_scores_:
            return pd.DataFrame(columns=["feature", "importance", "rank"])

        if method is None:
            method = list(self.importance_scores_.keys())[-1]

        scores = self.importance_scores_[method]
        return pd.DataFrame([
            {"feature": f, "importance": s, "rank": i + 1}
            for i, (f, s) in enumerate(scores.items())
        ])

    def compare_methods(self) -> pd.DataFrame:
        """
        Compare importance rankings across different methods.

        Returns:
            DataFrame with feature rankings from each method
        """
        if not self.importance_scores_:
            return pd.DataFrame()

        # Get all features
        all_features = set()
        for scores in self.importance_scores_.values():
            all_features.update(scores.keys())

        # Build comparison table
        data = []
        for feature in all_features:
            row = {"feature": feature}
            for method, scores in self.importance_scores_.items():
                if feature in scores:
                    # Find rank
                    rank = list(scores.keys()).index(feature) + 1
                    row[f"{method}_rank"] = rank
                    row[f"{method}_score"] = scores[feature]
                else:
                    row[f"{method}_rank"] = None
                    row[f"{method}_score"] = 0
            data.append(row)

        df = pd.DataFrame(data)

        # Add average rank
        rank_cols = [c for c in df.columns if c.endswith("_rank")]
        df["avg_rank"] = df[rank_cols].mean(axis=1)
        df = df.sort_values("avg_rank")

        return df
