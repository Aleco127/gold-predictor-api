"""
Data Preprocessing Module
=========================
Handles data cleaning, normalization, and sequence creation for ML models.

IMPORTANT: To prevent data leakage, always fit scalers on training data only.
Use `prepare_train_val_test()` for proper pipeline or call `fit()` explicitly
on training data before calling `transform()` on other splits.
"""

from pathlib import Path
from typing import Dict, Generator, List, Optional, Tuple, Union
import warnings

import joblib
import numpy as np
import pandas as pd
from loguru import logger
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import MinMaxScaler, StandardScaler


class DataProcessor:
    """
    Data processor for preparing OHLCV data for ML models.

    Handles:
    - Missing value handling
    - Feature scaling (MinMax or Standard)
    - Sequence creation for LSTM
    - Train/validation/test splits
    - Scaler persistence
    """

    FEATURE_COLUMNS = [
        # Trend
        "ema_9", "ema_21", "ema_50", "sma_20", "sma_50",
        "ema_cross", "ema_distance",
        # Momentum
        "rsi", "macd", "macd_histogram", "macd_signal",
        "stoch_k", "stoch_d", "roc", "willr",
        # Volatility
        "bb_lower", "bb_mid", "bb_upper", "bb_bandwidth", "bb_percent",
        "atr", "atr_percent", "volatility",
        # Price features
        "return_1", "return_5", "return_10",
        "body_percent", "range",
        "distance_from_high_20", "distance_from_low_20",
    ]

    def __init__(
        self,
        lookback: int = 60,
        horizon: int = 1,
        scaler_type: str = "minmax",
        feature_columns: Optional[List[str]] = None,
    ):
        """
        Initialize data processor.

        Args:
            lookback: Number of timesteps to look back for sequences
            horizon: Number of timesteps to predict ahead
            scaler_type: Type of scaler ('minmax' or 'standard')
            feature_columns: List of feature columns to use
        """
        self.lookback = lookback
        self.horizon = horizon
        self.scaler_type = scaler_type
        self.feature_columns = feature_columns or self.FEATURE_COLUMNS

        # Initialize scalers
        if scaler_type == "minmax":
            self.feature_scaler = MinMaxScaler(feature_range=(0, 1))
            self.target_scaler = MinMaxScaler(feature_range=(0, 1))
        else:
            self.feature_scaler = StandardScaler()
            self.target_scaler = StandardScaler()

        self._fitted = False

    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean and preprocess raw data.

        Args:
            df: Raw DataFrame with indicators

        Returns:
            Cleaned DataFrame
        """
        df = df.copy()

        # Get available feature columns
        available_features = [col for col in self.feature_columns if col in df.columns]

        if len(available_features) < len(self.feature_columns):
            missing = set(self.feature_columns) - set(available_features)
            logger.warning(f"Missing features: {missing}")

        # Select price and features
        columns_to_keep = ["open", "high", "low", "close"] + available_features
        columns_to_keep = [c for c in columns_to_keep if c in df.columns]
        df = df[columns_to_keep]

        # Handle infinite values
        df = df.replace([np.inf, -np.inf], np.nan)

        # Forward fill then backward fill NaN values
        df = df.ffill().bfill()

        # Drop any remaining NaN rows
        initial_len = len(df)
        df = df.dropna()
        dropped = initial_len - len(df)

        if dropped > 0:
            logger.info(f"Dropped {dropped} rows with NaN values")

        self.feature_columns = available_features
        logger.info(f"Cleaned data: {len(df)} rows, {len(available_features)} features")

        return df

    def fit(self, df: pd.DataFrame) -> "DataProcessor":
        """
        Fit scalers on training data.

        Args:
            df: Training DataFrame

        Returns:
            Self for chaining
        """
        # Fit feature scaler
        feature_data = df[self.feature_columns].values
        self.feature_scaler.fit(feature_data)

        # Fit target scaler on close price
        target_data = df["close"].values.reshape(-1, 1)
        self.target_scaler.fit(target_data)

        self._fitted = True
        logger.info("Scalers fitted on training data")

        return self

    def transform(
        self,
        df: pd.DataFrame,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Transform data and create sequences.

        Args:
            df: DataFrame to transform

        Returns:
            Tuple of (X sequences, y targets)
        """
        if not self._fitted:
            raise ValueError("DataProcessor not fitted. Call fit() first.")

        # Scale features
        feature_data = df[self.feature_columns].values
        scaled_features = self.feature_scaler.transform(feature_data)

        # Scale target (close price)
        target_data = df["close"].values.reshape(-1, 1)
        scaled_target = self.target_scaler.transform(target_data).flatten()

        # Create sequences
        X, y = create_sequences(
            scaled_features,
            scaled_target,
            lookback=self.lookback,
            horizon=self.horizon,
        )

        logger.info(f"Created {len(X)} sequences of shape {X.shape}")

        return X, y

    def fit_transform(
        self,
        df: pd.DataFrame,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Fit scalers and transform data.

        WARNING: Using fit_transform on the entire dataset before splitting
        causes DATA LEAKAGE. The scaler will learn statistics from future data.

        Use `prepare_train_val_test()` instead for proper pipeline.

        Args:
            df: DataFrame to fit and transform

        Returns:
            Tuple of (X sequences, y targets)
        """
        warnings.warn(
            "fit_transform() can cause data leakage if used on entire dataset. "
            "Use prepare_train_val_test() for proper train/val/test preparation.",
            UserWarning,
            stacklevel=2,
        )
        self.fit(df)
        return self.transform(df)

    def prepare_train_val_test(
        self,
        df: pd.DataFrame,
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
    ) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
        """
        Prepare train/val/test splits with NO DATA LEAKAGE.

        Scalers are fitted ONLY on training data, then applied to all splits.
        This is the recommended method for preparing data.

        Args:
            df: Clean DataFrame with features
            train_ratio: Proportion for training (default: 0.7)
            val_ratio: Proportion for validation (default: 0.15)

        Returns:
            Dict with 'train', 'val', 'test' keys, each containing (X, y) tuple
        """
        n = len(df)
        train_end = int(n * train_ratio)
        val_end = int(n * (train_ratio + val_ratio))

        # Split the DataFrame BEFORE any transformation
        df_train = df.iloc[:train_end]
        df_val = df.iloc[train_end:val_end]
        df_test = df.iloc[val_end:]

        logger.info(f"Splitting data: train={len(df_train)}, val={len(df_val)}, test={len(df_test)}")

        # Fit scalers ONLY on training data
        self.fit(df_train)
        logger.info("Scalers fitted on training data only (no data leakage)")

        # Transform each split separately
        X_train, y_train = self.transform(df_train)
        X_val, y_val = self.transform(df_val)
        X_test, y_test = self.transform(df_test)

        return {
            "train": (X_train, y_train),
            "val": (X_val, y_val),
            "test": (X_test, y_test),
        }

    def prepare_with_direction_labels(
        self,
        df: pd.DataFrame,
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        threshold: float = 0.0,
    ) -> Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """
        Prepare train/val/test splits with direction labels for classification.

        Scalers are fitted ONLY on training data.

        Args:
            df: Clean DataFrame with features
            train_ratio: Proportion for training
            val_ratio: Proportion for validation
            threshold: Minimum change % for up/down classification

        Returns:
            Dict with 'train', 'val', 'test' keys, each containing (X, y_price, y_direction) tuple
        """
        n = len(df)
        train_end = int(n * train_ratio)
        val_end = int(n * (train_ratio + val_ratio))

        # Split DataFrame BEFORE transformation
        df_train = df.iloc[:train_end]
        df_val = df.iloc[train_end:val_end]
        df_test = df.iloc[val_end:]

        # Fit scalers ONLY on training data
        self.fit(df_train)

        result = {}
        for name, split_df in [("train", df_train), ("val", df_val), ("test", df_test)]:
            X, y_price = self.transform(split_df)
            y_direction = self.get_direction_labels(split_df, threshold)
            # Align lengths (direction labels are shorter by horizon)
            min_len = min(len(X), len(y_direction))
            result[name] = (X[:min_len], y_price[:min_len], y_direction[:min_len])

        return result

    def inverse_transform_target(self, y: np.ndarray) -> np.ndarray:
        """
        Inverse transform scaled target values.

        Args:
            y: Scaled target values

        Returns:
            Original scale target values
        """
        if not self._fitted:
            raise ValueError("DataProcessor not fitted.")

        y_reshaped = y.reshape(-1, 1)
        return self.target_scaler.inverse_transform(y_reshaped).flatten()

    def prepare_latest(self, df: pd.DataFrame) -> np.ndarray:
        """
        Prepare the latest data point for prediction.

        Args:
            df: DataFrame with at least lookback rows

        Returns:
            Single sequence for prediction (1, lookback, features)
        """
        if not self._fitted:
            raise ValueError("DataProcessor not fitted.")

        if len(df) < self.lookback:
            raise ValueError(f"Need at least {self.lookback} rows, got {len(df)}")

        # Get latest lookback rows
        recent_df = df.tail(self.lookback)

        # Scale features
        feature_data = recent_df[self.feature_columns].values
        scaled_features = self.feature_scaler.transform(feature_data)

        # Reshape to (1, lookback, features)
        return scaled_features.reshape(1, self.lookback, -1)

    def get_direction_labels(
        self,
        df: pd.DataFrame,
        threshold: float = 0.0,
    ) -> np.ndarray:
        """
        Create direction labels for classification.

        Args:
            df: DataFrame with close prices
            threshold: Minimum change % to classify as up/down

        Returns:
            Array of labels: 0=down, 1=neutral, 2=up
        """
        close = df["close"].values
        future_close = np.roll(close, -self.horizon)

        # Calculate returns
        returns = (future_close - close) / close * 100

        # Create labels
        labels = np.ones(len(returns), dtype=int)  # Default to neutral
        labels[returns > threshold] = 2  # Up
        labels[returns < -threshold] = 0  # Down

        # Remove last horizon values (no future data)
        return labels[:-self.horizon]

    def split_data(
        self,
        X: np.ndarray,
        y: np.ndarray,
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
    ) -> Tuple[Tuple[np.ndarray, np.ndarray], ...]:
        """
        Split data into train/validation/test sets.

        Uses time-based split (no shuffling for time series).

        Args:
            X: Feature sequences
            y: Target values
            train_ratio: Proportion for training
            val_ratio: Proportion for validation

        Returns:
            Tuple of ((X_train, y_train), (X_val, y_val), (X_test, y_test))
        """
        n = len(X)
        train_end = int(n * train_ratio)
        val_end = int(n * (train_ratio + val_ratio))

        X_train, y_train = X[:train_end], y[:train_end]
        X_val, y_val = X[train_end:val_end], y[train_end:val_end]
        X_test, y_test = X[val_end:], y[val_end:]

        logger.info(
            f"Split data: train={len(X_train)}, val={len(X_val)}, test={len(X_test)}"
        )

        return (X_train, y_train), (X_val, y_val), (X_test, y_test)

    def save(self, path: str) -> None:
        """
        Save processor state to file.

        Args:
            path: Path to save file
        """
        state = {
            "lookback": self.lookback,
            "horizon": self.horizon,
            "scaler_type": self.scaler_type,
            "feature_columns": self.feature_columns,
            "feature_scaler": self.feature_scaler,
            "target_scaler": self.target_scaler,
            "_fitted": self._fitted,
        }
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(state, path)
        logger.info(f"Saved processor to {path}")

    @classmethod
    def load(cls, path: str) -> "DataProcessor":
        """
        Load processor state from file.

        Args:
            path: Path to load file

        Returns:
            Loaded DataProcessor instance
        """
        state = joblib.load(path)
        processor = cls(
            lookback=state["lookback"],
            horizon=state["horizon"],
            scaler_type=state["scaler_type"],
            feature_columns=state["feature_columns"],
        )
        processor.feature_scaler = state["feature_scaler"]
        processor.target_scaler = state["target_scaler"]
        processor._fitted = state["_fitted"]
        logger.info(f"Loaded processor from {path}")
        return processor


def create_sequences(
    features: np.ndarray,
    targets: np.ndarray,
    lookback: int = 60,
    horizon: int = 1,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create sequences for LSTM training.

    Args:
        features: Feature array of shape (samples, features)
        targets: Target array of shape (samples,)
        lookback: Number of timesteps to look back
        horizon: Number of timesteps to predict ahead

    Returns:
        Tuple of (X, y) arrays
    """
    X, y = [], []

    for i in range(lookback, len(features) - horizon + 1):
        X.append(features[i - lookback:i])
        y.append(targets[i + horizon - 1])

    return np.array(X), np.array(y)


class WalkForwardValidator:
    """
    Walk-forward validation for time series.

    Implements both expanding and sliding window approaches to validate
    models on truly out-of-sample data with rolling windows.

    Example:
        validator = WalkForwardValidator(
            n_splits=5,
            train_size=500,
            test_size=100,
            mode='sliding'
        )
        for train_idx, val_idx in validator.split(X):
            X_train, X_val = X[train_idx], X[val_idx]
            # Train and evaluate
    """

    def __init__(
        self,
        n_splits: int = 5,
        train_size: Optional[int] = None,
        test_size: Optional[int] = None,
        gap: int = 0,
        mode: str = "expanding",
    ):
        """
        Initialize walk-forward validator.

        Args:
            n_splits: Number of folds
            train_size: Initial training window size (required for sliding mode)
            test_size: Test/validation window size per fold
            gap: Number of samples to skip between train and test (prevents leakage)
            mode: 'expanding' (growing train) or 'sliding' (fixed train window)
        """
        self.n_splits = n_splits
        self.train_size = train_size
        self.test_size = test_size
        self.gap = gap
        self.mode = mode

        if mode not in ("expanding", "sliding"):
            raise ValueError("mode must be 'expanding' or 'sliding'")

        if mode == "sliding" and train_size is None:
            raise ValueError("train_size required for sliding mode")

    def split(
        self, X: np.ndarray
    ) -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:
        """
        Generate train/test indices for each fold.

        Args:
            X: Feature array (only used for length)

        Yields:
            Tuple of (train_indices, test_indices)
        """
        n_samples = len(X)

        # Calculate test size if not provided
        if self.test_size is None:
            # Reserve last 30% for validation folds
            available = int(n_samples * 0.3)
            test_size = available // self.n_splits
        else:
            test_size = self.test_size

        # Calculate initial train size for expanding mode
        if self.train_size is None:
            # Start with 50% of data
            initial_train = int(n_samples * 0.5)
        else:
            initial_train = self.train_size

        for fold in range(self.n_splits):
            if self.mode == "expanding":
                # Expanding window: train grows with each fold
                train_end = initial_train + fold * test_size
                train_start = 0
            else:
                # Sliding window: fixed train size
                train_start = fold * test_size
                train_end = train_start + self.train_size

            test_start = train_end + self.gap
            test_end = test_start + test_size

            # Ensure we don't exceed data bounds
            if test_end > n_samples:
                break

            train_idx = np.arange(train_start, train_end)
            test_idx = np.arange(test_start, test_end)

            yield train_idx, test_idx

    def get_n_splits(self, X: Optional[np.ndarray] = None) -> int:
        """Return number of splits."""
        return self.n_splits

    def validate(
        self,
        X: np.ndarray,
        y: np.ndarray,
        model_fn,
        metric_fn,
    ) -> Dict[str, Union[float, List[float]]]:
        """
        Run walk-forward validation and return metrics.

        Args:
            X: Feature array
            y: Target array
            model_fn: Function that takes (X_train, y_train) and returns fitted model
            metric_fn: Function that takes (y_true, y_pred) and returns metric

        Returns:
            Dict with 'mean', 'std', 'min', 'max', 'scores' keys
        """
        scores = []

        for fold, (train_idx, test_idx) in enumerate(self.split(X)):
            X_train, y_train = X[train_idx], y[train_idx]
            X_test, y_test = X[test_idx], y[test_idx]

            # Train model
            model = model_fn(X_train, y_train)

            # Predict and evaluate
            y_pred = model.predict(X_test)
            score = metric_fn(y_test, y_pred)
            scores.append(score)

            logger.info(f"Fold {fold + 1}: score = {score:.4f}")

        return {
            "mean": np.mean(scores),
            "std": np.std(scores),
            "min": np.min(scores),
            "max": np.max(scores),
            "scores": scores,
        }


class TimeSeriesCrossValidator:
    """
    Time series cross-validator wrapping sklearn's TimeSeriesSplit.

    Preserves temporal ordering and optionally adds gap between
    train and test to prevent look-ahead bias.

    Example:
        cv = TimeSeriesCrossValidator(n_splits=5, gap=10)
        for train_idx, val_idx in cv.split(X):
            # Train and evaluate
    """

    def __init__(
        self,
        n_splits: int = 5,
        gap: int = 0,
        max_train_size: Optional[int] = None,
        test_size: Optional[int] = None,
    ):
        """
        Initialize time series cross-validator.

        Args:
            n_splits: Number of folds
            gap: Gap between train and test to prevent leakage
            max_train_size: Maximum training set size
            test_size: Test set size for each fold
        """
        self.n_splits = n_splits
        self.gap = gap
        self.max_train_size = max_train_size
        self.test_size = test_size

        self._tscv = TimeSeriesSplit(
            n_splits=n_splits,
            gap=gap,
            max_train_size=max_train_size,
            test_size=test_size,
        )

    def split(
        self, X: np.ndarray
    ) -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:
        """
        Generate train/test indices for each fold.

        Args:
            X: Feature array

        Yields:
            Tuple of (train_indices, test_indices)
        """
        for train_idx, test_idx in self._tscv.split(X):
            yield train_idx, test_idx

    def get_n_splits(self, X: Optional[np.ndarray] = None) -> int:
        """Return number of splits."""
        return self.n_splits

    def cross_validate(
        self,
        X: np.ndarray,
        y: np.ndarray,
        model_fn,
        metric_fn,
        confidence_level: float = 0.95,
    ) -> Dict[str, Union[float, List[float]]]:
        """
        Run cross-validation and return metrics with confidence interval.

        Args:
            X: Feature array
            y: Target array
            model_fn: Function that takes (X_train, y_train) and returns fitted model
            metric_fn: Function that takes (y_true, y_pred) and returns metric
            confidence_level: Confidence level for interval (default: 95%)

        Returns:
            Dict with 'mean', 'std', 'ci_lower', 'ci_upper', 'scores' keys
        """
        from scipy import stats

        scores = []

        for fold, (train_idx, test_idx) in enumerate(self.split(X)):
            X_train, y_train = X[train_idx], y[train_idx]
            X_test, y_test = X[test_idx], y[test_idx]

            # Train model
            model = model_fn(X_train, y_train)

            # Predict and evaluate
            y_pred = model.predict(X_test)
            score = metric_fn(y_test, y_pred)
            scores.append(score)

            logger.info(
                f"Fold {fold + 1}/{self.n_splits}: "
                f"train={len(train_idx)}, test={len(test_idx)}, score={score:.4f}"
            )

        mean = np.mean(scores)
        std = np.std(scores)
        n = len(scores)

        # Calculate confidence interval using t-distribution
        t_value = stats.t.ppf((1 + confidence_level) / 2, n - 1)
        margin = t_value * std / np.sqrt(n)

        return {
            "mean": mean,
            "std": std,
            "ci_lower": mean - margin,
            "ci_upper": mean + margin,
            "scores": scores,
            "n_folds": n,
        }
