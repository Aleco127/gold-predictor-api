"""
Data Augmentation Module
========================
Time series data augmentation for improving model generalization.
"""

from typing import Optional, Tuple

import numpy as np
from loguru import logger

try:
    from imblearn.over_sampling import SMOTE
    SMOTE_AVAILABLE = True
except ImportError:
    SMOTE_AVAILABLE = False
    logger.warning("imbalanced-learn not installed. SMOTE unavailable.")


class TimeSeriesAugmenter:
    """
    Time series data augmentation for LSTM training (US-019).

    Supports:
    - Jittering: Add small random noise
    - Scaling: Multiply by random factor
    - Window slicing: Random subsequence selection
    - Magnitude warping: Smooth random scaling
    """

    def __init__(
        self,
        jitter_prob: float = 0.5,
        jitter_std: float = 0.005,
        scale_prob: float = 0.5,
        scale_range: Tuple[float, float] = (0.95, 1.05),
        window_slice_prob: float = 0.3,
        window_slice_ratio: float = 0.9,
        seed: Optional[int] = None,
    ):
        """
        Initialize augmenter.

        Args:
            jitter_prob: Probability of applying jittering
            jitter_std: Standard deviation of noise (as fraction of value)
            scale_prob: Probability of applying scaling
            scale_range: Min and max scaling factors
            window_slice_prob: Probability of window slicing
            window_slice_ratio: Ratio of sequence to keep when slicing
            seed: Random seed for reproducibility
        """
        self.jitter_prob = jitter_prob
        self.jitter_std = jitter_std
        self.scale_prob = scale_prob
        self.scale_range = scale_range
        self.window_slice_prob = window_slice_prob
        self.window_slice_ratio = window_slice_ratio

        self.rng = np.random.default_rng(seed)

    def jitter(self, X: np.ndarray) -> np.ndarray:
        """
        Add small random noise to sequences.

        Args:
            X: Input sequences of shape (batch, seq_len, features)

        Returns:
            Jittered sequences
        """
        noise = self.rng.normal(0, self.jitter_std, X.shape)
        return X + noise * np.abs(X)  # Scale noise by magnitude

    def scale(self, X: np.ndarray) -> np.ndarray:
        """
        Scale sequences by random factor.

        Args:
            X: Input sequences

        Returns:
            Scaled sequences
        """
        # Different scale factor per sample
        batch_size = X.shape[0]
        scales = self.rng.uniform(
            self.scale_range[0],
            self.scale_range[1],
            size=(batch_size, 1, 1),
        )
        return X * scales

    def window_slice(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Random window slicing - take random subsequences.

        Note: This changes sequence length, so may need padding afterward.

        Args:
            X: Input sequences of shape (batch, seq_len, features)
            y: Target values

        Returns:
            Sliced sequences and corresponding targets
        """
        batch_size, seq_len, features = X.shape
        slice_len = int(seq_len * self.window_slice_ratio)

        X_sliced = []
        for i in range(batch_size):
            start = self.rng.integers(0, seq_len - slice_len + 1)
            X_sliced.append(X[i, start:start + slice_len, :])

        return np.array(X_sliced), y

    def magnitude_warp(
        self,
        X: np.ndarray,
        sigma: float = 0.2,
        knot: int = 4,
    ) -> np.ndarray:
        """
        Apply smooth random scaling along time axis.

        Creates a smooth curve that scales different parts of the sequence
        by different amounts.

        Args:
            X: Input sequences
            sigma: Standard deviation of warp
            knot: Number of control points

        Returns:
            Warped sequences
        """
        from scipy.interpolate import CubicSpline

        batch_size, seq_len, features = X.shape

        X_warped = np.zeros_like(X)
        for i in range(batch_size):
            # Create random control points
            orig_steps = np.arange(seq_len)
            random_warp = self.rng.normal(1.0, sigma, (knot + 2,))
            warp_steps = np.linspace(0, seq_len - 1, num=knot + 2)

            # Interpolate to full sequence length
            cs = CubicSpline(warp_steps, random_warp)
            warp_factors = cs(orig_steps)

            # Apply warp
            X_warped[i] = X[i] * warp_factors.reshape(-1, 1)

        return X_warped

    def augment(
        self,
        X: np.ndarray,
        y: np.ndarray,
        include_original: bool = True,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply random augmentations to training data.

        Args:
            X: Input sequences of shape (batch, seq_len, features)
            y: Target values
            include_original: Include original samples in output

        Returns:
            Augmented (X, y) tuple
        """
        augmented_X = []
        augmented_y = []

        if include_original:
            augmented_X.append(X)
            augmented_y.append(y)

        # Apply augmentations based on probability
        for i in range(len(X)):
            x_sample = X[i:i + 1]
            y_sample = y[i:i + 1]

            # Jittering
            if self.rng.random() < self.jitter_prob:
                x_aug = self.jitter(x_sample)
                augmented_X.append(x_aug)
                augmented_y.append(y_sample)

            # Scaling
            if self.rng.random() < self.scale_prob:
                x_aug = self.scale(x_sample)
                augmented_X.append(x_aug)
                augmented_y.append(y_sample)

        # Concatenate all augmented samples
        X_out = np.concatenate(augmented_X, axis=0)
        y_out = np.concatenate(augmented_y, axis=0)

        # Shuffle
        indices = self.rng.permutation(len(X_out))
        X_out = X_out[indices]
        y_out = y_out[indices]

        logger.info(f"Augmented data: {len(X)} -> {len(X_out)} samples")
        return X_out, y_out


class SMOTEAugmenter:
    """
    SMOTE oversampling for class imbalance in direction classification (US-020).

    Uses imbalanced-learn's SMOTE to oversample minority classes.
    """

    def __init__(
        self,
        sampling_strategy: str = "auto",
        k_neighbors: int = 5,
        seed: Optional[int] = None,
    ):
        """
        Initialize SMOTE augmenter.

        Args:
            sampling_strategy: 'auto', 'minority', 'not majority', or dict
            k_neighbors: Number of neighbors for SMOTE
            seed: Random seed
        """
        if not SMOTE_AVAILABLE:
            raise ImportError(
                "imbalanced-learn not installed. "
                "Install with: pip install imbalanced-learn"
            )

        self.sampling_strategy = sampling_strategy
        self.k_neighbors = k_neighbors
        self.seed = seed

    def resample(
        self,
        X: np.ndarray,
        y: np.ndarray,
        log_balance: bool = True,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply SMOTE oversampling.

        Args:
            X: Features (will be flattened for SMOTE, then reshaped)
            y: Direction labels (0=down, 1=neutral, 2=up)
            log_balance: Log class distribution before/after

        Returns:
            Resampled (X, y) tuple
        """
        original_shape = X.shape

        # Flatten for SMOTE (required 2D input)
        X_flat = X.reshape(X.shape[0], -1)

        # Log original balance
        if log_balance:
            unique, counts = np.unique(y, return_counts=True)
            logger.info(f"Original class distribution: {dict(zip(unique, counts))}")

        # Apply SMOTE
        smote = SMOTE(
            sampling_strategy=self.sampling_strategy,
            k_neighbors=self.k_neighbors,
            random_state=self.seed,
        )

        X_resampled, y_resampled = smote.fit_resample(X_flat, y)

        # Reshape back to sequences
        n_samples = X_resampled.shape[0]
        X_resampled = X_resampled.reshape(n_samples, original_shape[1], original_shape[2])

        # Log new balance
        if log_balance:
            unique, counts = np.unique(y_resampled, return_counts=True)
            logger.info(f"Resampled class distribution: {dict(zip(unique, counts))}")

        return X_resampled, y_resampled


def augment_training_data(
    X_train: np.ndarray,
    y_train: np.ndarray,
    y_direction: Optional[np.ndarray] = None,
    jitter: bool = True,
    scale: bool = True,
    smote: bool = False,
    seed: Optional[int] = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convenience function to augment training data.

    Args:
        X_train: Training features
        y_train: Training targets (regression)
        y_direction: Direction labels for SMOTE (classification)
        jitter: Apply jittering augmentation
        scale: Apply scaling augmentation
        smote: Apply SMOTE (requires y_direction)
        seed: Random seed

    Returns:
        Augmented (X, y) tuple
    """
    # Time series augmentation
    if jitter or scale:
        augmenter = TimeSeriesAugmenter(
            jitter_prob=0.5 if jitter else 0,
            scale_prob=0.5 if scale else 0,
            seed=seed,
        )
        X_train, y_train = augmenter.augment(X_train, y_train)

    # SMOTE for direction labels
    if smote and y_direction is not None and SMOTE_AVAILABLE:
        smote_augmenter = SMOTEAugmenter(seed=seed)
        X_train, y_direction = smote_augmenter.resample(X_train, y_direction)
        # Align y_train with resampled indices (simplified - just return direction labels)
        return X_train, y_direction

    return X_train, y_train
