"""
Ensemble Predictor Module
=========================
Combines LSTM and XGBoost predictions for robust trading signals.
"""

from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch
from loguru import logger

from .lstm_model import GoldLSTM, LSTMTrainer
from .xgboost_model import GoldXGBoost


class Signal(Enum):
    """Trading signal types."""
    STRONG_BUY = "STRONG_BUY"
    BUY = "BUY"
    NEUTRAL = "NEUTRAL"
    SELL = "SELL"
    STRONG_SELL = "STRONG_SELL"


@dataclass
class Prediction:
    """Prediction result container."""
    signal: Signal
    confidence: float
    current_price: float
    predicted_price: float
    predicted_change_percent: float
    direction_probabilities: dict
    lstm_prediction: float
    xgb_prediction: dict
    timestamp: datetime


class EnsemblePredictor:
    """
    Ensemble predictor combining LSTM (price prediction) and XGBoost (direction classification).

    The ensemble works as follows:
    1. LSTM predicts the next price value
    2. XGBoost predicts the probability of direction (up/neutral/down)
    3. Predictions are combined using weighted voting
    4. Final signal is generated based on agreement and confidence
    """

    SIGNAL_THRESHOLDS = {
        "strong": 0.8,  # Strong signal when confidence > 80%
        "normal": 0.6,  # Normal signal when confidence > 60%
        "minimum": 0.5,  # Minimum confidence for any signal
    }

    def __init__(
        self,
        lstm_model: Optional[GoldLSTM] = None,
        xgb_model: Optional[GoldXGBoost] = None,
        lstm_weight: float = 0.6,
        xgb_weight: float = 0.4,
        confidence_threshold: float = 0.7,
    ):
        """
        Initialize ensemble predictor.

        Args:
            lstm_model: Trained LSTM model
            xgb_model: Trained XGBoost model
            lstm_weight: Weight for LSTM predictions
            xgb_weight: Weight for XGBoost predictions
            confidence_threshold: Minimum confidence for signals
        """
        self.lstm_model = lstm_model
        self.xgb_model = xgb_model
        self.lstm_weight = lstm_weight
        self.xgb_weight = xgb_weight
        self.confidence_threshold = confidence_threshold

        # Normalize weights
        total = lstm_weight + xgb_weight
        self.lstm_weight = lstm_weight / total
        self.xgb_weight = xgb_weight / total

    def load_models(
        self,
        lstm_path: str,
        xgb_path: str,
        device: str = "auto",
    ) -> None:
        """
        Load pre-trained models.

        Args:
            lstm_path: Path to LSTM model file
            xgb_path: Path to XGBoost model file
            device: Device for LSTM model
        """
        # Load LSTM
        self.lstm_model, _ = LSTMTrainer.load_model(lstm_path, device=device)
        self.lstm_model.eval()

        # Load XGBoost
        self.xgb_model = GoldXGBoost.load(xgb_path)

        logger.info("Ensemble models loaded successfully")

    def predict(
        self,
        X: np.ndarray,
        current_price: float,
        target_scaler=None,
    ) -> Prediction:
        """
        Generate ensemble prediction.

        Args:
            X: Input features of shape (1, lookback, features)
            current_price: Current market price
            target_scaler: Scaler for inverse transforming LSTM output

        Returns:
            Prediction object with signal and details
        """
        # Get individual predictions
        lstm_pred = self._get_lstm_prediction(X)
        xgb_probs = self._get_xgb_prediction(X)

        # Inverse transform LSTM prediction if scaler provided
        if target_scaler is not None:
            lstm_price = target_scaler.inverse_transform([[lstm_pred]])[0][0]
        else:
            lstm_price = lstm_pred * current_price  # Assume normalized

        # Calculate predicted change
        predicted_change = (lstm_price - current_price) / current_price * 100

        # Combine predictions
        signal, confidence = self._combine_predictions(
            lstm_prediction=lstm_price,
            current_price=current_price,
            xgb_probabilities=xgb_probs,
        )

        return Prediction(
            signal=signal,
            confidence=confidence,
            current_price=current_price,
            predicted_price=lstm_price,
            predicted_change_percent=predicted_change,
            direction_probabilities={
                "down": xgb_probs[0],
                "neutral": xgb_probs[1],
                "up": xgb_probs[2],
            },
            lstm_prediction=lstm_price,
            xgb_prediction={
                "down": xgb_probs[0],
                "neutral": xgb_probs[1],
                "up": xgb_probs[2],
            },
            timestamp=datetime.now(),
        )

    def _get_lstm_prediction(self, X: np.ndarray) -> float:
        """Get LSTM price prediction."""
        if self.lstm_model is None:
            raise ValueError("LSTM model not loaded")

        prediction = self.lstm_model.predict(X)
        return float(prediction[0][0])

    def _get_xgb_prediction(self, X: np.ndarray) -> np.ndarray:
        """Get XGBoost direction probabilities."""
        if self.xgb_model is None:
            raise ValueError("XGBoost model not loaded")

        probs = self.xgb_model.predict_proba(X)
        return probs[0]  # Shape: (3,) for [down, neutral, up]

    def _combine_predictions(
        self,
        lstm_prediction: float,
        current_price: float,
        xgb_probabilities: np.ndarray,
    ) -> Tuple[Signal, float]:
        """
        Combine LSTM and XGBoost predictions into final signal.

        Args:
            lstm_prediction: Predicted price from LSTM
            current_price: Current market price
            xgb_probabilities: Direction probabilities from XGBoost

        Returns:
            Tuple of (Signal, confidence)
        """
        # Calculate LSTM direction
        lstm_change_pct = (lstm_prediction - current_price) / current_price * 100

        # Convert LSTM prediction to direction signal
        if lstm_change_pct > 0.1:
            lstm_direction = 2  # Up
            lstm_confidence = min(abs(lstm_change_pct) / 0.5, 1.0)
        elif lstm_change_pct < -0.1:
            lstm_direction = 0  # Down
            lstm_confidence = min(abs(lstm_change_pct) / 0.5, 1.0)
        else:
            lstm_direction = 1  # Neutral
            lstm_confidence = 0.5

        # Get XGBoost prediction
        xgb_direction = np.argmax(xgb_probabilities)
        xgb_confidence = xgb_probabilities[xgb_direction]

        # Combine with weights
        if lstm_direction == xgb_direction:
            # Models agree - higher confidence
            combined_direction = lstm_direction
            combined_confidence = (
                lstm_confidence * self.lstm_weight +
                xgb_confidence * self.xgb_weight
            ) * 1.1  # Boost for agreement
            combined_confidence = min(combined_confidence, 1.0)
        else:
            # Models disagree - use weighted average
            direction_scores = np.array([0.0, 0.0, 0.0])

            # Add LSTM vote
            direction_scores[lstm_direction] += lstm_confidence * self.lstm_weight

            # Add XGBoost votes
            for i, prob in enumerate(xgb_probabilities):
                direction_scores[i] += prob * self.xgb_weight

            combined_direction = np.argmax(direction_scores)
            combined_confidence = direction_scores[combined_direction] * 0.9  # Penalty for disagreement

        # Generate signal based on direction and confidence
        signal = self._generate_signal(combined_direction, combined_confidence)

        return signal, combined_confidence

    def _generate_signal(self, direction: int, confidence: float) -> Signal:
        """
        Generate trading signal from direction and confidence.

        Args:
            direction: 0=down, 1=neutral, 2=up
            confidence: Confidence level (0-1)

        Returns:
            Trading signal
        """
        if confidence < self.SIGNAL_THRESHOLDS["minimum"]:
            return Signal.NEUTRAL

        if direction == 2:  # Up
            if confidence >= self.SIGNAL_THRESHOLDS["strong"]:
                return Signal.STRONG_BUY
            elif confidence >= self.SIGNAL_THRESHOLDS["normal"]:
                return Signal.BUY
            return Signal.NEUTRAL

        elif direction == 0:  # Down
            if confidence >= self.SIGNAL_THRESHOLDS["strong"]:
                return Signal.STRONG_SELL
            elif confidence >= self.SIGNAL_THRESHOLDS["normal"]:
                return Signal.SELL
            return Signal.NEUTRAL

        return Signal.NEUTRAL

    def is_strong_signal(self, prediction: Prediction) -> bool:
        """Check if prediction is a strong signal."""
        return prediction.signal in [Signal.STRONG_BUY, Signal.STRONG_SELL]

    def should_alert(self, prediction: Prediction) -> bool:
        """
        Determine if prediction should trigger an alert.

        Args:
            prediction: Prediction result

        Returns:
            True if alert should be sent
        """
        # Alert on strong signals
        if self.is_strong_signal(prediction):
            return True

        # Alert on regular signals with high confidence
        if prediction.signal in [Signal.BUY, Signal.SELL]:
            return prediction.confidence >= self.confidence_threshold

        return False

    def format_alert_message(self, prediction: Prediction) -> str:
        """
        Format prediction for alert message.

        Args:
            prediction: Prediction result

        Returns:
            Formatted alert string
        """
        emoji = {
            Signal.STRONG_BUY: "🟢🟢",
            Signal.BUY: "🟢",
            Signal.NEUTRAL: "⚪",
            Signal.SELL: "🔴",
            Signal.STRONG_SELL: "🔴🔴",
        }

        direction = {
            Signal.STRONG_BUY: "📈 STRONG UP",
            Signal.BUY: "📈 UP",
            Signal.NEUTRAL: "➡️ NEUTRAL",
            Signal.SELL: "📉 DOWN",
            Signal.STRONG_SELL: "📉 STRONG DOWN",
        }

        msg = f"""
{emoji[prediction.signal]} GOLD PREDICTION ALERT

Signal: {direction[prediction.signal]}
Confidence: {prediction.confidence:.1%}

Current Price: ${prediction.current_price:.2f}
Predicted: ${prediction.predicted_price:.2f}
Change: {prediction.predicted_change_percent:+.2f}%

Direction Probabilities:
  ↑ Up: {prediction.direction_probabilities['up']:.1%}
  → Neutral: {prediction.direction_probabilities['neutral']:.1%}
  ↓ Down: {prediction.direction_probabilities['down']:.1%}

Time: {prediction.timestamp.strftime('%Y-%m-%d %H:%M:%S')}
        """.strip()

        return msg

    def to_dict(self, prediction: Prediction) -> dict:
        """Convert prediction to dictionary for JSON serialization."""
        return {
            "signal": prediction.signal.value,
            "confidence": prediction.confidence,
            "current_price": prediction.current_price,
            "predicted_price": prediction.predicted_price,
            "predicted_change_percent": prediction.predicted_change_percent,
            "direction_probabilities": prediction.direction_probabilities,
            "lstm_prediction": prediction.lstm_prediction,
            "xgb_prediction": prediction.xgb_prediction,
            "timestamp": prediction.timestamp.isoformat(),
            "should_alert": self.should_alert(prediction),
        }


class StackingEnsemble:
    """
    Stacking ensemble that trains a meta-model on base model predictions (US-017).

    The meta-model learns the optimal way to combine predictions based on
    the validation set performance, avoiding data leakage.
    """

    def __init__(
        self,
        meta_model_type: str = "logistic",
        min_weight: float = 0.2,
    ):
        """
        Initialize stacking ensemble.

        Args:
            meta_model_type: 'logistic', 'xgboost', or 'neural'
            min_weight: Minimum weight for any model (prevents collapse)
        """
        self.meta_model_type = meta_model_type
        self.min_weight = min_weight
        self.meta_model = None
        self.is_fitted = False

    def fit(
        self,
        lstm_val_preds: np.ndarray,
        xgb_val_preds: np.ndarray,
        y_val: np.ndarray,
    ) -> "StackingEnsemble":
        """
        Train meta-model on validation set predictions.

        Uses ONLY validation predictions to prevent data leakage.

        Args:
            lstm_val_preds: LSTM predictions on validation set
            xgb_val_preds: XGBoost predictions on validation set (probabilities)
            y_val: True validation labels

        Returns:
            Self for chaining
        """
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler

        # Stack base model predictions as features
        X_meta = np.column_stack([
            lstm_val_preds.reshape(-1, 1) if lstm_val_preds.ndim == 1 else lstm_val_preds,
            xgb_val_preds if xgb_val_preds.ndim > 1 else xgb_val_preds.reshape(-1, 1),
        ])

        # Scale meta features
        self.meta_scaler = StandardScaler()
        X_meta_scaled = self.meta_scaler.fit_transform(X_meta)

        # Train meta-model
        if self.meta_model_type == "logistic":
            self.meta_model = LogisticRegression(
                max_iter=1000,
                random_state=42,
                class_weight="balanced",
            )
            self.meta_model.fit(X_meta_scaled, y_val)

        elif self.meta_model_type == "ridge":
            from sklearn.linear_model import RidgeClassifier
            self.meta_model = RidgeClassifier(
                alpha=1.0,
                random_state=42,
            )
            self.meta_model.fit(X_meta_scaled, y_val)

        elif self.meta_model_type == "xgboost":
            from xgboost import XGBClassifier
            self.meta_model = XGBClassifier(
                n_estimators=100,
                max_depth=3,
                learning_rate=0.1,
                random_state=42,
            )
            self.meta_model.fit(X_meta_scaled, y_val)

        elif self.meta_model_type == "neural":
            # Simple 2-layer MLP
            import torch
            import torch.nn as nn

            class MetaMLP(nn.Module):
                def __init__(self, input_size, n_classes=3):
                    super().__init__()
                    self.fc = nn.Sequential(
                        nn.Linear(input_size, 16),
                        nn.ReLU(),
                        nn.Dropout(0.2),
                        nn.Linear(16, n_classes),
                    )

                def forward(self, x):
                    return self.fc(x)

            self.meta_model = MetaMLP(X_meta_scaled.shape[1])
            # Training loop (simplified)
            optimizer = torch.optim.Adam(self.meta_model.parameters(), lr=0.01)
            criterion = nn.CrossEntropyLoss()
            X_tensor = torch.FloatTensor(X_meta_scaled)
            y_tensor = torch.LongTensor(y_val)

            for epoch in range(100):
                optimizer.zero_grad()
                output = self.meta_model(X_tensor)
                loss = criterion(output, y_tensor)
                loss.backward()
                optimizer.step()

        self.is_fitted = True
        logger.info(f"Stacking meta-model ({self.meta_model_type}) trained successfully")
        return self

    def predict(
        self,
        lstm_pred: np.ndarray,
        xgb_pred: np.ndarray,
    ) -> np.ndarray:
        """
        Make stacked prediction.

        Args:
            lstm_pred: LSTM prediction(s)
            xgb_pred: XGBoost prediction(s) (probabilities)

        Returns:
            Final class predictions
        """
        if not self.is_fitted:
            raise ValueError("Meta-model not fitted. Call fit() first.")

        # Stack predictions
        X_meta = np.column_stack([
            lstm_pred.reshape(-1, 1) if lstm_pred.ndim == 1 else lstm_pred,
            xgb_pred if xgb_pred.ndim > 1 else xgb_pred.reshape(-1, 1),
        ])
        X_meta_scaled = self.meta_scaler.transform(X_meta)

        if self.meta_model_type == "neural":
            import torch
            self.meta_model.eval()
            with torch.no_grad():
                output = self.meta_model(torch.FloatTensor(X_meta_scaled))
                return output.argmax(dim=1).numpy()
        else:
            return self.meta_model.predict(X_meta_scaled)

    def predict_proba(
        self,
        lstm_pred: np.ndarray,
        xgb_pred: np.ndarray,
    ) -> np.ndarray:
        """Get prediction probabilities."""
        if not self.is_fitted:
            raise ValueError("Meta-model not fitted. Call fit() first.")

        X_meta = np.column_stack([
            lstm_pred.reshape(-1, 1) if lstm_pred.ndim == 1 else lstm_pred,
            xgb_pred if xgb_pred.ndim > 1 else xgb_pred.reshape(-1, 1),
        ])
        X_meta_scaled = self.meta_scaler.transform(X_meta)

        if self.meta_model_type == "neural":
            import torch
            import torch.nn.functional as F
            self.meta_model.eval()
            with torch.no_grad():
                output = self.meta_model(torch.FloatTensor(X_meta_scaled))
                return F.softmax(output, dim=1).numpy()
        else:
            return self.meta_model.predict_proba(X_meta_scaled)


class DynamicWeightCalibrator:
    """
    Dynamic ensemble weight calibration based on recent performance (US-018).

    Adjusts model weights based on rolling window accuracy, giving more
    weight to models that have been performing better recently.
    """

    def __init__(
        self,
        window_size: int = 100,
        min_weight: float = 0.2,
        max_weight: float = 0.8,
        decay_factor: float = 0.95,
    ):
        """
        Initialize calibrator.

        Args:
            window_size: Rolling window for performance calculation
            min_weight: Minimum weight for any model
            max_weight: Maximum weight for any model
            decay_factor: Exponential decay for older predictions
        """
        self.window_size = window_size
        self.min_weight = min_weight
        self.max_weight = max_weight
        self.decay_factor = decay_factor

        # History tracking
        self.lstm_history = []  # (prediction, actual, timestamp)
        self.xgb_history = []

        # Current weights
        self.lstm_weight = 0.5
        self.xgb_weight = 0.5

    def update(
        self,
        lstm_pred: int,
        xgb_pred: int,
        actual: int,
        timestamp: Optional[datetime] = None,
    ) -> Tuple[float, float]:
        """
        Update weights based on new prediction result.

        Args:
            lstm_pred: LSTM direction prediction (0, 1, or 2)
            xgb_pred: XGBoost direction prediction
            actual: Actual direction

        Returns:
            Updated (lstm_weight, xgb_weight)
        """
        timestamp = timestamp or datetime.now()

        # Record predictions
        self.lstm_history.append({
            "pred": lstm_pred,
            "actual": actual,
            "correct": lstm_pred == actual,
            "timestamp": timestamp,
        })
        self.xgb_history.append({
            "pred": xgb_pred,
            "actual": actual,
            "correct": xgb_pred == actual,
            "timestamp": timestamp,
        })

        # Keep only recent history
        self.lstm_history = self.lstm_history[-self.window_size:]
        self.xgb_history = self.xgb_history[-self.window_size:]

        # Recalibrate weights
        self._calibrate()

        return self.lstm_weight, self.xgb_weight

    def _calibrate(self) -> None:
        """Recalibrate weights based on recent performance."""
        if len(self.lstm_history) < 10:
            # Not enough data, keep default weights
            return

        # Calculate weighted accuracy (recent predictions weighted more)
        lstm_accuracy = self._weighted_accuracy(self.lstm_history)
        xgb_accuracy = self._weighted_accuracy(self.xgb_history)

        # Calculate raw weights based on accuracy
        total = lstm_accuracy + xgb_accuracy
        if total > 0:
            lstm_raw = lstm_accuracy / total
            xgb_raw = xgb_accuracy / total
        else:
            lstm_raw = xgb_raw = 0.5

        # Apply min/max constraints
        self.lstm_weight = np.clip(lstm_raw, self.min_weight, self.max_weight)
        self.xgb_weight = np.clip(xgb_raw, self.min_weight, self.max_weight)

        # Normalize
        total = self.lstm_weight + self.xgb_weight
        self.lstm_weight /= total
        self.xgb_weight /= total

    def _weighted_accuracy(self, history: list) -> float:
        """Calculate exponentially weighted accuracy."""
        if not history:
            return 0.5

        weights = [self.decay_factor ** i for i in range(len(history) - 1, -1, -1)]
        weighted_correct = sum(
            w * h["correct"] for w, h in zip(weights, history)
        )
        total_weight = sum(weights)

        return weighted_correct / total_weight if total_weight > 0 else 0.5

    def get_weights(self) -> Tuple[float, float]:
        """Get current weights."""
        return self.lstm_weight, self.xgb_weight

    def get_performance_summary(self) -> dict:
        """Get performance summary for logging/monitoring."""
        lstm_correct = sum(1 for h in self.lstm_history if h["correct"])
        xgb_correct = sum(1 for h in self.xgb_history if h["correct"])
        n = len(self.lstm_history)

        return {
            "lstm_accuracy": lstm_correct / n if n > 0 else 0,
            "xgb_accuracy": xgb_correct / n if n > 0 else 0,
            "lstm_weight": self.lstm_weight,
            "xgb_weight": self.xgb_weight,
            "n_samples": n,
        }
