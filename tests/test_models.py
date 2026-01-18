"""Tests for ML models."""

import pytest
import numpy as np
import torch

from src.models.lstm_model import GoldLSTM, LSTMTrainer
from src.models.ensemble import EnsemblePredictor, Signal, Prediction


@pytest.fixture
def sample_sequences():
    """Create sample sequence data."""
    np.random.seed(42)
    n_samples = 100
    seq_length = 60
    n_features = 20

    X = np.random.randn(n_samples, seq_length, n_features).astype(np.float32)
    y = np.random.randn(n_samples).astype(np.float32)

    return X, y


class TestGoldLSTM:
    """Test suite for LSTM model."""

    def test_init(self):
        """Test model initialization."""
        model = GoldLSTM(
            input_size=20,
            hidden_size=64,
            num_layers=2,
        )
        assert model.input_size == 20
        assert model.hidden_size == 64

    def test_forward(self, sample_sequences):
        """Test forward pass."""
        X, _ = sample_sequences
        model = GoldLSTM(input_size=X.shape[-1], hidden_size=64)

        x_tensor = torch.FloatTensor(X[:10])
        output = model(x_tensor)

        assert output.shape == (10, 1)

    def test_predict(self, sample_sequences):
        """Test numpy prediction."""
        X, _ = sample_sequences
        model = GoldLSTM(input_size=X.shape[-1], hidden_size=64)

        predictions = model.predict(X[:10])

        assert predictions.shape == (10, 1)
        assert isinstance(predictions, np.ndarray)

    def test_training(self, sample_sequences):
        """Test training loop."""
        X, y = sample_sequences
        model = GoldLSTM(input_size=X.shape[-1], hidden_size=32, num_layers=1)
        trainer = LSTMTrainer(model, learning_rate=0.01, device="cpu")

        # Train for just a few epochs
        history = trainer.train(
            X[:80], y[:80],
            X[80:], y[80:],
            epochs=3,
            batch_size=16,
            early_stopping=10,
        )

        assert "train_losses" in history
        assert len(history["train_losses"]) == 3

    def test_bidirectional(self, sample_sequences):
        """Test bidirectional LSTM."""
        X, _ = sample_sequences
        model = GoldLSTM(
            input_size=X.shape[-1],
            hidden_size=32,
            bidirectional=True,
        )

        x_tensor = torch.FloatTensor(X[:5])
        output = model(x_tensor)

        assert output.shape == (5, 1)


class TestEnsemblePredictor:
    """Test suite for Ensemble Predictor."""

    def test_signal_generation(self):
        """Test signal generation."""
        predictor = EnsemblePredictor()

        # Test strong signal
        signal = predictor._generate_signal(2, 0.85)  # High confidence up
        assert signal == Signal.STRONG_BUY

        # Test normal signal
        signal = predictor._generate_signal(0, 0.65)  # Medium confidence down
        assert signal == Signal.SELL

        # Test neutral
        signal = predictor._generate_signal(1, 0.6)
        assert signal == Signal.NEUTRAL

    def test_format_alert_message(self):
        """Test alert message formatting."""
        predictor = EnsemblePredictor()

        prediction = Prediction(
            signal=Signal.STRONG_BUY,
            confidence=0.85,
            current_price=2650.0,
            predicted_price=2655.0,
            predicted_change_percent=0.19,
            direction_probabilities={"up": 0.8, "neutral": 0.15, "down": 0.05},
            lstm_prediction=2655.0,
            xgb_prediction={"up": 0.8, "neutral": 0.15, "down": 0.05},
            timestamp=pytest.importorskip("datetime").datetime.now(),
        )

        message = predictor.format_alert_message(prediction)

        assert "GOLD" in message
        assert "2650" in message
        assert "2655" in message

    def test_should_alert(self):
        """Test alert decision logic."""
        predictor = EnsemblePredictor(confidence_threshold=0.7)

        # Strong signal - should alert
        pred_strong = Prediction(
            signal=Signal.STRONG_BUY,
            confidence=0.9,
            current_price=2650.0,
            predicted_price=2655.0,
            predicted_change_percent=0.19,
            direction_probabilities={},
            lstm_prediction=2655.0,
            xgb_prediction={},
            timestamp=pytest.importorskip("datetime").datetime.now(),
        )
        assert predictor.should_alert(pred_strong)

        # Neutral signal - should not alert
        pred_neutral = Prediction(
            signal=Signal.NEUTRAL,
            confidence=0.5,
            current_price=2650.0,
            predicted_price=2650.5,
            predicted_change_percent=0.02,
            direction_probabilities={},
            lstm_prediction=2650.5,
            xgb_prediction={},
            timestamp=pytest.importorskip("datetime").datetime.now(),
        )
        assert not predictor.should_alert(pred_neutral)

    def test_to_dict(self):
        """Test prediction serialization."""
        predictor = EnsemblePredictor()

        prediction = Prediction(
            signal=Signal.BUY,
            confidence=0.75,
            current_price=2650.0,
            predicted_price=2653.0,
            predicted_change_percent=0.11,
            direction_probabilities={"up": 0.7, "neutral": 0.2, "down": 0.1},
            lstm_prediction=2653.0,
            xgb_prediction={"up": 0.7, "neutral": 0.2, "down": 0.1},
            timestamp=pytest.importorskip("datetime").datetime.now(),
        )

        result = predictor.to_dict(prediction)

        assert result["signal"] == "BUY"
        assert result["confidence"] == 0.75
        assert "should_alert" in result
