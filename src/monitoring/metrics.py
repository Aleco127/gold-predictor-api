"""
Metrics Collection Module
=========================
Prometheus-compatible metrics for monitoring predictions.
"""

from datetime import datetime
from typing import Optional

from loguru import logger

try:
    from prometheus_client import Counter, Gauge, Histogram, Info
    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False
    logger.warning("prometheus_client not available")


class MetricsCollector:
    """
    Collects and exposes prediction metrics.

    Metrics:
    - prediction_count: Total predictions made
    - prediction_latency: Prediction processing time
    - signal_count: Count by signal type
    - confidence_distribution: Histogram of confidence scores
    - model_info: Model version and configuration
    """

    def __init__(self, prefix: str = "gold_predictor"):
        """
        Initialize metrics collector.

        Args:
            prefix: Metric name prefix
        """
        self.prefix = prefix
        self._initialized = False

        if PROMETHEUS_AVAILABLE:
            self._init_metrics()

    def _init_metrics(self):
        """Initialize Prometheus metrics."""
        # Counters
        self.prediction_count = Counter(
            f"{self.prefix}_predictions_total",
            "Total number of predictions made",
            ["symbol"],
        )

        self.signal_count = Counter(
            f"{self.prefix}_signals_total",
            "Count of signals by type",
            ["signal_type"],
        )

        self.alert_count = Counter(
            f"{self.prefix}_alerts_total",
            "Total alerts sent",
            ["channel"],
        )

        self.error_count = Counter(
            f"{self.prefix}_errors_total",
            "Total errors encountered",
            ["error_type"],
        )

        # Gauges
        self.current_price = Gauge(
            f"{self.prefix}_current_price",
            "Current gold price",
            ["symbol"],
        )

        self.predicted_price = Gauge(
            f"{self.prefix}_predicted_price",
            "Predicted gold price",
            ["symbol"],
        )

        self.confidence = Gauge(
            f"{self.prefix}_confidence",
            "Current prediction confidence",
            ["symbol"],
        )

        self.model_accuracy = Gauge(
            f"{self.prefix}_model_accuracy",
            "Model accuracy (if tracked)",
            ["model_type"],
        )

        # Histograms
        self.prediction_latency = Histogram(
            f"{self.prefix}_prediction_latency_seconds",
            "Time to generate prediction",
            ["symbol"],
            buckets=[0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0],
        )

        self.confidence_distribution = Histogram(
            f"{self.prefix}_confidence_distribution",
            "Distribution of confidence scores",
            buckets=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
        )

        # Info
        self.model_info = Info(
            f"{self.prefix}_model",
            "Model information",
        )

        self._initialized = True
        logger.info("Prometheus metrics initialized")

    def record_prediction(
        self,
        symbol: str,
        signal: str,
        confidence: float,
        current_price: float,
        predicted_price: float,
        latency: Optional[float] = None,
    ):
        """
        Record a prediction.

        Args:
            symbol: Trading symbol
            signal: Signal type (BUY, SELL, etc.)
            confidence: Confidence score
            current_price: Current price
            predicted_price: Predicted price
            latency: Prediction latency in seconds
        """
        if not PROMETHEUS_AVAILABLE or not self._initialized:
            return

        self.prediction_count.labels(symbol=symbol).inc()
        self.signal_count.labels(signal_type=signal).inc()
        self.current_price.labels(symbol=symbol).set(current_price)
        self.predicted_price.labels(symbol=symbol).set(predicted_price)
        self.confidence.labels(symbol=symbol).set(confidence)
        self.confidence_distribution.observe(confidence)

        if latency is not None:
            self.prediction_latency.labels(symbol=symbol).observe(latency)

    def record_alert(self, channel: str = "telegram"):
        """Record an alert sent."""
        if not PROMETHEUS_AVAILABLE or not self._initialized:
            return

        self.alert_count.labels(channel=channel).inc()

    def record_error(self, error_type: str):
        """Record an error."""
        if not PROMETHEUS_AVAILABLE or not self._initialized:
            return

        self.error_count.labels(error_type=error_type).inc()

    def set_model_info(
        self,
        lstm_version: str = "1.0.0",
        xgb_version: str = "1.0.0",
        last_trained: Optional[str] = None,
    ):
        """Set model information."""
        if not PROMETHEUS_AVAILABLE or not self._initialized:
            return

        self.model_info.info({
            "lstm_version": lstm_version,
            "xgb_version": xgb_version,
            "last_trained": last_trained or datetime.now().isoformat(),
        })

    def set_model_accuracy(self, model_type: str, accuracy: float):
        """Set model accuracy gauge."""
        if not PROMETHEUS_AVAILABLE or not self._initialized:
            return

        self.model_accuracy.labels(model_type=model_type).set(accuracy)
