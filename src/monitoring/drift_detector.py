"""
Model and Data Drift Detection Module (US-023)
===============================================
Monitors model performance and data distribution for drift.
"""

from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
from loguru import logger

try:
    from scipy import stats
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    logger.warning("scipy not available. Some drift detection methods unavailable.")


@dataclass
class DriftAlert:
    """Alert for detected drift."""
    alert_type: str  # "data_drift", "model_drift", "performance_degradation"
    severity: str  # "warning", "critical"
    metric_name: str
    current_value: float
    baseline_value: float
    threshold: float
    timestamp: datetime = field(default_factory=datetime.utcnow)
    details: str = ""


@dataclass
class PerformanceMetrics:
    """Container for performance metrics."""
    timestamp: datetime
    actual: float
    predicted: float
    direction_actual: int  # -1, 0, 1
    direction_predicted: int
    error: float
    correct_direction: bool


class DriftDetector:
    """
    Detects model and data drift (US-023).

    Monitors:
    - Feature distribution drift (KS test, PSI)
    - Model performance drift (rolling accuracy degradation)
    - Prediction confidence drift
    """

    def __init__(
        self,
        window_size: int = 1000,
        baseline_window: int = 5000,
        drift_threshold: float = 0.05,
        performance_degradation_threshold: float = 0.1,
    ):
        """
        Initialize drift detector.

        Args:
            window_size: Size of recent window for drift detection
            baseline_window: Size of baseline data window
            drift_threshold: P-value threshold for statistical tests
            performance_degradation_threshold: Threshold for performance drop alert
        """
        self.window_size = window_size
        self.baseline_window = baseline_window
        self.drift_threshold = drift_threshold
        self.performance_degradation_threshold = performance_degradation_threshold

        # Storage for baseline and recent data
        self._baseline_features: Dict[str, deque] = {}
        self._recent_features: Dict[str, deque] = {}
        self._performance_history: deque = deque(maxlen=baseline_window)
        self._predictions_log: deque = deque(maxlen=window_size)
        self._alerts: List[DriftAlert] = []
        self._alert_callbacks: List[Callable[[DriftAlert], None]] = []

        # Performance tracking
        self._baseline_accuracy: Optional[float] = None
        self._baseline_mae: Optional[float] = None

    def register_alert_callback(self, callback: Callable[[DriftAlert], None]) -> None:
        """Register a callback to be called when drift is detected."""
        self._alert_callbacks.append(callback)

    def _trigger_alert(self, alert: DriftAlert) -> None:
        """Trigger alert and notify callbacks."""
        self._alerts.append(alert)
        logger.warning(
            f"[{alert.alert_type}] {alert.metric_name}: "
            f"{alert.current_value:.4f} vs baseline {alert.baseline_value:.4f} "
            f"(threshold: {alert.threshold})"
        )
        for callback in self._alert_callbacks:
            try:
                callback(alert)
            except Exception as e:
                logger.error(f"Error in alert callback: {e}")

    def add_feature_sample(
        self,
        feature_name: str,
        value: float,
        is_baseline: bool = False,
    ) -> None:
        """
        Add a feature sample for drift detection.

        Args:
            feature_name: Name of the feature
            value: Feature value
            is_baseline: If True, add to baseline; otherwise to recent window
        """
        if is_baseline:
            if feature_name not in self._baseline_features:
                self._baseline_features[feature_name] = deque(maxlen=self.baseline_window)
            self._baseline_features[feature_name].append(value)
        else:
            if feature_name not in self._recent_features:
                self._recent_features[feature_name] = deque(maxlen=self.window_size)
            self._recent_features[feature_name].append(value)

    def add_feature_batch(
        self,
        features: Dict[str, np.ndarray],
        is_baseline: bool = False,
    ) -> None:
        """Add batch of feature samples."""
        for feature_name, values in features.items():
            for value in values:
                self.add_feature_sample(feature_name, value, is_baseline)

    def record_prediction(
        self,
        actual: float,
        predicted: float,
        timestamp: Optional[datetime] = None,
    ) -> None:
        """
        Record a prediction for performance tracking.

        Args:
            actual: Actual value
            predicted: Predicted value
            timestamp: Time of prediction (defaults to now)
        """
        timestamp = timestamp or datetime.utcnow()

        # Calculate direction
        direction_actual = 1 if actual > 0 else (-1 if actual < 0 else 0)
        direction_predicted = 1 if predicted > 0 else (-1 if predicted < 0 else 0)

        metrics = PerformanceMetrics(
            timestamp=timestamp,
            actual=actual,
            predicted=predicted,
            direction_actual=direction_actual,
            direction_predicted=direction_predicted,
            error=abs(actual - predicted),
            correct_direction=(direction_actual == direction_predicted),
        )

        self._performance_history.append(metrics)
        self._predictions_log.append(metrics)

    def check_data_drift(self, feature_name: str) -> Tuple[bool, float, Dict]:
        """
        Check for data drift on a specific feature using KS test.

        Args:
            feature_name: Feature to check

        Returns:
            Tuple of (drift_detected, p_value, details)
        """
        if not SCIPY_AVAILABLE:
            return False, 1.0, {"error": "scipy not available"}

        baseline = self._baseline_features.get(feature_name)
        recent = self._recent_features.get(feature_name)

        if baseline is None or recent is None:
            return False, 1.0, {"error": "Insufficient data"}

        if len(baseline) < 100 or len(recent) < 100:
            return False, 1.0, {"error": "Insufficient samples"}

        # KS test
        baseline_array = np.array(baseline)
        recent_array = np.array(recent)
        ks_stat, p_value = stats.ks_2samp(baseline_array, recent_array)

        drift_detected = p_value < self.drift_threshold

        details = {
            "ks_statistic": ks_stat,
            "p_value": p_value,
            "baseline_mean": np.mean(baseline_array),
            "baseline_std": np.std(baseline_array),
            "recent_mean": np.mean(recent_array),
            "recent_std": np.std(recent_array),
        }

        if drift_detected:
            self._trigger_alert(DriftAlert(
                alert_type="data_drift",
                severity="warning" if p_value > 0.01 else "critical",
                metric_name=feature_name,
                current_value=details["recent_mean"],
                baseline_value=details["baseline_mean"],
                threshold=self.drift_threshold,
                details=f"KS test p-value: {p_value:.6f}",
            ))

        return drift_detected, p_value, details

    def check_all_data_drift(self) -> Dict[str, Tuple[bool, float]]:
        """Check drift for all tracked features."""
        results = {}
        for feature_name in self._baseline_features.keys():
            if feature_name in self._recent_features:
                drift, p_value, _ = self.check_data_drift(feature_name)
                results[feature_name] = (drift, p_value)
        return results

    def calculate_psi(
        self,
        feature_name: str,
        buckets: int = 10,
    ) -> Tuple[float, bool]:
        """
        Calculate Population Stability Index for a feature.

        PSI interpretation:
        - PSI < 0.1: No significant change
        - 0.1 <= PSI < 0.25: Moderate change
        - PSI >= 0.25: Significant change

        Returns:
            Tuple of (psi_value, significant_drift)
        """
        baseline = self._baseline_features.get(feature_name)
        recent = self._recent_features.get(feature_name)

        if baseline is None or recent is None or len(baseline) < 100 or len(recent) < 100:
            return 0.0, False

        baseline_array = np.array(baseline)
        recent_array = np.array(recent)

        # Create buckets based on baseline distribution
        _, bin_edges = np.histogram(baseline_array, bins=buckets)

        # Calculate frequencies
        baseline_counts, _ = np.histogram(baseline_array, bins=bin_edges)
        recent_counts, _ = np.histogram(recent_array, bins=bin_edges)

        # Normalize to proportions (add small epsilon to avoid log(0))
        eps = 1e-10
        baseline_props = (baseline_counts + eps) / (len(baseline_array) + buckets * eps)
        recent_props = (recent_counts + eps) / (len(recent_array) + buckets * eps)

        # Calculate PSI
        psi = np.sum((recent_props - baseline_props) * np.log(recent_props / baseline_props))

        significant_drift = psi >= 0.25

        if significant_drift:
            self._trigger_alert(DriftAlert(
                alert_type="data_drift",
                severity="critical" if psi >= 0.5 else "warning",
                metric_name=f"{feature_name}_psi",
                current_value=psi,
                baseline_value=0.0,
                threshold=0.25,
                details=f"PSI: {psi:.4f}",
            ))

        return psi, significant_drift

    def get_rolling_accuracy(
        self,
        window: Optional[int] = None,
    ) -> Tuple[float, int]:
        """
        Calculate rolling direction accuracy.

        Args:
            window: Window size (defaults to window_size)

        Returns:
            Tuple of (accuracy, sample_count)
        """
        window = window or self.window_size
        recent = list(self._predictions_log)[-window:]

        if len(recent) == 0:
            return 0.0, 0

        correct = sum(1 for m in recent if m.correct_direction)
        accuracy = correct / len(recent)

        return accuracy, len(recent)

    def get_rolling_mae(
        self,
        window: Optional[int] = None,
    ) -> Tuple[float, int]:
        """Calculate rolling Mean Absolute Error."""
        window = window or self.window_size
        recent = list(self._predictions_log)[-window:]

        if len(recent) == 0:
            return 0.0, 0

        mae = np.mean([m.error for m in recent])
        return mae, len(recent)

    def set_baseline_performance(
        self,
        accuracy: Optional[float] = None,
        mae: Optional[float] = None,
    ) -> None:
        """
        Set baseline performance metrics.

        If not provided, calculates from current history.
        """
        if accuracy is not None:
            self._baseline_accuracy = accuracy
        elif len(self._performance_history) > 100:
            correct = sum(1 for m in self._performance_history if m.correct_direction)
            self._baseline_accuracy = correct / len(self._performance_history)

        if mae is not None:
            self._baseline_mae = mae
        elif len(self._performance_history) > 100:
            self._baseline_mae = np.mean([m.error for m in self._performance_history])

        mae_str = f"{self._baseline_mae:.4f}" if self._baseline_mae is not None else "N/A"
        acc_str = f"{self._baseline_accuracy:.4f}" if self._baseline_accuracy is not None else "N/A"
        logger.info(f"Baseline performance set: accuracy={acc_str}, MAE={mae_str}")

    def check_performance_degradation(self) -> Tuple[bool, Dict]:
        """
        Check if model performance has degraded.

        Returns:
            Tuple of (degradation_detected, details)
        """
        if self._baseline_accuracy is None:
            return False, {"error": "Baseline not set"}

        current_accuracy, sample_count = self.get_rolling_accuracy()
        current_mae, _ = self.get_rolling_mae()

        if sample_count < 100:
            return False, {"error": "Insufficient samples"}

        accuracy_drop = self._baseline_accuracy - current_accuracy
        degradation_detected = accuracy_drop > self.performance_degradation_threshold

        details = {
            "baseline_accuracy": self._baseline_accuracy,
            "current_accuracy": current_accuracy,
            "accuracy_drop": accuracy_drop,
            "baseline_mae": self._baseline_mae,
            "current_mae": current_mae,
            "sample_count": sample_count,
        }

        if degradation_detected:
            self._trigger_alert(DriftAlert(
                alert_type="performance_degradation",
                severity="critical" if accuracy_drop > 0.2 else "warning",
                metric_name="direction_accuracy",
                current_value=current_accuracy,
                baseline_value=self._baseline_accuracy,
                threshold=self.performance_degradation_threshold,
                details=f"Accuracy dropped by {accuracy_drop:.2%}",
            ))

        return degradation_detected, details

    def get_alerts(
        self,
        since: Optional[datetime] = None,
        severity: Optional[str] = None,
    ) -> List[DriftAlert]:
        """
        Get recorded alerts.

        Args:
            since: Only return alerts after this time
            severity: Filter by severity

        Returns:
            List of DriftAlert
        """
        alerts = self._alerts

        if since:
            alerts = [a for a in alerts if a.timestamp >= since]

        if severity:
            alerts = [a for a in alerts if a.severity == severity]

        return alerts

    def clear_alerts(self) -> int:
        """Clear all alerts. Returns count of cleared alerts."""
        count = len(self._alerts)
        self._alerts = []
        return count

    def get_status_report(self) -> Dict:
        """Generate a status report with all metrics."""
        accuracy, acc_samples = self.get_rolling_accuracy()
        mae, mae_samples = self.get_rolling_mae()

        # Check drift on all features
        drift_results = self.check_all_data_drift()
        drifting_features = [f for f, (drift, _) in drift_results.items() if drift]

        # Check performance
        perf_degraded, perf_details = self.check_performance_degradation()

        return {
            "timestamp": datetime.utcnow().isoformat(),
            "performance": {
                "rolling_accuracy": accuracy,
                "rolling_mae": mae,
                "sample_count": acc_samples,
                "baseline_accuracy": self._baseline_accuracy,
                "degradation_detected": perf_degraded,
            },
            "data_drift": {
                "features_monitored": len(self._baseline_features),
                "drifting_features": drifting_features,
                "drift_results": {
                    f: {"drift": d, "p_value": p}
                    for f, (d, p) in drift_results.items()
                },
            },
            "alerts": {
                "total": len(self._alerts),
                "critical": len([a for a in self._alerts if a.severity == "critical"]),
                "warning": len([a for a in self._alerts if a.severity == "warning"]),
            },
        }


class PerformanceTracker:
    """
    Track model performance over time (US-023).

    Provides historical performance metrics for dashboards and reporting.
    """

    def __init__(self, max_history_days: int = 90):
        """
        Initialize performance tracker.

        Args:
            max_history_days: Maximum days of history to retain
        """
        self.max_history_days = max_history_days
        self._daily_metrics: Dict[str, Dict] = {}  # date_str -> metrics

    def record_daily_metrics(
        self,
        date: datetime,
        accuracy: float,
        mae: float,
        prediction_count: int,
        signal_distribution: Dict[str, int],
    ) -> None:
        """Record daily aggregated metrics."""
        date_str = date.strftime("%Y-%m-%d")

        self._daily_metrics[date_str] = {
            "date": date_str,
            "accuracy": accuracy,
            "mae": mae,
            "prediction_count": prediction_count,
            "signal_distribution": signal_distribution,
            "recorded_at": datetime.utcnow().isoformat(),
        }

        # Prune old data
        self._prune_old_data()

    def _prune_old_data(self) -> None:
        """Remove data older than max_history_days."""
        cutoff = datetime.utcnow() - timedelta(days=self.max_history_days)
        cutoff_str = cutoff.strftime("%Y-%m-%d")

        self._daily_metrics = {
            date: metrics
            for date, metrics in self._daily_metrics.items()
            if date >= cutoff_str
        }

    def get_daily_metrics(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
    ) -> List[Dict]:
        """Get daily metrics for date range."""
        metrics = list(self._daily_metrics.values())

        if start_date:
            start_str = start_date.strftime("%Y-%m-%d")
            metrics = [m for m in metrics if m["date"] >= start_str]

        if end_date:
            end_str = end_date.strftime("%Y-%m-%d")
            metrics = [m for m in metrics if m["date"] <= end_str]

        return sorted(metrics, key=lambda m: m["date"])

    def get_weekly_summary(self) -> Dict:
        """Get summary for the last 7 days."""
        week_ago = datetime.utcnow() - timedelta(days=7)
        recent = self.get_daily_metrics(start_date=week_ago)

        if not recent:
            return {"error": "No data available"}

        accuracies = [m["accuracy"] for m in recent]
        maes = [m["mae"] for m in recent]
        total_predictions = sum(m["prediction_count"] for m in recent)

        return {
            "period": "7 days",
            "days_with_data": len(recent),
            "avg_accuracy": np.mean(accuracies),
            "min_accuracy": np.min(accuracies),
            "max_accuracy": np.max(accuracies),
            "avg_mae": np.mean(maes),
            "total_predictions": total_predictions,
            "accuracy_trend": (
                "improving" if len(accuracies) > 1 and accuracies[-1] > accuracies[0]
                else "declining" if len(accuracies) > 1 and accuracies[-1] < accuracies[0]
                else "stable"
            ),
        }
