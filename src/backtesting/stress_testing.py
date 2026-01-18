"""
Stress Testing Module (US-024)
==============================
Test model performance on historical market stress periods.
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from loguru import logger


@dataclass
class StressPeriod:
    """Definition of a historical stress period."""
    name: str
    start_date: datetime
    end_date: datetime
    description: str
    expected_behavior: str  # "high_volatility", "trending", "reversal"
    severity: str  # "moderate", "severe", "extreme"


@dataclass
class StressTestResult:
    """Results from a stress test period."""
    period: StressPeriod
    accuracy: float
    mae: float
    direction_accuracy: float
    max_drawdown: float
    total_predictions: int
    correct_predictions: int
    avg_confidence: float
    volatility_during: float
    passed: bool
    details: Dict


# Predefined historical stress periods for gold
GOLD_STRESS_PERIODS = [
    StressPeriod(
        name="COVID-19 Crash",
        start_date=datetime(2020, 2, 24),
        end_date=datetime(2020, 3, 23),
        description="COVID-19 market crash - gold initially dropped then surged",
        expected_behavior="high_volatility",
        severity="extreme",
    ),
    StressPeriod(
        name="COVID-19 Recovery",
        start_date=datetime(2020, 3, 23),
        end_date=datetime(2020, 8, 7),
        description="Gold rally to all-time highs during COVID recovery",
        expected_behavior="trending",
        severity="moderate",
    ),
    StressPeriod(
        name="2022 Rate Hike Start",
        start_date=datetime(2022, 3, 1),
        end_date=datetime(2022, 5, 16),
        description="Fed starts aggressive rate hikes - gold volatility",
        expected_behavior="high_volatility",
        severity="severe",
    ),
    StressPeriod(
        name="2022 Dollar Strength",
        start_date=datetime(2022, 9, 1),
        end_date=datetime(2022, 11, 3),
        description="Strong USD pressures gold to 2.5-year lows",
        expected_behavior="trending",
        severity="severe",
    ),
    StressPeriod(
        name="2023 Banking Crisis",
        start_date=datetime(2023, 3, 8),
        end_date=datetime(2023, 3, 24),
        description="SVB collapse - gold safe haven rally",
        expected_behavior="reversal",
        severity="severe",
    ),
    StressPeriod(
        name="Flash Crash Example",
        start_date=datetime(2021, 8, 6),
        end_date=datetime(2021, 8, 10),
        description="Gold flash crash - $100 drop in minutes",
        expected_behavior="high_volatility",
        severity="extreme",
    ),
]


class StressTester:
    """
    Run stress tests on model using historical crisis periods (US-024).

    Tests model performance on:
    - COVID-19 crash (March 2020)
    - 2022 rate hike volatility
    - Flash crash events
    - Other historical stress periods
    """

    def __init__(
        self,
        accuracy_threshold: float = 0.45,  # Lower threshold for stress periods
        max_drawdown_threshold: float = 0.30,
        min_samples: int = 50,
    ):
        """
        Initialize stress tester.

        Args:
            accuracy_threshold: Minimum acceptable accuracy during stress
            max_drawdown_threshold: Maximum acceptable drawdown
            min_samples: Minimum samples required for valid test
        """
        self.accuracy_threshold = accuracy_threshold
        self.max_drawdown_threshold = max_drawdown_threshold
        self.min_samples = min_samples
        self._results: List[StressTestResult] = []

    def run_stress_test(
        self,
        model_predict_fn: Callable[[pd.DataFrame], Tuple[np.ndarray, np.ndarray]],
        data: pd.DataFrame,
        period: StressPeriod,
        target_col: str = "close",
    ) -> StressTestResult:
        """
        Run stress test for a specific period.

        Args:
            model_predict_fn: Function that takes features and returns (predictions, confidences)
            data: DataFrame with datetime index containing the stress period
            period: StressPeriod to test
            target_col: Column to use as target

        Returns:
            StressTestResult with performance metrics
        """
        logger.info(f"Running stress test: {period.name}")

        # Filter data to stress period
        period_data = data[
            (data.index >= period.start_date) &
            (data.index <= period.end_date)
        ]

        if len(period_data) < self.min_samples:
            logger.warning(
                f"Insufficient data for {period.name}: "
                f"{len(period_data)} samples (need {self.min_samples})"
            )
            return StressTestResult(
                period=period,
                accuracy=0.0,
                mae=float("inf"),
                direction_accuracy=0.0,
                max_drawdown=1.0,
                total_predictions=len(period_data),
                correct_predictions=0,
                avg_confidence=0.0,
                volatility_during=0.0,
                passed=False,
                details={"error": "Insufficient data"},
            )

        # Get predictions
        try:
            predictions, confidences = model_predict_fn(period_data)
        except Exception as e:
            logger.error(f"Prediction failed for {period.name}: {e}")
            return StressTestResult(
                period=period,
                accuracy=0.0,
                mae=float("inf"),
                direction_accuracy=0.0,
                max_drawdown=1.0,
                total_predictions=len(period_data),
                correct_predictions=0,
                avg_confidence=0.0,
                volatility_during=0.0,
                passed=False,
                details={"error": str(e)},
            )

        # Calculate actual returns/changes
        actuals = period_data[target_col].values[1:]  # Shift by 1 for next-period prediction
        predictions = predictions[:-1] if len(predictions) > len(actuals) else predictions

        # Ensure arrays match
        min_len = min(len(actuals), len(predictions))
        actuals = actuals[:min_len]
        predictions = predictions[:min_len]
        confidences = confidences[:min_len] if len(confidences) >= min_len else confidences

        # Calculate metrics
        mae = np.mean(np.abs(actuals - predictions))

        # Direction accuracy
        actual_direction = np.sign(np.diff(period_data[target_col].values[:min_len + 1]))
        pred_direction = np.sign(predictions - period_data[target_col].values[:min_len])

        correct_direction = np.sum(actual_direction == pred_direction)
        direction_accuracy = correct_direction / len(actual_direction) if len(actual_direction) > 0 else 0

        # Calculate simulated returns based on predictions
        simulated_returns = actual_direction * pred_direction * np.abs(actual_direction)
        cumulative_returns = np.cumsum(simulated_returns)

        # Max drawdown
        peak = np.maximum.accumulate(cumulative_returns)
        drawdowns = (peak - cumulative_returns) / (np.abs(peak) + 1e-10)
        max_drawdown = np.max(drawdowns) if len(drawdowns) > 0 else 0

        # Volatility during period
        volatility = np.std(np.diff(period_data[target_col].values)) / np.mean(period_data[target_col].values)

        # Determine if passed
        passed = (
            direction_accuracy >= self.accuracy_threshold and
            max_drawdown <= self.max_drawdown_threshold
        )

        result = StressTestResult(
            period=period,
            accuracy=direction_accuracy,
            mae=mae,
            direction_accuracy=direction_accuracy,
            max_drawdown=max_drawdown,
            total_predictions=min_len,
            correct_predictions=int(correct_direction),
            avg_confidence=np.mean(confidences) if len(confidences) > 0 else 0,
            volatility_during=volatility,
            passed=passed,
            details={
                "period_start": period.start_date.isoformat(),
                "period_end": period.end_date.isoformat(),
                "data_points": len(period_data),
                "severity": period.severity,
                "expected_behavior": period.expected_behavior,
            },
        )

        self._results.append(result)

        logger.info(
            f"Stress test {period.name}: "
            f"accuracy={direction_accuracy:.2%}, "
            f"max_drawdown={max_drawdown:.2%}, "
            f"passed={passed}"
        )

        return result

    def run_all_stress_tests(
        self,
        model_predict_fn: Callable[[pd.DataFrame], Tuple[np.ndarray, np.ndarray]],
        data: pd.DataFrame,
        periods: Optional[List[StressPeriod]] = None,
        target_col: str = "close",
    ) -> List[StressTestResult]:
        """
        Run stress tests for all predefined periods.

        Args:
            model_predict_fn: Prediction function
            data: Complete historical data
            periods: List of periods to test (uses GOLD_STRESS_PERIODS if None)
            target_col: Target column name

        Returns:
            List of StressTestResult
        """
        periods = periods or GOLD_STRESS_PERIODS
        results = []

        for period in periods:
            result = self.run_stress_test(
                model_predict_fn=model_predict_fn,
                data=data,
                period=period,
                target_col=target_col,
            )
            results.append(result)

        return results

    def get_summary_report(self) -> Dict:
        """Generate summary report of all stress tests."""
        if not self._results:
            return {"error": "No stress tests run yet"}

        passed_count = sum(1 for r in self._results if r.passed)
        total_count = len(self._results)

        by_severity = {}
        for severity in ["moderate", "severe", "extreme"]:
            severity_results = [r for r in self._results if r.period.severity == severity]
            if severity_results:
                by_severity[severity] = {
                    "count": len(severity_results),
                    "passed": sum(1 for r in severity_results if r.passed),
                    "avg_accuracy": np.mean([r.direction_accuracy for r in severity_results]),
                    "avg_drawdown": np.mean([r.max_drawdown for r in severity_results]),
                }

        return {
            "total_tests": total_count,
            "passed": passed_count,
            "failed": total_count - passed_count,
            "pass_rate": passed_count / total_count if total_count > 0 else 0,
            "by_severity": by_severity,
            "all_results": [
                {
                    "period": r.period.name,
                    "accuracy": r.direction_accuracy,
                    "max_drawdown": r.max_drawdown,
                    "passed": r.passed,
                    "severity": r.period.severity,
                }
                for r in self._results
            ],
            "overall_assessment": (
                "ROBUST" if passed_count / total_count >= 0.8 else
                "ACCEPTABLE" if passed_count / total_count >= 0.6 else
                "NEEDS_IMPROVEMENT" if passed_count / total_count >= 0.4 else
                "POOR"
            ),
        }

    def clear_results(self) -> None:
        """Clear all stored results."""
        self._results = []


def create_synthetic_stress_data(
    base_data: pd.DataFrame,
    stress_type: str = "high_volatility",
    volatility_multiplier: float = 3.0,
) -> pd.DataFrame:
    """
    Create synthetic stress data for testing.

    Useful when historical data for specific periods is unavailable.

    Args:
        base_data: Base DataFrame to modify
        stress_type: Type of stress to simulate
        volatility_multiplier: How much to increase volatility

    Returns:
        Modified DataFrame simulating stress conditions
    """
    df = base_data.copy()

    if stress_type == "high_volatility":
        # Increase volatility by amplifying returns
        for col in ["open", "high", "low", "close"]:
            if col in df.columns:
                returns = df[col].pct_change()
                amplified_returns = returns * volatility_multiplier
                df[col] = df[col].iloc[0] * (1 + amplified_returns).cumprod()

    elif stress_type == "trending":
        # Add a strong trend
        trend = np.linspace(0, 0.1, len(df))  # 10% trend
        for col in ["open", "high", "low", "close"]:
            if col in df.columns:
                df[col] = df[col] * (1 + trend)

    elif stress_type == "reversal":
        # Create sudden reversal
        mid_point = len(df) // 2
        for col in ["open", "high", "low", "close"]:
            if col in df.columns:
                # First half up, second half down
                trend_up = np.linspace(0, 0.05, mid_point)
                trend_down = np.linspace(0.05, -0.02, len(df) - mid_point)
                trend = np.concatenate([trend_up, trend_down])
                df[col] = df[col] * (1 + trend)

    elif stress_type == "flash_crash":
        # Simulate sudden drop and recovery
        crash_point = len(df) // 3
        recovery_end = 2 * len(df) // 3

        for col in ["open", "high", "low", "close"]:
            if col in df.columns:
                values = df[col].values.copy()
                # Sudden 5% drop
                values[crash_point:recovery_end] *= 0.95
                # Partial recovery
                recovery = np.linspace(0.95, 0.98, recovery_end - crash_point)
                values[crash_point:recovery_end] = values[crash_point:recovery_end] / 0.95 * recovery
                df[col] = values

    logger.info(f"Created synthetic {stress_type} stress data: {len(df)} rows")
    return df


def run_model_stress_validation(
    model: Any,
    data: pd.DataFrame,
    feature_columns: List[str],
    target_col: str = "close",
    min_pass_rate: float = 0.6,
) -> Tuple[bool, Dict]:
    """
    Convenience function to run complete stress validation on a model.

    Args:
        model: Model with predict method
        data: Historical data
        feature_columns: List of feature column names
        target_col: Target column name
        min_pass_rate: Minimum pass rate required

    Returns:
        Tuple of (passed, report)
    """
    def predict_fn(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        features = df[feature_columns].values
        predictions = model.predict(features)
        # Assume model has confidence output or use default
        confidences = getattr(model, "predict_confidence", lambda x: np.ones(len(x)))(features)
        return predictions, confidences

    tester = StressTester()
    results = tester.run_all_stress_tests(
        model_predict_fn=predict_fn,
        data=data,
        target_col=target_col,
    )

    report = tester.get_summary_report()
    passed = report["pass_rate"] >= min_pass_rate

    return passed, report
