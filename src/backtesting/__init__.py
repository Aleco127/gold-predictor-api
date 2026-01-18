"""Backtesting module for strategy validation."""

from .stress_testing import (
    StressTester,
    StressPeriod,
    StressTestResult,
    GOLD_STRESS_PERIODS,
    create_synthetic_stress_data,
    run_model_stress_validation,
)
from .backtest_engine import (
    BacktestEngine,
    BacktestResult,
    BacktestMetrics,
    BacktestTrade,
)

__all__ = [
    # Stress testing
    "StressTester",
    "StressPeriod",
    "StressTestResult",
    "GOLD_STRESS_PERIODS",
    "create_synthetic_stress_data",
    "run_model_stress_validation",
    # Backtesting
    "BacktestEngine",
    "BacktestResult",
    "BacktestMetrics",
    "BacktestTrade",
]
