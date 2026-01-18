"""
Hyperparameter Tuning Module
============================
Optuna-based hyperparameter optimization for LSTM and XGBoost models.
"""

import json
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
from loguru import logger

try:
    import optuna
    from optuna.trial import Trial
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False
    logger.warning("optuna not installed. Install with: pip install optuna")


class HyperparameterTuner:
    """
    Hyperparameter tuner using Optuna for automatic optimization.

    Supports:
    - LSTM hyperparameters (hidden_size, layers, dropout, lr, etc.)
    - XGBoost hyperparameters (depth, learning_rate, n_estimators, etc.)
    - Sequence length optimization
    - Cross-validated objective functions
    """

    # Default search spaces
    LSTM_SEARCH_SPACE = {
        "hidden_size": {"type": "categorical", "choices": [64, 128, 256]},
        "num_layers": {"type": "int", "low": 1, "high": 4},
        "dropout": {"type": "float", "low": 0.1, "high": 0.5},
        "learning_rate": {"type": "loguniform", "low": 1e-5, "high": 1e-2},
        "batch_size": {"type": "categorical", "choices": [16, 32, 64]},
        "use_attention": {"type": "categorical", "choices": [True, False]},
        "use_layer_norm": {"type": "categorical", "choices": [True, False]},
        "weight_decay": {"type": "loguniform", "low": 1e-6, "high": 1e-3},
    }

    XGBOOST_SEARCH_SPACE = {
        "max_depth": {"type": "int", "low": 3, "high": 10},
        "learning_rate": {"type": "loguniform", "low": 0.01, "high": 0.3},
        "n_estimators": {"type": "int", "low": 50, "high": 500, "step": 50},
        "min_child_weight": {"type": "int", "low": 1, "high": 10},
        "subsample": {"type": "float", "low": 0.6, "high": 1.0},
        "colsample_bytree": {"type": "float", "low": 0.6, "high": 1.0},
        "reg_alpha": {"type": "loguniform", "low": 1e-8, "high": 10.0},
        "reg_lambda": {"type": "loguniform", "low": 1e-8, "high": 10.0},
    }

    SEQUENCE_LENGTH_SPACE = {
        "lookback": {"type": "categorical", "choices": [30, 60, 90, 120, 180]},
    }

    def __init__(
        self,
        study_name: str = "gold_predictor_tuning",
        direction: str = "minimize",
        storage: Optional[str] = None,
        load_if_exists: bool = True,
    ):
        """
        Initialize hyperparameter tuner.

        Args:
            study_name: Name of the Optuna study
            direction: 'minimize' for loss, 'maximize' for accuracy
            storage: Optional SQLite path for persistent storage
            load_if_exists: Load existing study if it exists
        """
        if not OPTUNA_AVAILABLE:
            raise ImportError("optuna not installed. Install with: pip install optuna")

        self.study_name = study_name
        self.direction = direction
        self.storage = storage
        self.load_if_exists = load_if_exists

        self.study: Optional[optuna.Study] = None
        self.best_params: Dict[str, Any] = {}
        self.best_value: Optional[float] = None

    def _sample_param(self, trial: Trial, name: str, config: Dict) -> Any:
        """Sample a hyperparameter based on its config."""
        param_type = config["type"]

        if param_type == "int":
            return trial.suggest_int(
                name,
                config["low"],
                config["high"],
                step=config.get("step", 1),
            )
        elif param_type == "float":
            return trial.suggest_float(
                name,
                config["low"],
                config["high"],
            )
        elif param_type == "loguniform":
            return trial.suggest_float(
                name,
                config["low"],
                config["high"],
                log=True,
            )
        elif param_type == "categorical":
            return trial.suggest_categorical(name, config["choices"])
        else:
            raise ValueError(f"Unknown parameter type: {param_type}")

    def _create_objective(
        self,
        train_fn: Callable,
        search_space: Dict[str, Dict],
        fixed_params: Optional[Dict] = None,
    ) -> Callable:
        """Create objective function for Optuna."""

        def objective(trial: Trial) -> float:
            # Sample hyperparameters
            params = {}
            for name, config in search_space.items():
                params[name] = self._sample_param(trial, name, config)

            # Add fixed parameters
            if fixed_params:
                params.update(fixed_params)

            # Train and evaluate
            try:
                score = train_fn(params, trial)
                return score
            except Exception as e:
                logger.error(f"Trial failed: {e}")
                # Return a bad score to continue optimization
                return float("inf") if self.direction == "minimize" else float("-inf")

        return objective

    def tune_lstm(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        n_trials: int = 50,
        timeout: Optional[int] = None,
        search_space: Optional[Dict] = None,
        fixed_params: Optional[Dict] = None,
        n_epochs: int = 50,
    ) -> Dict[str, Any]:
        """
        Tune LSTM hyperparameters.

        Args:
            X_train: Training features
            y_train: Training targets
            X_val: Validation features
            y_val: Validation targets
            n_trials: Number of optimization trials
            timeout: Optional timeout in seconds
            search_space: Custom search space (uses default if None)
            fixed_params: Parameters to keep fixed
            n_epochs: Number of epochs per trial

        Returns:
            Dict with best parameters and value
        """
        import torch
        from ..models.lstm_model import GoldLSTM, LSTMTrainer

        search_space = search_space or self.LSTM_SEARCH_SPACE

        def train_fn(params: Dict, trial: Trial) -> float:
            # Build model
            model = GoldLSTM(
                input_size=X_train.shape[2],
                hidden_size=params.get("hidden_size", 128),
                num_layers=params.get("num_layers", 2),
                dropout=params.get("dropout", 0.3),
                use_attention=params.get("use_attention", False),
                use_layer_norm=params.get("use_layer_norm", False),
            )

            # Build trainer
            trainer = LSTMTrainer(
                model,
                learning_rate=params.get("learning_rate", 0.001),
                weight_decay=params.get("weight_decay", 1e-4),
            )

            # Train
            history = trainer.train(
                X_train,
                y_train,
                X_val,
                y_val,
                epochs=n_epochs,
                batch_size=params.get("batch_size", 32),
                early_stopping=10,
            )

            # Report intermediate values for pruning
            for epoch, val_loss in enumerate(history["val_losses"]):
                trial.report(val_loss, epoch)
                if trial.should_prune():
                    raise optuna.TrialPruned()

            return history["best_val_loss"]

        return self._run_optimization(
            train_fn,
            search_space,
            n_trials,
            timeout,
            fixed_params,
        )

    def tune_xgboost(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        n_trials: int = 50,
        timeout: Optional[int] = None,
        search_space: Optional[Dict] = None,
        fixed_params: Optional[Dict] = None,
        objective: str = "multi:softprob",
    ) -> Dict[str, Any]:
        """
        Tune XGBoost hyperparameters.

        Args:
            X_train: Training features (flattened for XGBoost)
            y_train: Training targets (direction labels)
            X_val: Validation features
            y_val: Validation targets
            n_trials: Number of optimization trials
            timeout: Optional timeout in seconds
            search_space: Custom search space
            fixed_params: Parameters to keep fixed
            objective: XGBoost objective function

        Returns:
            Dict with best parameters and value
        """
        from xgboost import XGBClassifier
        from sklearn.metrics import accuracy_score

        search_space = search_space or self.XGBOOST_SEARCH_SPACE

        # Flatten sequences for XGBoost
        X_train_flat = X_train.reshape(X_train.shape[0], -1)
        X_val_flat = X_val.reshape(X_val.shape[0], -1)

        def train_fn(params: Dict, trial: Trial) -> float:
            model = XGBClassifier(
                max_depth=params.get("max_depth", 6),
                learning_rate=params.get("learning_rate", 0.1),
                n_estimators=params.get("n_estimators", 200),
                min_child_weight=params.get("min_child_weight", 1),
                subsample=params.get("subsample", 0.8),
                colsample_bytree=params.get("colsample_bytree", 0.8),
                reg_alpha=params.get("reg_alpha", 0),
                reg_lambda=params.get("reg_lambda", 1),
                objective=objective,
                use_label_encoder=False,
                eval_metric="mlogloss",
                random_state=42,
            )

            model.fit(
                X_train_flat,
                y_train,
                eval_set=[(X_val_flat, y_val)],
                verbose=False,
            )

            y_pred = model.predict(X_val_flat)
            accuracy = accuracy_score(y_val, y_pred)

            # Return negative accuracy for minimization, or accuracy for maximization
            return -accuracy if self.direction == "minimize" else accuracy

        return self._run_optimization(
            train_fn,
            search_space,
            n_trials,
            timeout,
            fixed_params,
        )

    def tune_sequence_length(
        self,
        data_fn: Callable[[int], Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]],
        train_fn: Callable[[np.ndarray, np.ndarray, np.ndarray, np.ndarray], float],
        n_trials: int = 10,
        lookback_values: List[int] = None,
    ) -> Dict[str, Any]:
        """
        Find optimal sequence length (lookback window).

        Args:
            data_fn: Function that takes lookback and returns (X_train, y_train, X_val, y_val)
            train_fn: Function that takes data and returns validation score
            n_trials: Number of trials per lookback value
            lookback_values: List of lookback values to test

        Returns:
            Dict with best lookback and corresponding score
        """
        lookback_values = lookback_values or [30, 60, 90, 120, 180]

        results = {}
        for lookback in lookback_values:
            logger.info(f"Testing lookback={lookback}")

            scores = []
            for trial in range(n_trials):
                X_train, y_train, X_val, y_val = data_fn(lookback)
                score = train_fn(X_train, y_train, X_val, y_val)
                scores.append(score)

            results[lookback] = {
                "mean": np.mean(scores),
                "std": np.std(scores),
                "min": np.min(scores),
                "max": np.max(scores),
                "scores": scores,
            }
            logger.info(f"  Lookback {lookback}: mean={results[lookback]['mean']:.4f} ± {results[lookback]['std']:.4f}")

        # Find best lookback
        if self.direction == "minimize":
            best_lookback = min(results.keys(), key=lambda k: results[k]["mean"])
        else:
            best_lookback = max(results.keys(), key=lambda k: results[k]["mean"])

        self.best_params = {"lookback": best_lookback}
        self.best_value = results[best_lookback]["mean"]

        return {
            "best_lookback": best_lookback,
            "best_score": results[best_lookback],
            "all_results": results,
        }

    def _run_optimization(
        self,
        train_fn: Callable,
        search_space: Dict,
        n_trials: int,
        timeout: Optional[int],
        fixed_params: Optional[Dict],
    ) -> Dict[str, Any]:
        """Run Optuna optimization."""
        # Create study
        self.study = optuna.create_study(
            study_name=self.study_name,
            direction=self.direction,
            storage=self.storage,
            load_if_exists=self.load_if_exists,
            pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=10),
        )

        # Create objective
        objective = self._create_objective(train_fn, search_space, fixed_params)

        # Optimize
        logger.info(f"Starting optimization with {n_trials} trials")
        self.study.optimize(
            objective,
            n_trials=n_trials,
            timeout=timeout,
            show_progress_bar=True,
        )

        # Store results
        self.best_params = self.study.best_params
        self.best_value = self.study.best_value

        logger.info(f"Best value: {self.best_value}")
        logger.info(f"Best params: {self.best_params}")

        return {
            "best_params": self.best_params,
            "best_value": self.best_value,
            "n_trials": len(self.study.trials),
            "study": self.study,
        }

    def save_results(self, path: str) -> None:
        """Save tuning results to JSON file."""
        results = {
            "best_params": self.best_params,
            "best_value": self.best_value,
            "study_name": self.study_name,
            "direction": self.direction,
        }

        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(results, f, indent=2)

        logger.info(f"Saved tuning results to {path}")

    def load_results(self, path: str) -> Dict[str, Any]:
        """Load tuning results from JSON file."""
        with open(path, "r") as f:
            results = json.load(f)

        self.best_params = results["best_params"]
        self.best_value = results["best_value"]

        logger.info(f"Loaded tuning results from {path}")
        return results

    def get_importance(self) -> Optional[Dict[str, float]]:
        """Get hyperparameter importance scores."""
        if self.study is None:
            return None

        try:
            importance = optuna.importance.get_param_importances(self.study)
            return importance
        except Exception as e:
            logger.warning(f"Could not calculate importance: {e}")
            return None

    def plot_optimization_history(self, save_path: Optional[str] = None):
        """Plot optimization history."""
        if self.study is None:
            logger.warning("No study available for plotting")
            return

        try:
            fig = optuna.visualization.plot_optimization_history(self.study)
            if save_path:
                fig.write_image(save_path)
            return fig
        except Exception as e:
            logger.warning(f"Could not create plot: {e}")
            return None

    def plot_param_importances(self, save_path: Optional[str] = None):
        """Plot parameter importances."""
        if self.study is None:
            logger.warning("No study available for plotting")
            return

        try:
            fig = optuna.visualization.plot_param_importances(self.study)
            if save_path:
                fig.write_image(save_path)
            return fig
        except Exception as e:
            logger.warning(f"Could not create plot: {e}")
            return None
