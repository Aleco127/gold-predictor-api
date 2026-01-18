# PRD: Gold Predictor ML Improvements

## Overview
Comprehensive machine learning improvements for the Gold Predictor system to enhance model generalization, prevent overfitting, and leverage more historical data effectively. This PRD addresses critical issues in data preprocessing, model architecture, regularization, and validation strategies.

## Problem Statement
The current Gold Predictor system has several ML issues that limit its predictive accuracy and generalization:
- Data leakage in preprocessing (scaler fitted on all data before split)
- Limited training data (30 days) with no walk-forward validation
- Insufficient regularization leading to overfitting
- Feature redundancy with 43 correlated indicators
- No hyperparameter tuning
- Misaligned loss functions between training and evaluation

## Target App
Gold Predictor (Python ML System)

## Success Metrics
- [ ] Validation accuracy improves from 67% to 72%+ on XGBoost
- [ ] LSTM validation loss reduces by 15%+ with more data
- [ ] Models generalize across different market regimes (bull/bear/sideways)
- [ ] Walk-forward backtest shows consistent performance (low variance across folds)
- [ ] Feature set reduced to 15-20 informative features without accuracy loss
- [ ] No data leakage in preprocessing pipeline

---

## User Stories

### Phase 1: Data Pipeline Fixes (Critical - Fix Overfitting)

---

### US-001: Fix Data Leakage in Scaler Fitting
**Description**: As a ML engineer, I want the scaler to be fitted only on training data so that validation/test sets don't influence training statistics.

**Acceptance Criteria**:
- [ ] `DataProcessor.fit()` method only uses training data
- [ ] `DataProcessor.transform()` applies fitted scaler to any data
- [ ] Separate `fit_transform()` deprecated or removed
- [ ] Unit test verifies no future data leakage
- [ ] Typecheck passes

**Files to modify**:
- `src/preprocessing/data_processor.py`
- `src/models/lstm_model.py`
- `src/models/xgboost_model.py`

**Dependencies**: None

---

### US-002: Implement Walk-Forward Validation
**Description**: As a ML engineer, I want walk-forward validation so that models are tested on truly out-of-sample data with rolling windows.

**Acceptance Criteria**:
- [ ] `WalkForwardValidator` class created with configurable window sizes
- [ ] Supports expanding and sliding window modes
- [ ] Returns metrics for each fold (mean, std, min, max)
- [ ] Integrates with both LSTM and XGBoost training
- [ ] Typecheck passes

**Files to modify**:
- `src/preprocessing/data_processor.py` (new class)
- `src/models/lstm_model.py`
- `src/models/xgboost_model.py`

**Dependencies**: US-001

---

### US-003: Implement TimeSeriesSplit Cross-Validation
**Description**: As a ML engineer, I want proper time series cross-validation so that temporal ordering is preserved during model selection.

**Acceptance Criteria**:
- [ ] `TimeSeriesCrossValidator` wraps sklearn's TimeSeriesSplit
- [ ] Configurable gap parameter to prevent leakage
- [ ] Returns train/val indices for k folds
- [ ] Reports cross-validated metrics with confidence intervals
- [ ] Typecheck passes

**Files to modify**:
- `src/preprocessing/data_processor.py`
- `main.py` (training function)

**Dependencies**: US-001

---

### Phase 2: Feature Engineering Improvements

---

### US-004: Feature Correlation Analysis and Pruning
**Description**: As a ML engineer, I want to remove highly correlated features so that the model doesn't learn redundant information.

**Acceptance Criteria**:
- [ ] `FeatureSelector` class computes correlation matrix
- [ ] Automatically removes features with |r| > threshold (default 0.85)
- [ ] Keeps feature with highest variance when removing correlated pairs
- [ ] Logs which features were removed and why
- [ ] Reduces feature set from 43 to ~20-25
- [ ] Typecheck passes

**Files to modify**:
- `src/features/feature_selector.py` (new file)
- `src/features/technical_indicators.py`
- `src/preprocessing/data_processor.py`

**Dependencies**: None

---

### US-005: Feature Importance Analysis
**Description**: As a ML engineer, I want to analyze feature importance so that I can identify the most predictive indicators.

**Acceptance Criteria**:
- [ ] Extract XGBoost feature importance after training
- [ ] Implement permutation importance for LSTM
- [ ] Generate importance report (top 20 features)
- [ ] Option to retrain with only top-N features
- [ ] Typecheck passes

**Files to modify**:
- `src/features/feature_selector.py`
- `src/models/xgboost_model.py`
- `src/models/ensemble.py`

**Dependencies**: US-004

---

### US-006: Add Temporal Features
**Description**: As a ML engineer, I want temporal features (hour, day of week, session) so that the model can capture time-based patterns.

**Acceptance Criteria**:
- [ ] Hour of day encoded (cyclical: sin/cos)
- [ ] Day of week encoded (cyclical: sin/cos)
- [ ] Trading session indicator (Asia/Europe/US)
- [ ] Time since market open feature
- [ ] Features added to TechnicalIndicators class
- [ ] Typecheck passes

**Files to modify**:
- `src/features/technical_indicators.py`

**Dependencies**: None

---

### US-007: Add Volatility Regime Detection
**Description**: As a ML engineer, I want volatility regime features so that the model can adapt to different market conditions.

**Acceptance Criteria**:
- [ ] Volatility regime classifier (low/medium/high)
- [ ] Based on rolling ATR percentile (20-day window)
- [ ] VIX-style proxy using gold options implied vol (if available)
- [ ] Regime persistence indicator
- [ ] Typecheck passes

**Files to modify**:
- `src/features/technical_indicators.py`

**Dependencies**: None

---

### Phase 3: Model Architecture Improvements

---

### US-008: Increase LSTM Regularization
**Description**: As a ML engineer, I want stronger regularization in LSTM so that it generalizes better with more data.

**Acceptance Criteria**:
- [ ] Dropout increased to configurable value (default 0.35)
- [ ] L2 weight decay added to optimizer (default 1e-4)
- [ ] Recurrent dropout option added
- [ ] Layer normalization option added
- [ ] All regularization params configurable via config
- [ ] Typecheck passes

**Files to modify**:
- `src/models/lstm_model.py`
- `src/config.py`

**Dependencies**: None

---

### US-009: Add Batch Normalization to LSTM
**Description**: As a ML engineer, I want batch normalization between LSTM layers so that training is more stable.

**Acceptance Criteria**:
- [ ] Optional BatchNorm1d after LSTM output
- [ ] LayerNorm alternative option
- [ ] Configurable via model parameters
- [ ] Training stability metrics logged
- [ ] Typecheck passes

**Files to modify**:
- `src/models/lstm_model.py`

**Dependencies**: US-008

---

### US-010: Implement Attention Mechanism
**Description**: As a ML engineer, I want an attention mechanism so that the model can focus on important timesteps.

**Acceptance Criteria**:
- [ ] Self-attention layer after LSTM
- [ ] Attention weights extractable for visualization
- [ ] Optional multi-head attention
- [ ] Configurable number of attention heads
- [ ] Typecheck passes

**Files to modify**:
- `src/models/lstm_model.py`
- `src/models/attention.py` (new file)

**Dependencies**: US-009

---

### US-011: Add Residual Connections
**Description**: As a ML engineer, I want residual connections so that deeper networks can be trained effectively.

**Acceptance Criteria**:
- [ ] Skip connections between LSTM layers
- [ ] Dense connections option (DenseNet style)
- [ ] Works with variable number of layers
- [ ] Typecheck passes

**Files to modify**:
- `src/models/lstm_model.py`

**Dependencies**: US-010

---

### Phase 4: Training Process Improvements

---

### US-012: Implement Learning Rate Warmup
**Description**: As a ML engineer, I want learning rate warmup so that training is more stable in early epochs.

**Acceptance Criteria**:
- [ ] Linear warmup over N steps (configurable)
- [ ] Integrates with existing LR scheduler
- [ ] Warmup steps logged
- [ ] Default: 100 steps warmup
- [ ] Typecheck passes

**Files to modify**:
- `src/models/lstm_model.py`

**Dependencies**: None

---

### US-013: Add Alternative Loss Functions
**Description**: As a ML engineer, I want alternative loss functions so that I can choose the best objective for the task.

**Acceptance Criteria**:
- [ ] Huber loss option (robust to outliers)
- [ ] Quantile loss option (for prediction intervals)
- [ ] Direction-weighted MSE (penalize wrong direction more)
- [ ] Configurable via training params
- [ ] Typecheck passes

**Files to modify**:
- `src/models/lstm_model.py`
- `src/models/losses.py` (new file)

**Dependencies**: None

---

### US-014: Implement Gradient Accumulation
**Description**: As a ML engineer, I want gradient accumulation so that I can effectively use larger batch sizes on limited memory.

**Acceptance Criteria**:
- [ ] Accumulation steps configurable
- [ ] Effective batch size = batch_size * accumulation_steps
- [ ] Gradient normalization after accumulation
- [ ] Memory usage stays constant
- [ ] Typecheck passes

**Files to modify**:
- `src/models/lstm_model.py`

**Dependencies**: None

---

### Phase 5: Hyperparameter Optimization

---

### US-015: Implement Optuna Integration
**Description**: As a ML engineer, I want Optuna hyperparameter tuning so that model parameters are optimized automatically.

**Acceptance Criteria**:
- [ ] `HyperparameterTuner` class wraps Optuna
- [ ] Defines search space for LSTM (hidden_size, layers, dropout, lr)
- [ ] Defines search space for XGBoost (depth, lr, estimators, reg)
- [ ] Uses cross-validation for objective
- [ ] Saves best parameters to config
- [ ] Typecheck passes

**Files to modify**:
- `src/tuning/hyperparameter_tuner.py` (new file)
- `requirements.txt` (add optuna)
- `main.py` (add tune command)

**Dependencies**: US-003

---

### US-016: Sequence Length Optimization
**Description**: As a ML engineer, I want to find the optimal lookback window so that the model uses the right amount of history.

**Acceptance Criteria**:
- [ ] Test lookback values: 30, 60, 90, 120, 180
- [ ] Compare validation metrics for each
- [ ] Report optimal lookback with confidence interval
- [ ] Update config with best value
- [ ] Typecheck passes

**Files to modify**:
- `src/tuning/hyperparameter_tuner.py`
- `src/config.py`

**Dependencies**: US-015

---

### Phase 6: Ensemble Improvements

---

### US-017: Implement Stacking Ensemble
**Description**: As a ML engineer, I want a stacking ensemble so that model predictions are combined optimally.

**Acceptance Criteria**:
- [ ] Meta-model trained on base model predictions
- [ ] Uses validation set predictions only (no leakage)
- [ ] Meta-model options: LogisticRegression, XGBoost, Neural Net
- [ ] Outperforms simple weighted average
- [ ] Typecheck passes

**Files to modify**:
- `src/models/ensemble.py`
- `src/models/stacking.py` (new file)

**Dependencies**: US-003

---

### US-018: Dynamic Weight Calibration
**Description**: As a ML engineer, I want dynamic ensemble weights so that model contributions adapt to market conditions.

**Acceptance Criteria**:
- [ ] Weights based on recent model performance (rolling window)
- [ ] Higher weight to model with better recent accuracy
- [ ] Minimum weight floor (e.g., 0.2) to prevent collapse
- [ ] Calibration logged for monitoring
- [ ] Typecheck passes

**Files to modify**:
- `src/models/ensemble.py`

**Dependencies**: US-017

---

### Phase 7: Data Augmentation

---

### US-019: Implement Time Series Augmentation
**Description**: As a ML engineer, I want data augmentation so that the model sees more varied training examples.

**Acceptance Criteria**:
- [ ] Jittering: add small noise (±0.5% of value)
- [ ] Scaling: multiply by factor [0.95, 1.05]
- [ ] Window slicing: random subsequence selection
- [ ] Augmentation probability configurable
- [ ] Applied only during training
- [ ] Typecheck passes

**Files to modify**:
- `src/preprocessing/augmentation.py` (new file)
- `src/models/lstm_model.py`

**Dependencies**: None

---

### US-020: Implement SMOTE for Direction Classes
**Description**: As a ML engineer, I want SMOTE oversampling so that minority direction classes are better represented.

**Acceptance Criteria**:
- [ ] SMOTE applied to minority class (down/up when imbalanced)
- [ ] Uses imbalanced-learn library
- [ ] Only applied to XGBoost training data
- [ ] Class balance logged before/after
- [ ] Typecheck passes

**Files to modify**:
- `src/preprocessing/augmentation.py`
- `src/models/xgboost_model.py`
- `requirements.txt` (add imbalanced-learn)

**Dependencies**: US-019

---

### Phase 8: Extended Data Support

---

### US-021: Support 1-Year Historical Data
**Description**: As a ML engineer, I want to train on 1 year of data so that the model learns long-term patterns.

**Acceptance Criteria**:
- [ ] Capital.com connector fetches up to 1 year (paginated)
- [ ] Data stored locally to avoid repeated API calls
- [ ] Incremental update support (fetch only new data)
- [ ] Memory-efficient processing (chunked loading)
- [ ] Typecheck passes

**Files to modify**:
- `src/data/capital_connector.py`
- `src/data/data_cache.py` (new file)

**Dependencies**: None

---

### US-022: Multi-Timeframe Features
**Description**: As a ML engineer, I want multi-timeframe features so that the model sees patterns at different scales.

**Acceptance Criteria**:
- [ ] Aggregate M5 to M15, H1, H4 timeframes
- [ ] Calculate indicators on each timeframe
- [ ] Align features to M5 timestamps
- [ ] Feature naming: {indicator}_{timeframe}
- [ ] Typecheck passes

**Files to modify**:
- `src/features/technical_indicators.py`
- `src/features/multi_timeframe.py` (new file)

**Dependencies**: US-021

---

### Phase 9: Monitoring and Robustness

---

### US-023: Implement Model Monitoring
**Description**: As a ML engineer, I want model monitoring so that I can detect performance degradation.

**Acceptance Criteria**:
- [ ] Track prediction accuracy over time (rolling 100 predictions)
- [ ] Alert when accuracy drops below threshold
- [ ] Log prediction vs actual for analysis
- [ ] Concept drift detection (ADWIN algorithm)
- [ ] Typecheck passes

**Files to modify**:
- `src/monitoring/model_monitor.py` (new file)
- `src/api/server.py`

**Dependencies**: None

---

### US-024: Stress Testing on Historical Crashes
**Description**: As a ML engineer, I want stress testing so that I know how the model performs in extreme conditions.

**Acceptance Criteria**:
- [ ] Test on 2020 COVID crash period
- [ ] Test on 2022 rate hike volatility
- [ ] Report max drawdown during stress periods
- [ ] Flag if model fails stress tests
- [ ] Typecheck passes

**Files to modify**:
- `src/backtesting/stress_test.py` (new file)
- `main.py` (add stress-test command)

**Dependencies**: US-021

---

## Technical Notes
- Use Optuna for hyperparameter optimization (lightweight, works well with PyTorch)
- Consider imbalanced-learn for SMOTE implementation
- Multi-timeframe features will significantly increase feature count - apply selection after
- Walk-forward validation is compute-intensive - consider caching intermediate results
- Attention mechanism adds ~20% to training time but significantly improves interpretability

## Out of Scope
- Transformer architecture replacement (future PRD)
- Real-time online learning
- Alternative data sources (news sentiment, order flow)
- Multi-asset portfolio optimization
- Production deployment optimizations

## Open Questions
- [ ] Should we implement a full Transformer model or just attention on LSTM?
- [ ] What is the optimal lookback for 1-year data? (needs experimentation)
- [ ] Should meta-model in stacking be simple (LR) or complex (NN)?
