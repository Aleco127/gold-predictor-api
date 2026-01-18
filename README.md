# Gold Price Predictor

ML-based gold (XAUUSD) price prediction system with n8n integration for automated alerts.

## Features

- **LSTM + XGBoost Ensemble**: Combines deep learning price prediction with gradient boosting direction classification
- **MetaTrader 5 Integration**: Real-time data from MT5 demo/live accounts
- **Technical Indicators**: RSI, MACD, Bollinger Bands, ATR, and more via pandas-ta
- **FastAPI Server**: RESTful API with authentication and health checks
- **n8n Workflow**: Ready-to-use workflow for Telegram alerts every 5 minutes
- **Docker Support**: Production-ready containerization

## Quick Start

### 1. Setup Environment

```bash
# Clone and enter directory
cd gold-predictor

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Configure Environment

```bash
# Copy example config
cp .env.example .env

# Edit with your credentials
# - MT5 demo account (IC Markets, XM, etc.)
# - API key for authentication
# - Telegram bot token (optional, for n8n alerts)
```

### 3. Train Models

```bash
# Train with 30 days of data
python main.py train --days 30 --epochs 100
```

### 4. Run Server

```bash
# Start API server
python main.py serve

# API available at http://localhost:8000
```

### 5. Test Prediction

```bash
curl -X POST http://localhost:8000/predict \
  -H "X-API-Key: your-api-key" \
  -H "Content-Type: application/json"
```

## Docker Deployment

```bash
# Build and run
docker-compose up --build

# With monitoring (Prometheus + Grafana)
docker-compose --profile monitoring up
```

## n8n Integration

1. Import `n8n-workflows/gold-prediction-alert.json` to n8n
2. Configure environment variables:
   - `GOLD_PREDICTOR_URL`: API URL (e.g., `http://gold-predictor:8000`)
   - `GOLD_PREDICTOR_API_KEY`: Your API key
   - `TELEGRAM_CHAT_ID`: Your Telegram chat ID
3. Set up Telegram credentials in n8n
4. Activate the workflow

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check |
| `/predict` | POST | Get prediction (requires API key) |
| `/price` | GET | Current price (requires API key) |
| `/train` | POST | Train models (requires API key) |
| `/metrics` | GET | Prometheus metrics |

## Project Structure

```
gold-predictor/
├── src/
│   ├── api/              # FastAPI server
│   ├── data/             # MT5 connector
│   ├── features/         # Technical indicators
│   ├── models/           # LSTM, XGBoost, Ensemble
│   ├── preprocessing/    # Data processor
│   ├── scheduler/        # Prediction scheduler
│   ├── monitoring/       # Prometheus metrics
│   └── storage/          # Prediction storage
├── models/               # Saved ML models
├── n8n-workflows/        # n8n workflow JSONs
├── tests/                # Test suite
├── monitoring/           # Prometheus config
├── main.py               # Entry point
├── requirements.txt      # Python dependencies
├── Dockerfile            # Docker image
└── docker-compose.yml    # Docker orchestration
```

## Commands

```bash
# Train models
python main.py train --days 30 --epochs 100

# Run API server
python main.py serve

# Run backtesting
python main.py backtest --days 7

# Run scheduler
python main.py schedule --interval 5

# Run tests
pytest tests/
```

## Signal Types

| Signal | Description | Confidence |
|--------|-------------|------------|
| STRONG_BUY | High confidence upward | ≥80% |
| BUY | Moderate upward signal | ≥60% |
| NEUTRAL | No clear direction | - |
| SELL | Moderate downward signal | ≥60% |
| STRONG_SELL | High confidence downward | ≥80% |

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| MT5_LOGIN | MT5 account login | - |
| MT5_PASSWORD | MT5 account password | - |
| MT5_SERVER | MT5 server name | ICMarketsSC-Demo |
| API_KEY | API authentication key | - |
| MODEL_LOOKBACK | Sequence length | 60 |
| CONFIDENCE_THRESHOLD | Min confidence for alerts | 0.7 |
| LOG_LEVEL | Logging level | INFO |

## Testing

```bash
# Run all tests
pytest tests/

# With coverage
pytest --cov=src tests/

# Specific test file
pytest tests/test_models.py -v
```

## License

MIT License
