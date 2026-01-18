# Gold Predictor v2 - Guía de Configuración

## Variables de Entorno (.env)

### Fuente de Datos
```bash
# Selección de fuente: "capital" o "mt5"
DATA_SOURCE=capital
```

### Capital.com API (Principal)
```bash
CAPITAL_API_KEY=tu_api_key
CAPITAL_PASSWORD=tu_password
CAPITAL_IDENTIFIER=tu_email_o_api_key
CAPITAL_DEMO=true                    # true para demo, false para real
```

### MetaTrader 5 (Alternativo)
```bash
MT5_LOGIN=tu_login_demo
MT5_PASSWORD=tu_password_demo
MT5_SERVER=ICMarketsSC-Demo
```

### Seguridad API
```bash
API_KEY=tu_clave_api_generada        # Para autenticar requests al servidor
```

### Configuración de Trading
```bash
SYMBOL=GOLD                          # Símbolo a operar (GOLD/XAUUSD)
TIMEFRAME=M5                         # Timeframe (M5, M15, H1, H4)
```

### Notificaciones Telegram (via n8n)
```bash
TELEGRAM_BOT_TOKEN=tu_token_bot
TELEGRAM_CHAT_ID=tu_chat_id
```

### News API (Análisis de Sentimiento)
```bash
NEWS_API_KEY=tu_newsapi_key          # Obtener en https://newsapi.org
NEWS_CACHE_DURATION_MINUTES=30       # Duración del cache de noticias
```

---

## Configuración del Modelo ML

| Parámetro | Valor Default | Descripción |
|-----------|---------------|-------------|
| `MODEL_LOOKBACK` | 60 | Velas históricas para predicción |
| `PREDICTION_HORIZON` | 1 | Velas futuras a predecir |
| `CONFIDENCE_THRESHOLD` | 0.7 | Umbral mínimo para señales (70%) |
| `LSTM_WEIGHT` | 0.6 | Peso del modelo LSTM en ensemble |
| `XGB_WEIGHT` | 0.4 | Peso del modelo XGBoost en ensemble |

---

## Gestión de Riesgo (RiskManager)

| Parámetro | Valor Default | Descripción |
|-----------|---------------|-------------|
| `DAILY_LOSS_LIMIT_PCT` | 3.0% | Pérdida máxima diaria permitida |
| `MAX_DAILY_TRADES` | 50 | Máximo de trades por día |
| `DEFAULT_ACCOUNT_BALANCE` | $10,000 | Balance para cálculos de riesgo |

### Tamaño de Posición (Volatilidad Ajustada)

| Parámetro | Valor Default | Descripción |
|-----------|---------------|-------------|
| `BASE_POSITION_SIZE` | 0.01 lots | Tamaño base de posición |
| `MAX_POSITION_SIZE` | 0.1 lots | Tamaño máximo de posición |
| `MIN_POSITION_SIZE` | 0.01 lots | Tamaño mínimo de posición |
| `VOLATILITY_HIGH_THRESHOLD` | 1.5x | ATR > 1.5x promedio = reducir tamaño |
| `VOLATILITY_LOW_THRESHOLD` | 0.7x | ATR < 0.7x promedio = aumentar tamaño |
| `VOLATILITY_LOOKBACK` | 20 | Períodos para calcular ATR promedio |

---

## Trailing Stop (PositionManager)

| Parámetro | Valor Default | Descripción |
|-----------|---------------|-------------|
| `TRAILING_STOP_ATR_MULTIPLIER` | 1.5x | Distancia del trailing como múltiplo ATR |
| `TRAILING_ACTIVATION_PIPS` | 10 pips | Pips en profit para activar trailing |
| `TRAILING_STEP_PIPS` | 5 pips | Mínimo pips para mover el stop |
| `PIP_VALUE` | 0.01 | Valor de 1 pip para GOLD |

### Take Profit Parcial (3 niveles)

| Nivel | ATR Multiplier | Porcentaje | Acción |
|-------|----------------|------------|--------|
| TP1 | 1x ATR | 50% | Cerrar 50%, mover SL a breakeven |
| TP2 | 2x ATR | 30% | Cerrar 30% adicional |
| TP3 | 3x ATR | 20% | Cerrar posición restante |

---

## Calendario Económico (EconomicCalendar)

| Parámetro | Valor Default | Descripción |
|-----------|---------------|-------------|
| `REFRESH_INTERVAL_MINUTES` | 60 | Frecuencia de actualización |
| `PAUSE_MINUTES_BEFORE` | 30 | Minutos de pausa antes de evento |
| `PAUSE_MINUTES_AFTER` | 30 | Minutos de pausa después de evento |

### Eventos de Alto Impacto (Pausa Automática)
- Federal Reserve (Fed Rate Decision)
- CPI (Consumer Price Index)
- NFP (Non-Farm Payrolls)
- GDP (Gross Domestic Product)
- Unemployment Rate
- FOMC Minutes
- ECB Decision

### Monedas Relevantes para Oro
`USD`, `EUR`, `GBP`, `JPY`, `CHF`, `AUD`, `CNY`

---

## Indicadores Técnicos

| Indicador | Parámetro | Valor Default |
|-----------|-----------|---------------|
| RSI | Período | 14 |
| MACD | Fast/Slow/Signal | 12/26/9 |
| Bollinger Bands | Período | 20 |
| Bollinger Bands | Std Dev | 2.0 |
| EMA | Períodos | 9, 21, 50 |
| ATR | Período | 14 |

---

## Multi-Timeframe Analysis

| Parámetro | Valor Default | Descripción |
|-----------|---------------|-------------|
| Timeframes | M5, M15, H1, H4 | Marcos temporales analizados |
| Bars per TF | 200 | Velas por timeframe |
| Cache Duration | 5 min | Duración del cache por TF |

### Cálculo de Confluencia
- **80-100%**: Confluencia fuerte - señal confiable
- **60-79%**: Confluencia moderada - proceder con cautela
- **40-59%**: Confluencia débil - señales mixtas
- **0-39%**: Sin confluencia - evitar operar

---

## Backtesting Engine

| Parámetro | Valor Default | Descripción |
|-----------|---------------|-------------|
| `INITIAL_BALANCE` | $10,000 | Balance inicial simulado |
| `POSITION_SIZE_PERCENT` | 2% | Tamaño de posición como % del balance |
| `STOP_LOSS_ATR_MULTIPLIER` | 2.0x | Stop loss como múltiplo ATR |
| `TAKE_PROFIT_ATR_MULTIPLIER` | 3.0x | Take profit como múltiplo ATR |
| `MAX_POSITIONS` | 1 | Posiciones simultáneas máximas |
| `COMMISSION_PERCENT` | 0% | Comisión por trade |
| `SLIPPAGE_PERCENT` | 0.01% | Slippage simulado |

### Walk-Forward Validation

| Parámetro | Valor Default | Descripción |
|-----------|---------------|-------------|
| `TRAIN_DAYS` | 30 | Días de entrenamiento por ventana |
| `TEST_DAYS` | 7 | Días de prueba por ventana |
| `NUM_WINDOWS` | 4 | Número de ventanas walk-forward |

### Criterios de Robustez
- Consistency Score > 60%
- No warnings críticos
- Win Rate promedio > 40%

---

## Almacenamiento de Datos

### Historical Data Store
| Parámetro | Valor | Descripción |
|-----------|-------|-------------|
| Formato | Parquet | Compresión Snappy |
| Directorio | `data/historical/` | Ubicación de archivos |
| Patrón nombre | `{symbol}_{timeframe}.parquet` | Formato de archivos |

### Trade Database
| Parámetro | Valor | Descripción |
|-----------|-------|-------------|
| Formato | SQLite | Base de datos |
| Ubicación | `data/trades.db` | Archivo DB |

### Prediction Store
| Parámetro | Valor | Descripción |
|-----------|-------|-------------|
| Formato | SQLite | Base de datos |
| Ubicación | `predictions.db` | Archivo DB |

---

## Servidor API

| Parámetro | Valor Default | Descripción |
|-----------|---------------|-------------|
| `HOST` | 0.0.0.0 | Dirección de escucha |
| `PORT` | 8000 | Puerto del servidor |
| `LOG_LEVEL` | INFO | Nivel de logging |

---

## Endpoints API Disponibles

### Predicción y Trading
| Endpoint | Método | Descripción |
|----------|--------|-------------|
| `/predict` | POST | Obtener predicción |
| `/predict-and-trade` | POST | Predicción + ejecutar trade |
| `/health` | GET | Estado del servidor |

### Gestión de Riesgo
| Endpoint | Método | Descripción |
|----------|--------|-------------|
| `/api/daily-stats` | GET | Estadísticas diarias P&L |
| `/api/risk-status` | GET | Estado del risk manager |
| `/api/record-pnl` | POST | Registrar P&L de trade |
| `/api/update-balance` | POST | Actualizar balance cuenta |

### Calendario y Noticias
| Endpoint | Método | Descripción |
|----------|--------|-------------|
| `/api/calendar` | GET | Eventos económicos |
| `/api/calendar/upcoming` | GET | Próximos eventos |
| `/api/news` | GET | Noticias recientes |
| `/api/news/headlines` | GET | Solo titulares |

### Sentimiento
| Endpoint | Método | Descripción |
|----------|--------|-------------|
| `/api/sentiment` | GET | Sentimiento agregado |
| `/api/sentiment/analyze` | POST | Analizar texto |
| `/api/sentiment/analyze-batch` | POST | Analizar múltiples textos |

### Multi-Timeframe
| Endpoint | Método | Descripción |
|----------|--------|-------------|
| `/api/timeframes` | GET | Análisis multi-TF |
| `/api/timeframes/confluence` | GET | Score de confluencia |
| `/api/timeframes/{tf}` | GET | Datos de timeframe específico |

### Performance
| Endpoint | Método | Descripción |
|----------|--------|-------------|
| `/api/performance` | GET | Métricas de trading |
| `/api/performance/drawdown` | GET | Info de drawdown |
| `/api/performance/equity-curve` | GET | Curva de equity |
| `/api/daily-report` | GET | Reporte diario formateado |

### Trades
| Endpoint | Método | Descripción |
|----------|--------|-------------|
| `/api/trades` | GET | Historial de trades |
| `/api/trades/summary` | GET | Resumen estadístico |

### Historical Data
| Endpoint | Método | Descripción |
|----------|--------|-------------|
| `/api/historical` | GET | Lista datos disponibles |
| `/api/historical/{symbol}/{tf}` | GET | Info de dataset |
| `/api/historical/download` | POST | Descargar datos |
| `/api/historical/update` | POST | Actualizar datos |

### Backtesting
| Endpoint | Método | Descripción |
|----------|--------|-------------|
| `/api/backtest` | POST | Ejecutar backtest |
| `/api/backtest/status` | GET | Estado del engine |
| `/api/backtest/walk-forward` | POST | Validación walk-forward |

---

## Ejemplo de Archivo .env Completo

```bash
# === FUENTE DE DATOS ===
DATA_SOURCE=capital

# === CAPITAL.COM ===
CAPITAL_API_KEY=xxxxxxxxxx
CAPITAL_PASSWORD=xxxxxxxxxx
CAPITAL_IDENTIFIER=tu@email.com
CAPITAL_DEMO=true

# === MT5 (BACKUP) ===
MT5_LOGIN=12345678
MT5_PASSWORD=xxxxxxxxxx
MT5_SERVER=ICMarketsSC-Demo

# === SEGURIDAD API ===
API_KEY=mi-clave-secreta-cambiar-en-produccion

# === TRADING ===
SYMBOL=GOLD
TIMEFRAME=M5

# === MODELO ===
MODEL_LOOKBACK=60
PREDICTION_HORIZON=1
CONFIDENCE_THRESHOLD=0.7

# === RISK MANAGEMENT ===
DAILY_LOSS_LIMIT_PCT=3.0
MAX_DAILY_TRADES=50
DEFAULT_ACCOUNT_BALANCE=10000

# === POSITION SIZING ===
BASE_POSITION_SIZE=0.01
MAX_POSITION_SIZE=0.1
MIN_POSITION_SIZE=0.01
VOLATILITY_HIGH_THRESHOLD=1.5
VOLATILITY_LOW_THRESHOLD=0.7

# === TRAILING STOP ===
TRAILING_STOP_ATR_MULTIPLIER=1.5
TRAILING_ACTIVATION_PIPS=10
TRAILING_STEP_PIPS=5

# === TELEGRAM (via n8n) ===
TELEGRAM_BOT_TOKEN=123456:ABC-xxxxx
TELEGRAM_CHAT_ID=-100123456789

# === NEWS API ===
NEWS_API_KEY=xxxxxxxxxx
NEWS_CACHE_DURATION_MINUTES=30

# === SERVIDOR ===
HOST=0.0.0.0
PORT=8000
LOG_LEVEL=INFO
DATABASE_URL=sqlite:///./predictions.db
```

---

## Workflows n8n

### gold-prediction-trade.json
- **Trigger**: Cada 5 minutos
- **Acción**: Llama `/predict-and-trade`, envía alerta Telegram si hay señal

### daily-report.json
- **Trigger**: 22:00 UTC (Lun-Vie)
- **Acción**: Genera reporte diario y envía por Telegram

---

## Notas Importantes

1. **Modo Demo**: Siempre probar primero con `CAPITAL_DEMO=true`
2. **API Key**: Generar una clave segura para producción
3. **News API**: Plan gratuito tiene límite de 100 requests/día
4. **Trailing Stop**: Solo se activa después de X pips en profit
5. **Walk-Forward**: Requiere al menos 16 semanas de datos históricos
6. **Daily Reset**: El contador de pérdidas se resetea a medianoche UTC
