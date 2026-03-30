# 📈 Stock Market Predictor v1.0

A comprehensive CLI tool that predicts next-day stock price movement using:

- **Technical Indicators** — RSI, MACD, Bollinger Bands, Stochastic, SMA crossovers, ATR, OBV, Williams %R
- **Machine Learning** — Random Forest + Gradient Boosting ensemble with walk-forward cross-validation
- **Deep Learning** — LSTM recurrent neural network for sequence-based prediction
- **Sentiment Analysis** — News headline sentiment scoring via TextBlob

Supports **US Stocks**, **Korean Stocks (KOSPI/KOSDAQ)**, and **Crypto**.

---

## ⚙️ Installation

```bash
# Clone / copy the project
cd stock_predictor

# Install dependencies
pip install -r requirements.txt

# (Optional) Install TensorFlow for LSTM support
pip install tensorflow
```

---

## 🚀 Usage

### Basic — US Stock
```bash
python stock_predictor.py AAPL
python stock_predictor.py TSLA
python stock_predictor.py MSFT
```

### Korean Stocks
```bash
python stock_predictor.py 005930 --market kospi    # Samsung Electronics
python stock_predictor.py 035720 --market kosdaq   # Kakao
```

### Crypto
```bash
python stock_predictor.py BTC-USD --market crypto
python stock_predictor.py ETH-USD --market crypto
```

### Options
```bash
# Use 5 years of data
python stock_predictor.py AAPL --period 5y

# Skip slow models
python stock_predictor.py AAPL --skip-lstm
python stock_predictor.py AAPL --skip-ml --skip-sentiment

# Export results to JSON
python stock_predictor.py AAPL --export prediction.json
```

### All Options
| Flag               | Description                                |
|--------------------|--------------------------------------------|
| `--market`         | `us`, `kospi`, `kosdaq`, `crypto`          |
| `--period`         | `1y`, `2y`, `5y`, `max` (default: `2y`)    |
| `--skip-ml`        | Skip Random Forest / Gradient Boosting     |
| `--skip-lstm`      | Skip LSTM deep learning model              |
| `--skip-sentiment` | Skip news sentiment analysis               |
| `--export FILE`    | Save full results to JSON                  |

---

## 📊 How It Works

### 1. Technical Analysis (weight: 30%)
Computes 15+ technical indicators and aggregates them into a weighted bullish/bearish score:

| Indicator          | What it detects                     |
|--------------------|-------------------------------------|
| RSI                | Overbought / oversold               |
| MACD               | Momentum direction                  |
| Bollinger Bands    | Price deviation from mean            |
| SMA Crossovers     | Trend changes (5/20, 50/200)        |
| Stochastic         | Short-term momentum extremes        |
| Volume Ratio       | Unusual trading activity             |

### 2. Machine Learning (weight: 35%)
Trains a **Random Forest** and **Gradient Boosting** classifier on technical features using time-series cross-validation, then ensembles their probability predictions.

### 3. LSTM Deep Learning (weight: 20%)
An LSTM neural network that learns temporal patterns from 30-day lookback windows of technical indicators.

### 4. Sentiment Analysis (weight: 15%)
Fetches recent news headlines via Yahoo Finance and scores them using TextBlob polarity analysis.

### 5. Ensemble
All four engines produce a BULLISH / BEARISH / NEUTRAL signal with confidence. The ensemble combines them using the weights above to produce the final prediction.

---

## ⚠️ Disclaimer

**This tool is for EDUCATIONAL purposes only.**

- No model can reliably predict stock markets
- Past performance does not guarantee future results
- Never invest solely based on algorithmic predictions
- Always do your own research and consult a financial advisor

---

## 📁 Project Structure

```
stock_predictor/
├── stock_predictor.py   # Main application
├── requirements.txt     # Python dependencies
└── README.md            # This file
```

## License

MIT — Use freely, but invest responsibly.
