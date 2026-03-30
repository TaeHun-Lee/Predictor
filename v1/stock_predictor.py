#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════╗
║              STOCK MARKET PREDICTOR v1.0                        ║
║  Technical Indicators + ML Models + Sentiment Analysis          ║
║  Supports: US Stocks, Korean Stocks (KOSPI/KOSDAQ), Crypto      ║
║                                                                  ║
║  ⚠ DISCLAIMER: This tool is for EDUCATIONAL purposes only.      ║
║  Past performance does not guarantee future results.             ║
║  Never invest solely based on algorithmic predictions.           ║
╚══════════════════════════════════════════════════════════════════╝
"""

import os
import sys
import json
import warnings
import argparse
from datetime import datetime, timedelta
from textwrap import dedent

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ─── Optional dependency imports ────────────────────────────────────────
try:
    import yfinance as yf
    HAS_YFINANCE = True
except ImportError:
    HAS_YFINANCE = False

try:
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.model_selection import TimeSeriesSplit
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import classification_report, accuracy_score
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout
    from tensorflow.keras.callbacks import EarlyStopping
    HAS_TF = True
except ImportError:
    HAS_TF = False

try:
    from textblob import TextBlob
    HAS_TEXTBLOB = True
except ImportError:
    HAS_TEXTBLOB = False

try:
    import requests
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False

try:
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    from rich.text import Text
    from rich.progress import track
    from rich import box
    HAS_RICH = True
    console = Console()
except ImportError:
    HAS_RICH = False
    console = None


# ═══════════════════════════════════════════════════════════════════
#  DISPLAY HELPERS
# ═══════════════════════════════════════════════════════════════════

def print_banner():
    banner = """
    ╔══════════════════════════════════════════════════════════════╗
    ║           📈  STOCK MARKET PREDICTOR  v1.0  📉             ║
    ║    Technical Analysis · Machine Learning · Sentiment        ║
    ╚══════════════════════════════════════════════════════════════╝
    """
    if HAS_RICH:
        console.print(Panel.fit(
            "[bold cyan]📈 STOCK MARKET PREDICTOR v1.0 📉[/]\n"
            "[dim]Technical Analysis · Machine Learning · Sentiment[/]",
            border_style="bright_blue",
        ))
    else:
        print(banner)


def print_signal(signal: str, confidence: float):
    """Print a colored prediction signal."""
    icons = {"BULLISH": "🟢", "BEARISH": "🔴", "NEUTRAL": "🟡"}
    icon = icons.get(signal, "⚪")
    bar_len = int(confidence * 30)
    bar = "█" * bar_len + "░" * (30 - bar_len)

    if HAS_RICH:
        color = {"BULLISH": "green", "BEARISH": "red", "NEUTRAL": "yellow"}.get(signal, "white")
        console.print(f"\n  {icon}  Prediction: [{color} bold]{signal}[/]")
        console.print(f"     Confidence: [{color}]{bar}[/] {confidence:.1%}")
    else:
        print(f"\n  {icon}  Prediction: {signal}")
        print(f"     Confidence: {bar} {confidence:.1%}")


# ═══════════════════════════════════════════════════════════════════
#  DATA FETCHING
# ═══════════════════════════════════════════════════════════════════

class DataFetcher:
    """Fetch historical price data from Yahoo Finance."""

    # Common ticker mappings for convenience
    MARKET_SUFFIXES = {
        "kospi": ".KS",
        "kosdaq": ".KQ",
    }

    @staticmethod
    def get_ticker(symbol: str, market: str = "us") -> str:
        """Convert a symbol + market into a yfinance-compatible ticker."""
        market = market.lower()
        if market in ("us", "nyse", "nasdaq", "crypto"):
            return symbol.upper()
        suffix = DataFetcher.MARKET_SUFFIXES.get(market, "")
        return f"{symbol}{suffix}"

    @staticmethod
    def fetch(symbol: str, market: str = "us", period: str = "2y") -> pd.DataFrame:
        """Download OHLCV data. Returns a DataFrame with standard columns."""
        if not HAS_YFINANCE:
            raise ImportError("yfinance is required. Install: pip install yfinance")

        ticker = DataFetcher.get_ticker(symbol, market)
        print(f"  ⏳  Fetching data for {ticker} (period={period}) ...")

        df = yf.download(ticker, period=period, progress=False, auto_adjust=True)
        if df.empty:
            raise ValueError(f"No data returned for {ticker}. Check the symbol/market.")

        # Flatten MultiIndex columns if present
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        df.index = pd.to_datetime(df.index)
        df = df.sort_index()
        print(f"  ✅  Loaded {len(df)} trading days ({df.index[0].date()} → {df.index[-1].date()})")
        return df


# ═══════════════════════════════════════════════════════════════════
#  TECHNICAL INDICATORS ENGINE
# ═══════════════════════════════════════════════════════════════════

class TechnicalAnalysis:
    """Calculate common technical indicators and derive a signal."""

    @staticmethod
    def compute(df: pd.DataFrame) -> pd.DataFrame:
        """Add all indicator columns to the dataframe."""
        df = df.copy()
        close = df["Close"].squeeze()
        high = df["High"].squeeze()
        low = df["Low"].squeeze()
        volume = df["Volume"].squeeze()

        # ── Moving Averages ──
        df["SMA_5"]   = close.rolling(5).mean()
        df["SMA_10"]  = close.rolling(10).mean()
        df["SMA_20"]  = close.rolling(20).mean()
        df["SMA_50"]  = close.rolling(50).mean()
        df["SMA_200"] = close.rolling(200).mean()
        df["EMA_12"]  = close.ewm(span=12, adjust=False).mean()
        df["EMA_26"]  = close.ewm(span=26, adjust=False).mean()

        # ── MACD ──
        df["MACD"]        = df["EMA_12"] - df["EMA_26"]
        df["MACD_Signal"] = df["MACD"].ewm(span=9, adjust=False).mean()
        df["MACD_Hist"]   = df["MACD"] - df["MACD_Signal"]

        # ── RSI (14-period) ──
        delta = close.diff()
        gain = delta.clip(lower=0).rolling(14).mean()
        loss = (-delta.clip(upper=0)).rolling(14).mean()
        rs = gain / loss.replace(0, np.nan)
        df["RSI"] = 100 - (100 / (1 + rs))

        # ── Bollinger Bands (20, 2σ) ──
        df["BB_Mid"]   = close.rolling(20).mean()
        bb_std         = close.rolling(20).std()
        df["BB_Upper"] = df["BB_Mid"] + 2 * bb_std
        df["BB_Lower"] = df["BB_Mid"] - 2 * bb_std
        df["BB_Width"] = (df["BB_Upper"] - df["BB_Lower"]) / df["BB_Mid"]
        df["BB_Pct"]   = (close - df["BB_Lower"]) / (df["BB_Upper"] - df["BB_Lower"])

        # ── Stochastic Oscillator (14-period) ──
        low_14  = low.rolling(14).min()
        high_14 = high.rolling(14).max()
        df["Stoch_K"] = 100 * (close - low_14) / (high_14 - low_14)
        df["Stoch_D"] = df["Stoch_K"].rolling(3).mean()

        # ── ATR (Average True Range, 14-period) ──
        tr1 = high - low
        tr2 = (high - close.shift(1)).abs()
        tr3 = (low - close.shift(1)).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        df["ATR"] = tr.rolling(14).mean()

        # ── OBV (On-Balance Volume) ──
        sign = np.sign(close.diff()).fillna(0)
        df["OBV"] = (sign * volume).cumsum()

        # ── Volume SMA ──
        df["Vol_SMA_20"] = volume.rolling(20).mean()
        df["Vol_Ratio"]  = volume / df["Vol_SMA_20"]

        # ── Price Rate of Change ──
        df["ROC_5"]  = close.pct_change(5)  * 100
        df["ROC_10"] = close.pct_change(10) * 100
        df["ROC_20"] = close.pct_change(20) * 100

        # ── Williams %R ──
        df["Williams_R"] = -100 * (high_14 - close) / (high_14 - low_14)

        # ── Target: next-day direction (1 = up, 0 = down) ──
        df["Next_Return"] = close.pct_change().shift(-1)
        df["Target"]      = (df["Next_Return"] > 0).astype(int)

        return df

    @staticmethod
    def get_signal(df: pd.DataFrame) -> dict:
        """Produce a composite technical signal from the latest row."""
        latest = df.iloc[-1]
        close  = float(latest["Close"])
        scores = []  # list of (indicator_name, signal, weight)

        # RSI
        rsi = float(latest["RSI"]) if pd.notna(latest["RSI"]) else 50
        if rsi < 30:
            scores.append(("RSI", "BULLISH", 1.5))
        elif rsi > 70:
            scores.append(("RSI", "BEARISH", 1.5))
        else:
            scores.append(("RSI", "NEUTRAL", 0.5))

        # MACD
        if pd.notna(latest["MACD_Hist"]):
            macd_h = float(latest["MACD_Hist"])
            if macd_h > 0:
                scores.append(("MACD", "BULLISH", 1.2))
            else:
                scores.append(("MACD", "BEARISH", 1.2))

        # Bollinger %B
        if pd.notna(latest["BB_Pct"]):
            bb = float(latest["BB_Pct"])
            if bb < 0.2:
                scores.append(("Bollinger", "BULLISH", 1.0))
            elif bb > 0.8:
                scores.append(("Bollinger", "BEARISH", 1.0))
            else:
                scores.append(("Bollinger", "NEUTRAL", 0.3))

        # SMA crossovers
        for fast, slow, w in [("SMA_5", "SMA_20", 0.8), ("SMA_50", "SMA_200", 1.5)]:
            if pd.notna(latest[fast]) and pd.notna(latest[slow]):
                if latest[fast] > latest[slow]:
                    scores.append((f"{fast}/{slow}", "BULLISH", w))
                else:
                    scores.append((f"{fast}/{slow}", "BEARISH", w))

        # Stochastic
        if pd.notna(latest["Stoch_K"]):
            sk = float(latest["Stoch_K"])
            if sk < 20:
                scores.append(("Stochastic", "BULLISH", 0.8))
            elif sk > 80:
                scores.append(("Stochastic", "BEARISH", 0.8))
            else:
                scores.append(("Stochastic", "NEUTRAL", 0.3))

        # Volume confirmation
        if pd.notna(latest["Vol_Ratio"]):
            vr = float(latest["Vol_Ratio"])
            if vr > 1.5:
                scores.append(("Volume", "BULLISH" if latest["ROC_5"] > 0 else "BEARISH", 0.7))

        # Aggregate
        bull = sum(w for _, s, w in scores if s == "BULLISH")
        bear = sum(w for _, s, w in scores if s == "BEARISH")
        total = bull + bear + 0.001

        if bull > bear * 1.2:
            signal = "BULLISH"
            confidence = bull / total
        elif bear > bull * 1.2:
            signal = "BEARISH"
            confidence = bear / total
        else:
            signal = "NEUTRAL"
            confidence = 1 - abs(bull - bear) / total

        return {
            "signal": signal,
            "confidence": round(confidence, 4),
            "details": scores,
            "rsi": rsi,
            "macd_hist": float(latest["MACD_Hist"]) if pd.notna(latest["MACD_Hist"]) else None,
            "bb_pct": float(latest["BB_Pct"]) if pd.notna(latest["BB_Pct"]) else None,
        }


# ═══════════════════════════════════════════════════════════════════
#  MACHINE LEARNING ENGINE
# ═══════════════════════════════════════════════════════════════════

class MLPredictor:
    """Train Random Forest + Gradient Boosting on technical features."""

    FEATURE_COLS = [
        "RSI", "MACD", "MACD_Signal", "MACD_Hist",
        "BB_Pct", "BB_Width",
        "Stoch_K", "Stoch_D",
        "ATR",
        "Vol_Ratio",
        "ROC_5", "ROC_10", "ROC_20",
        "Williams_R",
    ]

    def __init__(self):
        self.rf = RandomForestClassifier(
            n_estimators=200, max_depth=8, min_samples_leaf=10, random_state=42, n_jobs=-1
        )
        self.gb = GradientBoostingClassifier(
            n_estimators=150, max_depth=5, learning_rate=0.05, random_state=42
        )
        self.scaler = StandardScaler()

    def prepare_data(self, df: pd.DataFrame):
        """Extract features and target from the indicator-enriched dataframe."""
        feature_df = df[self.FEATURE_COLS + ["Target"]].dropna()
        X = feature_df[self.FEATURE_COLS].values
        y = feature_df["Target"].values
        return X, y

    def train_and_predict(self, df: pd.DataFrame) -> dict:
        """Walk-forward validation + final prediction for next day."""
        if not HAS_SKLEARN:
            raise ImportError("scikit-learn is required for ML. Install: pip install scikit-learn")

        X, y = self.prepare_data(df)
        if len(X) < 100:
            raise ValueError("Not enough data for ML training (need ≥ 100 rows after indicator warm-up).")

        # Time-series split for validation
        tscv = TimeSeriesSplit(n_splits=5)
        rf_scores, gb_scores = [], []

        print("  🤖  Training ML models (walk-forward cross-validation) ...")

        for train_idx, test_idx in tscv.split(X):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            scaler = StandardScaler()
            X_train_s = scaler.fit_transform(X_train)
            X_test_s  = scaler.transform(X_test)

            self.rf.fit(X_train_s, y_train)
            self.gb.fit(X_train_s, y_train)

            rf_scores.append(self.rf.score(X_test_s, y_test))
            gb_scores.append(self.gb.score(X_test_s, y_test))

        # Final fit on all data except last row
        X_all_s = self.scaler.fit_transform(X[:-1])
        self.rf.fit(X_all_s, y[:-1])
        self.gb.fit(X_all_s, y[:-1])

        # Predict next day
        X_last = self.scaler.transform(X[-1:])
        rf_prob = self.rf.predict_proba(X_last)[0]
        gb_prob = self.gb.predict_proba(X_last)[0]

        # Ensemble average
        avg_prob = (rf_prob + gb_prob) / 2
        pred_class = int(np.argmax(avg_prob))
        confidence = float(avg_prob[pred_class])

        signal = "BULLISH" if pred_class == 1 else "BEARISH"

        # Feature importance (from Random Forest)
        importances = dict(zip(self.FEATURE_COLS, self.rf.feature_importances_))
        top_features = sorted(importances.items(), key=lambda x: -x[1])[:5]

        return {
            "signal": signal,
            "confidence": round(confidence, 4),
            "rf_accuracy": round(np.mean(rf_scores), 4),
            "gb_accuracy": round(np.mean(gb_scores), 4),
            "top_features": top_features,
        }


# ═══════════════════════════════════════════════════════════════════
#  LSTM DEEP LEARNING ENGINE
# ═══════════════════════════════════════════════════════════════════

class LSTMPredictor:
    """LSTM-based sequence predictor for next-day direction."""

    FEATURE_COLS = MLPredictor.FEATURE_COLS
    SEQ_LEN = 30  # look-back window

    def build_model(self, n_features: int):
        model = Sequential([
            LSTM(64, return_sequences=True, input_shape=(self.SEQ_LEN, n_features)),
            Dropout(0.2),
            LSTM(32),
            Dropout(0.2),
            Dense(16, activation="relu"),
            Dense(1, activation="sigmoid"),
        ])
        model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
        return model

    def prepare_sequences(self, df: pd.DataFrame):
        feature_df = df[self.FEATURE_COLS + ["Target"]].dropna()
        scaler = StandardScaler()
        X_raw = scaler.fit_transform(feature_df[self.FEATURE_COLS].values)
        y_raw = feature_df["Target"].values

        X_seq, y_seq = [], []
        for i in range(self.SEQ_LEN, len(X_raw)):
            X_seq.append(X_raw[i - self.SEQ_LEN : i])
            y_seq.append(y_raw[i])

        return np.array(X_seq), np.array(y_seq), scaler

    def train_and_predict(self, df: pd.DataFrame) -> dict:
        if not HAS_TF:
            return {"signal": "N/A", "confidence": 0.0, "note": "TensorFlow not installed. pip install tensorflow"}

        print("  🧠  Training LSTM model ...")
        X, y, scaler = self.prepare_sequences(df)

        if len(X) < 60:
            return {"signal": "N/A", "confidence": 0.0, "note": "Not enough data for LSTM."}

        split = int(len(X) * 0.8)
        X_train, X_test = X[:split], X[split:]
        y_train, y_test = y[:split], y[split:]

        tf.random.set_seed(42)
        model = self.build_model(X.shape[2])
        model.fit(
            X_train, y_train,
            epochs=30, batch_size=32,
            validation_data=(X_test, y_test),
            callbacks=[EarlyStopping(patience=5, restore_best_weights=True)],
            verbose=0,
        )

        test_acc = model.evaluate(X_test, y_test, verbose=0)[1]

        # Predict next day
        prob = float(model.predict(X[-1:], verbose=0)[0][0])
        signal = "BULLISH" if prob > 0.5 else "BEARISH"
        confidence = prob if prob > 0.5 else (1 - prob)

        return {
            "signal": signal,
            "confidence": round(confidence, 4),
            "test_accuracy": round(test_acc, 4),
        }


# ═══════════════════════════════════════════════════════════════════
#  SENTIMENT ANALYSIS ENGINE
# ═══════════════════════════════════════════════════════════════════

class SentimentAnalyzer:
    """Simple sentiment analysis using TextBlob on news headlines."""

    @staticmethod
    def fetch_headlines(symbol: str) -> list[str]:
        """Try to get news headlines from yfinance."""
        if not HAS_YFINANCE:
            return []
        try:
            ticker = yf.Ticker(symbol)
            news = ticker.news or []
            headlines = []
            for item in news[:20]:
                title = item.get("title", "")
                if title:
                    headlines.append(title)
            return headlines
        except Exception:
            return []

    @staticmethod
    def analyze(symbol: str) -> dict:
        if not HAS_TEXTBLOB:
            return {
                "signal": "N/A",
                "confidence": 0.0,
                "note": "TextBlob not installed. pip install textblob",
            }

        headlines = SentimentAnalyzer.fetch_headlines(symbol)
        if not headlines:
            return {
                "signal": "NEUTRAL",
                "confidence": 0.5,
                "n_headlines": 0,
                "note": "No headlines found; defaulting to neutral.",
            }

        sentiments = []
        for h in headlines:
            blob = TextBlob(h)
            sentiments.append(blob.sentiment.polarity)

        avg = np.mean(sentiments)
        if avg > 0.05:
            signal = "BULLISH"
            confidence = min(0.5 + avg, 0.95)
        elif avg < -0.05:
            signal = "BEARISH"
            confidence = min(0.5 + abs(avg), 0.95)
        else:
            signal = "NEUTRAL"
            confidence = 0.5

        return {
            "signal": signal,
            "confidence": round(confidence, 4),
            "avg_polarity": round(avg, 4),
            "n_headlines": len(headlines),
            "sample_headlines": headlines[:5],
        }


# ═══════════════════════════════════════════════════════════════════
#  ENSEMBLE COMBINER
# ═══════════════════════════════════════════════════════════════════

class EnsembleCombiner:
    """Combine signals from all engines into a final prediction."""

    WEIGHTS = {
        "technical":  0.30,
        "ml":         0.35,
        "lstm":       0.20,
        "sentiment":  0.15,
    }

    @staticmethod
    def combine(tech: dict, ml: dict, lstm: dict, sentiment: dict) -> dict:
        signals = {
            "technical":  tech,
            "ml":         ml,
            "lstm":       lstm,
            "sentiment":  sentiment,
        }

        score = 0.0  # positive = bullish, negative = bearish
        total_weight = 0.0

        for key, result in signals.items():
            if result.get("signal") in ("N/A", None):
                continue
            w = EnsembleCombiner.WEIGHTS[key]
            conf = result.get("confidence", 0.5)

            if result["signal"] == "BULLISH":
                score += w * conf
            elif result["signal"] == "BEARISH":
                score -= w * conf
            # NEUTRAL contributes 0

            total_weight += w

        if total_weight == 0:
            return {"signal": "NEUTRAL", "confidence": 0.5, "score": 0}

        norm_score = score / total_weight  # range roughly [-1, 1]

        if norm_score > 0.1:
            signal = "BULLISH"
        elif norm_score < -0.1:
            signal = "BEARISH"
        else:
            signal = "NEUTRAL"

        confidence = min(abs(norm_score) + 0.5, 0.95)

        return {
            "signal": signal,
            "confidence": round(confidence, 4),
            "raw_score": round(norm_score, 4),
            "components": {k: v.get("signal", "N/A") for k, v in signals.items()},
        }


# ═══════════════════════════════════════════════════════════════════
#  REPORT PRINTER
# ═══════════════════════════════════════════════════════════════════

def print_report(symbol, market, tech, ml, lstm, sentiment, ensemble, df):
    """Print a beautiful summary report."""
    close = float(df["Close"].iloc[-1])
    date = df.index[-1].strftime("%Y-%m-%d")

    if HAS_RICH:
        # ── Header ──
        console.print()
        console.rule(f"[bold] Analysis Report: {symbol.upper()} ({market.upper()}) ", style="cyan")
        console.print(f"  Date: {date}   |   Last Close: ${close:,.2f}\n")

        # ── Technical Analysis ──
        t = Table(title="📊 Technical Analysis", box=box.ROUNDED, border_style="blue")
        t.add_column("Indicator", style="cyan", width=20)
        t.add_column("Signal", justify="center", width=12)
        t.add_column("Weight", justify="right", width=8)
        for name, sig, weight in tech.get("details", []):
            color = {"BULLISH": "green", "BEARISH": "red", "NEUTRAL": "yellow"}.get(sig, "white")
            t.add_row(name, f"[{color}]{sig}[/]", f"{weight:.1f}")
        console.print(t)
        print_signal(tech["signal"], tech["confidence"])

        console.print(f"\n  RSI: {tech.get('rsi', 'N/A'):.1f}  |  MACD Hist: {tech.get('macd_hist', 'N/A')}  |  BB%: {tech.get('bb_pct', 'N/A')}")

        # ── ML Results ──
        console.print()
        console.rule("[bold] 🤖 Machine Learning ", style="magenta")
        if ml.get("signal") != "N/A":
            print_signal(ml["signal"], ml["confidence"])
            console.print(f"  RF CV Accuracy: {ml.get('rf_accuracy', 'N/A'):.1%}")
            console.print(f"  GB CV Accuracy: {ml.get('gb_accuracy', 'N/A'):.1%}")
            console.print("  Top Features:")
            for feat, imp in ml.get("top_features", []):
                bar = "█" * int(imp * 100)
                console.print(f"    {feat:16s} {bar} ({imp:.3f})")
        else:
            console.print(f"  [dim]{ml.get('note', 'Skipped')}[/]")

        # ── LSTM Results ──
        console.print()
        console.rule("[bold] 🧠 LSTM Deep Learning ", style="green")
        if lstm.get("signal") != "N/A":
            print_signal(lstm["signal"], lstm["confidence"])
            console.print(f"  Test Accuracy: {lstm.get('test_accuracy', 'N/A'):.1%}")
        else:
            console.print(f"  [dim]{lstm.get('note', 'Skipped')}[/]")

        # ── Sentiment ──
        console.print()
        console.rule("[bold] 💬 Sentiment Analysis ", style="yellow")
        if sentiment.get("signal") != "N/A":
            print_signal(sentiment["signal"], sentiment["confidence"])
            console.print(f"  Headlines analyzed: {sentiment.get('n_headlines', 0)}")
            console.print(f"  Avg polarity: {sentiment.get('avg_polarity', 'N/A')}")
            for h in sentiment.get("sample_headlines", []):
                console.print(f"    • {h[:80]}")
        else:
            console.print(f"  [dim]{sentiment.get('note', 'Skipped')}[/]")

        # ── FINAL ENSEMBLE ──
        console.print()
        console.rule("[bold bright_white] ⚡ FINAL ENSEMBLE PREDICTION ", style="bright_white")
        print_signal(ensemble["signal"], ensemble["confidence"])
        console.print(f"\n  Raw score: {ensemble.get('raw_score', 0):.4f}  (>0 bullish, <0 bearish)")
        console.print("  Components: " + " | ".join(
            f"{k}: {v}" for k, v in ensemble.get("components", {}).items()
        ))

        # Disclaimer
        console.print()
        console.print(Panel(
            "[bold red]⚠ DISCLAIMER[/]\n"
            "This is NOT financial advice. Predictions are based on historical patterns\n"
            "and statistical models that have NO guarantee of future accuracy.\n"
            "Always do your own research. Invest responsibly.",
            border_style="red",
        ))

    else:
        # Fallback plain text
        print(f"\n{'='*60}")
        print(f"  Analysis Report: {symbol.upper()} ({market.upper()})")
        print(f"  Date: {date}  |  Last Close: ${close:,.2f}")
        print(f"{'='*60}")
        print(f"\n--- Technical Analysis ---")
        for name, sig, weight in tech.get("details", []):
            print(f"  {name:16s}  {sig:8s}  (w={weight:.1f})")
        print_signal(tech["signal"], tech["confidence"])

        print(f"\n--- Machine Learning ---")
        if ml.get("signal") != "N/A":
            print_signal(ml["signal"], ml["confidence"])
            print(f"  RF Accuracy: {ml.get('rf_accuracy', 'N/A')}")
            print(f"  GB Accuracy: {ml.get('gb_accuracy', 'N/A')}")
        else:
            print(f"  {ml.get('note', 'Skipped')}")

        print(f"\n--- LSTM ---")
        if lstm.get("signal") != "N/A":
            print_signal(lstm["signal"], lstm["confidence"])
        else:
            print(f"  {lstm.get('note', 'Skipped')}")

        print(f"\n--- Sentiment ---")
        if sentiment.get("signal") != "N/A":
            print_signal(sentiment["signal"], sentiment["confidence"])
        else:
            print(f"  {sentiment.get('note', 'Skipped')}")

        print(f"\n{'='*60}")
        print("  ⚡ FINAL ENSEMBLE PREDICTION")
        print_signal(ensemble["signal"], ensemble["confidence"])
        print(f"  Raw score: {ensemble.get('raw_score', 0):.4f}")
        print(f"\n  ⚠ DISCLAIMER: NOT financial advice. Invest responsibly.")
        print(f"{'='*60}\n")


# ═══════════════════════════════════════════════════════════════════
#  EXPORT
# ═══════════════════════════════════════════════════════════════════

def export_results(symbol, tech, ml, lstm, sentiment, ensemble, output_path):
    """Save prediction results to a JSON file."""
    result = {
        "symbol": symbol,
        "timestamp": datetime.now().isoformat(),
        "prediction_for": (datetime.now() + timedelta(days=1)).strftime("%Y-%m-%d"),
        "technical": tech,
        "ml": ml,
        "lstm": lstm,
        "sentiment": sentiment,
        "ensemble": ensemble,
    }
    # Convert numpy types
    def convert(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    with open(output_path, "w") as f:
        json.dump(result, f, indent=2, default=convert)
    print(f"  💾  Results saved to {output_path}")


# ═══════════════════════════════════════════════════════════════════
#  CLI ENTRY POINT
# ═══════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Stock Market Predictor — next-day direction prediction",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=dedent("""\
        Examples:
          python stock_predictor.py AAPL
          python stock_predictor.py 005930 --market kospi
          python stock_predictor.py BTC-USD --market crypto
          python stock_predictor.py TSLA --period 5y --skip-lstm
          python stock_predictor.py MSFT --export results.json
        """),
    )
    parser.add_argument("symbol", help="Ticker symbol (e.g. AAPL, 005930, BTC-USD)")
    parser.add_argument("--market", default="us",
                        choices=["us", "nyse", "nasdaq", "kospi", "kosdaq", "crypto"],
                        help="Market for the symbol (default: us)")
    parser.add_argument("--period", default="2y",
                        help="Data period: 1y, 2y, 5y, max (default: 2y)")
    parser.add_argument("--skip-ml", action="store_true", help="Skip ML models")
    parser.add_argument("--skip-lstm", action="store_true", help="Skip LSTM model")
    parser.add_argument("--skip-sentiment", action="store_true", help="Skip sentiment analysis")
    parser.add_argument("--export", metavar="FILE", help="Export results to JSON file")

    args = parser.parse_args()

    print_banner()

    # ── Step 1: Fetch Data ──
    try:
        df = DataFetcher.fetch(args.symbol, args.market, args.period)
    except Exception as e:
        print(f"  ❌  Error fetching data: {e}")
        sys.exit(1)

    # ── Step 2: Technical Analysis ──
    print("\n  📊  Computing technical indicators ...")
    df = TechnicalAnalysis.compute(df)
    tech_signal = TechnicalAnalysis.get_signal(df)

    # ── Step 3: ML Prediction ──
    if args.skip_ml or not HAS_SKLEARN:
        ml_result = {"signal": "N/A", "confidence": 0.0, "note": "Skipped (--skip-ml or sklearn not installed)"}
    else:
        try:
            ml_pred = MLPredictor()
            ml_result = ml_pred.train_and_predict(df)
        except Exception as e:
            ml_result = {"signal": "N/A", "confidence": 0.0, "note": str(e)}

    # ── Step 4: LSTM Prediction ──
    if args.skip_lstm or not HAS_TF:
        lstm_result = {"signal": "N/A", "confidence": 0.0, "note": "Skipped (--skip-lstm or tensorflow not installed)"}
    else:
        try:
            lstm_pred = LSTMPredictor()
            lstm_result = lstm_pred.train_and_predict(df)
        except Exception as e:
            lstm_result = {"signal": "N/A", "confidence": 0.0, "note": str(e)}

    # ── Step 5: Sentiment Analysis ──
    if args.skip_sentiment or not HAS_TEXTBLOB:
        sent_result = {"signal": "N/A", "confidence": 0.0, "note": "Skipped (--skip-sentiment or textblob not installed)"}
    else:
        print("\n  💬  Analyzing sentiment ...")
        sent_result = SentimentAnalyzer.analyze(
            DataFetcher.get_ticker(args.symbol, args.market)
        )

    # ── Step 6: Ensemble ──
    ensemble = EnsembleCombiner.combine(tech_signal, ml_result, lstm_result, sent_result)

    # ── Step 7: Report ──
    print_report(args.symbol, args.market, tech_signal, ml_result, lstm_result, sent_result, ensemble, df)

    # ── Step 8: Export ──
    if args.export:
        export_results(args.symbol, tech_signal, ml_result, lstm_result, sent_result, ensemble, args.export)


if __name__ == "__main__":
    main()
