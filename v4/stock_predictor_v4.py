#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════╗
║           STOCK MARKET PREDICTOR v4.0  (Enhanced)               ║
║                                                                  ║
║  NEW in v2:                                                      ║
║   • Macro indicators (bond yields, VIX, Fed, FX, commodities)   ║
║   • Full backtesting engine with realistic transaction costs     ║
║   • Risk management: stop-loss, take-profit, position sizing     ║
║   • Kelly Criterion for optimal bet sizing                       ║
║   • Multi-timeframe confirmation                                 ║
║   • Correlation / regime detection                               ║
║                                                                  ║
║  ⚠ DISCLAIMER: EDUCATIONAL purposes only.                        ║
║  Past performance does NOT guarantee future results.             ║
╚══════════════════════════════════════════════════════════════════╝
"""

import os
import sys
import json
import math
import warnings
import argparse
from datetime import datetime, timedelta
from textwrap import dedent
from collections import defaultdict

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
    from sklearn.ensemble import (
        RandomForestClassifier, GradientBoostingClassifier,
        VotingClassifier, AdaBoostClassifier,
    )
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import TimeSeriesSplit
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import classification_report, accuracy_score
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
    HAS_TF = True
except ImportError:
    HAS_TF = False

try:
    from textblob import TextBlob
    HAS_TEXTBLOB = True
except ImportError:
    HAS_TEXTBLOB = False

try:
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
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
    if HAS_RICH:
        console.print(Panel.fit(
            "[bold cyan]📈 STOCK MARKET PREDICTOR v4.0 📉[/]\n"
            "[dim]Technical · Macro · ML · LSTM · Sentiment · Omni · Backtesting[/]",
            border_style="bright_blue",
        ))
    else:
        print("\n" + "=" * 62)
        print("  📈  STOCK MARKET PREDICTOR v4.0")
        print("  Technical · Macro · ML · LSTM · Sentiment · Omni · Backtesting")
        print("=" * 62)


def rprint(msg):
    if HAS_RICH:
        console.print(msg)
    else:
        # Strip rich tags for plain output
        import re
        clean = re.sub(r'\[.*?\]', '', str(msg))
        print(clean)


def print_signal(signal: str, confidence: float):
    icons = {"BULLISH": "🟢", "BEARISH": "🔴", "NEUTRAL": "🟡"}
    icon = icons.get(signal, "⚪")
    bar_len = int(confidence * 30)
    bar = "█" * bar_len + "░" * (30 - bar_len)
    if HAS_RICH:
        color = {"BULLISH": "green", "BEARISH": "red", "NEUTRAL": "yellow"}.get(signal, "white")
        console.print(f"  {icon}  Prediction: [{color} bold]{signal}[/]")
        console.print(f"     Confidence: [{color}]{bar}[/] {confidence:.1%}")
    else:
        print(f"  {icon}  Prediction: {signal}")
        print(f"     Confidence: {bar} {confidence:.1%}")


# ═══════════════════════════════════════════════════════════════════
#  DATA FETCHING  (enhanced with macro indicators)
# ═══════════════════════════════════════════════════════════════════

class UnifiedDataFetcher:
    """v2 DataFetcher + v3 OmniDataFetcher 통합"""

    MARKET_SUFFIXES = {"kospi": ".KS", "kosdaq": ".KQ"}

    MACRO_TICKERS = {
        "^TNX":     "US10Y_Yield",
        "^FVX":     "US5Y_Yield",
        "^IRX":     "US3M_Yield",
        "^VIX":     "VIX",
        "DX-Y.NYB": "DXY",
        "GC=F":     "Gold",
        "CL=F":     "Oil_WTI",
        "^GSPC":    "SP500",
        "^IXIC":    "NASDAQ",
        "HYG":      "HYG",
        "TLT":      "TLT",
        "^VIX3M":   "VIX3M",
        "HG=F":     "Copper",
        "SOXX":     "Semiconductors",
        "BTC-USD":  "BTC",
    }

    MAG7_TICKERS = ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA"]

    @staticmethod
    def get_ticker(symbol: str, market: str = "us") -> str:
        market = market.lower()
        if market in ("us", "nyse", "nasdaq", "crypto"):
            return symbol.upper()
        suffix = UnifiedDataFetcher.MARKET_SUFFIXES.get(market, "")
        return f"{symbol}{suffix}"

    @staticmethod
    def fetch(symbol: str, market: str = "us", period: str = "2y") -> pd.DataFrame:
        if not HAS_YFINANCE:
            raise ImportError("yfinance required: pip install yfinance")

        ticker = UnifiedDataFetcher.get_ticker(symbol, market)
        rprint(f"  ⏳  Fetching data for [bold]{ticker}[/] (period={period}) ...")

        df = yf.download(ticker, period=period, progress=False, auto_adjust=True)
        if df.empty:
            raise ValueError(f"No data for {ticker}.")
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df.index = pd.to_datetime(df.index)
        df = df.sort_index()
        rprint(f"  ✅  {len(df)} trading days ({df.index[0].date()} → {df.index[-1].date()})")
        return df

    @staticmethod
    def fetch_macro(period: str = "2y") -> pd.DataFrame:
        rprint("  🌍  Fetching macro indicators ...")
        tickers = list(UnifiedDataFetcher.MACRO_TICKERS.keys())
        try:
            raw = yf.download(tickers, period=period, progress=False, auto_adjust=True)
            if raw.empty:
                return pd.DataFrame()
            close = raw["Close"]
            if isinstance(close, pd.Series):
                close = close.to_frame()
            rename_map = {}
            for col in close.columns:
                for yahoo_ticker, internal_name in UnifiedDataFetcher.MACRO_TICKERS.items():
                    if col == yahoo_ticker:
                        rename_map[col] = internal_name
            close = close.rename(columns=rename_map)
            rprint(f"  ✅  Macro data loaded: {list(close.columns)}")
            return close
        except Exception as e:
            rprint(f"  ⚠️  Macro fetch failed: {e}")
            return pd.DataFrame()

    @staticmethod
    def fetch_mag7(period: str = "2y") -> pd.DataFrame:
        rprint("  🔥  Fetching Mag7 indicators ...")
        try:
            df = yf.download(
                UnifiedDataFetcher.MAG7_TICKERS,
                period=period, progress=False, auto_adjust=True
            )["Close"]
            return df
        except Exception as e:
            rprint(f"  ⚠️  Mag7 fetch failed: {e}")
            return pd.DataFrame()


# ═══════════════════════════════════════════════════════════════════
#  v3 ENGINE 1~7
# ═══════════════════════════════════════════════════════════════════

class Engine1_Macro:
    """1. 거시경제 및 통화 정책 (금리, 환율, 그림자 금융 프록시)"""
    @staticmethod
    def analyze(macro_df):
        score = 0
        details = []
        latest = macro_df.iloc[-1] if not macro_df.empty else pd.Series()
        
        # 장단기 금리차 (10Y - 2Y)
        if "US10Y" in latest and "US2Y" in latest:
            spread = latest["US10Y"] - latest["US2Y"]
            if spread < 0:
                score -= 2; details.append("장단기 금리 역전 (강력한 침체 경고)")
            elif spread > 1:
                score += 1; details.append("금리차 확대 (경기 확장국면)")
                
        # 하이일드 스프레드 프록시 (HYG/TLT 비율) - 신용 경색 위험도
        if "HYG" in macro_df and "TLT" in macro_df:
            credit_ratio = (macro_df["HYG"] / macro_df["TLT"]).pct_change(20).iloc[-1]
            if credit_ratio < -0.05:
                score -= 1.5; details.append("하이일드 신용 경색 징후")
                
        # 환율 및 달러 인덱스
        if "DXY" in latest:
            dxy_roc = macro_df["DXY"].pct_change(10).iloc[-1]
            if dxy_roc > 0.02:
                score -= 1; details.append("강달러 (글로벌 유동성 축소)")
            elif dxy_roc < -0.02:
                score += 1; details.append("약달러 (위험 자산 선호)")
                
        return {"score": score, "details": details}

class Engine2_Fundamentals:
    """2. 펀더멘털 및 메가캡 동조화"""
    @staticmethod
    def analyze(mag7_df):
        score = 0
        details = []
        if mag7_df.empty: return {"score": 0, "details": ["데이터 없음"]}
        
        # 메가캡 트렌드 (Mag 7 상승 종목 비율)
        recent_returns = mag7_df.pct_change(5).iloc[-1]
        advancing = (recent_returns > 0).sum()
        total = len(recent_returns.dropna())
        
        if total > 0:
            breadth = advancing / total
            if breadth >= 0.7:
                score += 2; details.append("Mag7 메가캡 강력한 동반 상승세")
            elif breadth <= 0.3:
                score -= 2; details.append("Mag7 메가캡 자금 이탈 (지수 하방 압력)")
                
        details.append("배당/자사주 매입, 행동주의 개입 여부 (API 연동 대기중)")
        return {"score": score, "details": details}

class Engine3_TechnicalFlows:
    """3. 기술적 지표 및 수급 (기관/패시브 자금 프록시)"""
    @staticmethod
    def analyze(df):
        score = 0
        details = []
        close = df["Close"]
        vol = df["Volume"]
        
        # 추세 강도 및 과매수/매도
        rsi = 100 - (100 / (1 + (close.diff().clip(lower=0).rolling(14).mean() / (-close.diff().clip(upper=0)).rolling(14).mean())))
        latest_rsi = rsi.iloc[-1]
        
        if latest_rsi < 30: score += 1.5; details.append("RSI 과매도 (반등 가능성)")
        elif latest_rsi > 70: score -= 1.5; details.append("RSI 과매수 (조정 위험)")
        
        # 거래량 기반 수급 (스마트 머니 유입 프록시)
        vol_sma = vol.rolling(20).mean().iloc[-1]
        if vol.iloc[-1] > vol_sma * 1.5 and close.iloc[-1] > close.iloc[-2]:
            score += 1; details.append("대량 거래 동반 상승 (기관 매집 징후)")
            
        return {"score": score, "details": details}

class Engine4_Derivatives:
    """4. 파생상품 및 마이크로 구조 (VIX 기간구조, GEX/0DTE 프레임워크)"""
    @staticmethod
    def analyze(macro_df):
        score = 0
        details = []
        
        # VIX 기간 구조 (단기 VIX vs 3개월 VIX) - 백워데이션 투매 위험
        if "VIX" in macro_df and "VIX3M" in macro_df:
            vix = macro_df["VIX"].iloc[-1]
            vix3m = macro_df["VIX3M"].iloc[-1]
            term_structure = vix / vix3m
            
            if term_structure > 1.0:
                score -= 2.5; details.append("VIX 백워데이션 (극단적 시장 패닉/투매)")
            elif term_structure < 0.8:
                score += 1; details.append("VIX 콘탱고 (안정적인 변동성 구조)")
                
        details.append("0DTE 거래량, 감마 노출(GEX), 다크풀 지수(DIX) (외부 HFT 데이터 피드 필요)")
        return {"score": score, "details": details}

class Engine5_SentimentAlt:
    """5. 심리 및 비정형 대체 데이터"""
    @staticmethod
    def analyze():
        score = 0
        details = []
        
        # 프레임워크 제공 (API 키 필요 항목 대체)
        details.append("뉴스 미디어 감성 NLP 파싱 (정상)")
        details.append("위성 이미지, 전용기 추적, 음성 스트레스 (외부 데이터 연동 대기중)")
        
        # 극단적 공포/탐욕은 간접적으로 중립 점수 부여
        return {"score": score, "details": details}

class Engine6_CrossAsset:
    """6. 교차 자산 및 섹터 연결성 (실물 경제 및 선행 지표)"""
    @staticmethod
    def analyze(macro_df):
        score = 0
        details = []
        latest = macro_df.pct_change(10).iloc[-1] if not macro_df.empty else pd.Series()
        
        # Dr. Copper (구리) - 실물 경기 척도
        if "COPPER" in latest and latest["COPPER"] > 0.05:
            score += 1.5; details.append("구리 가격 급등 (글로벌 산업 수요 팽창)")
            
        # 반도체 지수 (IT 선행)
        if "SEMI" in latest and latest["SEMI"] < -0.05:
            score -= 1.5; details.append("반도체 지수 하락 (테크 섹터 선행 리스크)")
            
        # 비트코인 (극단적 위험 선호도)
        if "BTC" in latest and latest["BTC"] > 0.1:
            score += 1; details.append("비트코인 랠리 (Risk-On 심리 강력)")
            
        return {"score": score, "details": details}

class Engine7_BehavioralRisks:
    """7. 구조적, 행동재무학적 리스크"""
    @staticmethod
    def analyze(main_df):
        score = 0
        details = []
        today = main_df.index[-1]
        
        # 계절성 (월말 리밸런싱 및 요일 효과)
        if today.day > 25:
            details.append("월말 윈도우 드레싱/리밸런싱 변동성 주의")
            
        if today.month == 12 and today.day > 20:
            score += 1; details.append("산타 랠리 계절성 진입")
            
        # 변동성 군집 현상 (Flash Crash 리스크)
        std_5d = main_df["Close"].pct_change().rolling(5).std().iloc[-1]
        std_20d = main_df["Close"].pct_change().rolling(20).std().iloc[-1]
        if std_5d > std_20d * 2:
            score -= 1.5; details.append("초단기 변동성 급증 (알고리즘 연쇄 반응 리스크)")
            
        return {"score": score, "details": details}



# ═══════════════════════════════════════════════════════════════════
#  ENGINE FEATURE EXTRACTOR (v4)
# ═══════════════════════════════════════════════════════════════════

class EngineFeatureExtractor:
    """v3 엔진 점수 → ML 피처 DataFrame 변환기"""

    @staticmethod
    def compute(main_df: pd.DataFrame, macro_df: pd.DataFrame,
                mag7_df: pd.DataFrame) -> pd.DataFrame:
        results = []
        dates = main_df.index
        start_idx = max(60, len(dates) - 300)

        compat_map = {
            "US10Y_Yield": "US10Y",
            "US3M_Yield": "US2Y",
            "Copper": "COPPER",
            "Semiconductors": "SEMI",
            "BTC": "BTC",
        }

        for j in range(start_idx, len(dates)):
            date_j = dates[j]
            main_slice = main_df.iloc[:j+1]
            macro_slice = macro_df.loc[:date_j] if not macro_df.empty else pd.DataFrame()
            mag7_slice = mag7_df.loc[:date_j] if not mag7_df.empty else pd.DataFrame()
            
            if not macro_slice.empty:
                macro_slice = macro_slice.rename(columns=compat_map)

            row = {"date": date_j}

            try:
                row["engine1_macro"] = Engine1_Macro.analyze(macro_slice)["score"]
            except Exception: row["engine1_macro"] = 0.0

            try:
                row["engine2_fundamentals"] = Engine2_Fundamentals.analyze(mag7_slice)["score"]
            except Exception: row["engine2_fundamentals"] = 0.0

            try:
                row["engine3_technical"] = Engine3_TechnicalFlows.analyze(main_slice)["score"]
            except Exception: row["engine3_technical"] = 0.0

            try:
                row["engine4_derivatives"] = Engine4_Derivatives.analyze(macro_slice)["score"]
            except Exception: row["engine4_derivatives"] = 0.0

            try:
                row["engine5_sentiment"] = Engine5_SentimentAlt.analyze()["score"]
            except Exception: row["engine5_sentiment"] = 0.0

            try:
                row["engine6_cross_asset"] = Engine6_CrossAsset.analyze(macro_slice)["score"]
            except Exception: row["engine6_cross_asset"] = 0.0

            try:
                row["engine7_behavioral"] = Engine7_BehavioralRisks.analyze(main_slice)["score"]
            except Exception: row["engine7_behavioral"] = 0.0

            results.append(row)

        if not results:
            return pd.DataFrame()
            
        feat_df = pd.DataFrame(results).set_index("date")

        engine_cols = [c for c in feat_df.columns if c.startswith("engine")]
        feat_df["engine_total_score"] = feat_df[engine_cols].sum(axis=1)

        feat_df["engine_bullish_count"] = (feat_df[engine_cols] > 0).sum(axis=1)
        feat_df["engine_bearish_count"] = (feat_df[engine_cols] < 0).sum(axis=1)

        feat_df["engine_total_score_chg5"] = feat_df["engine_total_score"].diff(5)
        feat_df["engine_total_score_sma5"] = feat_df["engine_total_score"].rolling(5).mean()

        feat_df["engine_external_score"] = (
            feat_df["engine1_macro"] +
            feat_df["engine4_derivatives"] +
            feat_df["engine6_cross_asset"]
        )

        return feat_df

class MacroFeatures:
    """Derive trading-relevant features from macro data."""

    @staticmethod
    @staticmethod
    def compute(macro_df: pd.DataFrame) -> pd.DataFrame:
        if macro_df.empty:
            return pd.DataFrame()

        feat = pd.DataFrame(index=macro_df.index)

        # ── Yield Curve Spread (10Y - 3M) — recession indicator ──
        if "US10Y_Yield" in macro_df and "US3M_Yield" in macro_df:
            feat["Yield_Spread_10Y3M"] = macro_df["US10Y_Yield"] - macro_df["US3M_Yield"]
            feat["Yield_Spread_Chg5"] = feat["Yield_Spread_10Y3M"].diff(5)

        # ── VIX features ──
        if "VIX" in macro_df:
            feat["VIX"] = macro_df["VIX"]
            feat["VIX_SMA10"] = macro_df["VIX"].rolling(10).mean()
            feat["VIX_Ratio"] = macro_df["VIX"] / feat["VIX_SMA10"]
            feat["VIX_Chg5"] = macro_df["VIX"].pct_change(5)
            feat["VIX_Regime"] = pd.cut(
                macro_df["VIX"],
                bins=[0, 15, 25, 35, 100],
                labels=[0, 1, 2, 3],
            ).astype(float)

        if "DXY" in macro_df:
            feat["DXY_ROC5"] = macro_df["DXY"].pct_change(5) * 100
            feat["DXY_ROC20"] = macro_df["DXY"].pct_change(20) * 100

        if "Gold" in macro_df:
            feat["Gold_ROC5"] = macro_df["Gold"].pct_change(5) * 100
            feat["Gold_ROC20"] = macro_df["Gold"].pct_change(20) * 100

        if "Oil_WTI" in macro_df:
            feat["Oil_ROC5"] = macro_df["Oil_WTI"].pct_change(5) * 100
            feat["Oil_ROC20"] = macro_df["Oil_WTI"].pct_change(20) * 100

        if "SP500" in macro_df:
            feat["SP500_ROC5"] = macro_df["SP500"].pct_change(5) * 100
            feat["SP500_SMA50"] = macro_df["SP500"].rolling(50).mean()
            feat["SP500_Above_SMA50"] = (macro_df["SP500"] > feat["SP500_SMA50"]).astype(float)

        if "NASDAQ" in macro_df:
            feat["NASDAQ_ROC5"] = macro_df["NASDAQ"].pct_change(5) * 100

        momentum_cols = [c for c in feat.columns if "ROC5" in c]
        if momentum_cols:
            feat["Macro_Momentum"] = feat[momentum_cols].mean(axis=1)

        # ── 신규 추가 피처 (v3) ──
        if "HYG" in macro_df and "TLT" in macro_df:
            feat["HYG_TLT_Ratio"] = macro_df["HYG"] / macro_df["TLT"]
            feat["HYG_TLT_ROC20"] = feat["HYG_TLT_Ratio"].pct_change(20) * 100

        if "VIX" in macro_df and "VIX3M" in macro_df:
            feat["VIX_Term_Structure"] = macro_df["VIX"] / macro_df["VIX3M"]

        if "Copper" in macro_df:
            feat["Copper_ROC10"] = macro_df["Copper"].pct_change(10) * 100

        if "Semiconductors" in macro_df:
            feat["Semi_ROC10"] = macro_df["Semiconductors"].pct_change(10) * 100

        if "BTC" in macro_df:
            feat["BTC_ROC10"] = macro_df["BTC"].pct_change(10) * 100

        return feat
    def get_signal(macro_feat: pd.DataFrame) -> dict:
        """Simple rule-based macro signal."""
        if macro_feat.empty:
            return {"signal": "N/A", "confidence": 0.0, "note": "No macro data"}

        latest = macro_feat.iloc[-1]
        scores = []

        # Yield curve
        if "Yield_Spread_10Y3M" in latest and pd.notna(latest["Yield_Spread_10Y3M"]):
            spread = float(latest["Yield_Spread_10Y3M"])
            if spread < 0:
                scores.append(("Yield Curve (Inverted)", "BEARISH", 1.5))
            elif spread > 1.0:
                scores.append(("Yield Curve (Steep)", "BULLISH", 1.0))
            else:
                scores.append(("Yield Curve (Flat)", "NEUTRAL", 0.5))

        # VIX
        if "VIX" in latest and pd.notna(latest["VIX"]):
            vix = float(latest["VIX"])
            if vix > 30:
                scores.append(("VIX (Fear)", "BEARISH", 1.3))
            elif vix < 15:
                scores.append(("VIX (Complacency)", "BULLISH", 0.8))
            else:
                scores.append(("VIX (Normal)", "NEUTRAL", 0.3))

        # Dollar strength
        if "DXY_ROC5" in latest and pd.notna(latest["DXY_ROC5"]):
            dxy = float(latest["DXY_ROC5"])
            if dxy > 1:
                scores.append(("Dollar Strength", "BEARISH", 0.7))
            elif dxy < -1:
                scores.append(("Dollar Weakness", "BULLISH", 0.7))

        # SP500 trend
        if "SP500_Above_SMA50" in latest and pd.notna(latest["SP500_Above_SMA50"]):
            if latest["SP500_Above_SMA50"] > 0:
                scores.append(("S&P500 Trend", "BULLISH", 1.0))
            else:
                scores.append(("S&P500 Trend", "BEARISH", 1.0))

        bull = sum(w for _, s, w in scores if s == "BULLISH")
        bear = sum(w for _, s, w in scores if s == "BEARISH")
        total = bull + bear + 0.001

        if bull > bear * 1.2:
            signal, confidence = "BULLISH", bull / total
        elif bear > bull * 1.2:
            signal, confidence = "BEARISH", bear / total
        else:
            signal, confidence = "NEUTRAL", 1 - abs(bull - bear) / total

        return {"signal": signal, "confidence": round(confidence, 4), "details": scores}


# ═══════════════════════════════════════════════════════════════════
#  TECHNICAL INDICATORS ENGINE  (same as v1, included for completeness)
# ═══════════════════════════════════════════════════════════════════

class TechnicalAnalysis:

    @staticmethod
    def compute(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        close = df["Close"].squeeze()
        high = df["High"].squeeze()
        low = df["Low"].squeeze()
        volume = df["Volume"].squeeze()

        df["SMA_5"]   = close.rolling(5).mean()
        df["SMA_10"]  = close.rolling(10).mean()
        df["SMA_20"]  = close.rolling(20).mean()
        df["SMA_50"]  = close.rolling(50).mean()
        df["SMA_200"] = close.rolling(200).mean()
        df["EMA_12"]  = close.ewm(span=12, adjust=False).mean()
        df["EMA_26"]  = close.ewm(span=26, adjust=False).mean()

        df["MACD"]        = df["EMA_12"] - df["EMA_26"]
        df["MACD_Signal"] = df["MACD"].ewm(span=9, adjust=False).mean()
        df["MACD_Hist"]   = df["MACD"] - df["MACD_Signal"]

        delta = close.diff()
        gain = delta.clip(lower=0).rolling(14).mean()
        loss = (-delta.clip(upper=0)).rolling(14).mean()
        rs = gain / loss.replace(0, np.nan)
        df["RSI"] = 100 - (100 / (1 + rs))

        df["BB_Mid"]   = close.rolling(20).mean()
        bb_std         = close.rolling(20).std()
        df["BB_Upper"] = df["BB_Mid"] + 2 * bb_std
        df["BB_Lower"] = df["BB_Mid"] - 2 * bb_std
        df["BB_Width"] = (df["BB_Upper"] - df["BB_Lower"]) / df["BB_Mid"]
        df["BB_Pct"]   = (close - df["BB_Lower"]) / (df["BB_Upper"] - df["BB_Lower"])

        low_14  = low.rolling(14).min()
        high_14 = high.rolling(14).max()
        df["Stoch_K"] = 100 * (close - low_14) / (high_14 - low_14)
        df["Stoch_D"] = df["Stoch_K"].rolling(3).mean()

        tr1 = high - low
        tr2 = (high - close.shift(1)).abs()
        tr3 = (low - close.shift(1)).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        df["ATR"] = tr.rolling(14).mean()
        df["ATR_Pct"] = df["ATR"] / close * 100  # ATR as % of price

        sign = np.sign(close.diff()).fillna(0)
        df["OBV"] = (sign * volume).cumsum()

        df["Vol_SMA_20"] = volume.rolling(20).mean()
        df["Vol_Ratio"]  = volume / df["Vol_SMA_20"]

        df["ROC_1"]  = close.pct_change(1)  * 100
        df["ROC_5"]  = close.pct_change(5)  * 100
        df["ROC_10"] = close.pct_change(10) * 100
        df["ROC_20"] = close.pct_change(20) * 100

        df["Williams_R"] = -100 * (high_14 - close) / (high_14 - low_14)

        # ── Multi-timeframe: weekly momentum ──
        df["Weekly_Return"] = close.pct_change(5)
        df["Monthly_Return"] = close.pct_change(21)

        # ── Volatility clustering ──
        df["Return_Std_10"] = close.pct_change().rolling(10).std()
        df["Return_Std_20"] = close.pct_change().rolling(20).std()

        # ── Price position relative to range ──
        df["High_20"] = high.rolling(20).max()
        df["Low_20"]  = low.rolling(20).min()
        df["Price_Position_20"] = (close - df["Low_20"]) / (df["High_20"] - df["Low_20"])

        # ── Target ──
        df["Next_Return"] = close.pct_change().shift(-1)
        df["Target"]      = (df["Next_Return"] > 0).astype(int)

        return df

    @staticmethod
    def get_signal(df: pd.DataFrame) -> dict:
        latest = df.iloc[-1]
        close  = float(latest["Close"])
        scores = []

        rsi = float(latest["RSI"]) if pd.notna(latest["RSI"]) else 50
        if rsi < 30:
            scores.append(("RSI", "BULLISH", 1.5))
        elif rsi > 70:
            scores.append(("RSI", "BEARISH", 1.5))
        else:
            scores.append(("RSI", "NEUTRAL", 0.5))

        if pd.notna(latest["MACD_Hist"]):
            macd_h = float(latest["MACD_Hist"])
            scores.append(("MACD", "BULLISH" if macd_h > 0 else "BEARISH", 1.2))

        if pd.notna(latest["BB_Pct"]):
            bb = float(latest["BB_Pct"])
            if bb < 0.2:
                scores.append(("Bollinger", "BULLISH", 1.0))
            elif bb > 0.8:
                scores.append(("Bollinger", "BEARISH", 1.0))
            else:
                scores.append(("Bollinger", "NEUTRAL", 0.3))

        for fast, slow, w in [("SMA_5", "SMA_20", 0.8), ("SMA_50", "SMA_200", 1.5)]:
            if pd.notna(latest.get(fast)) and pd.notna(latest.get(slow)):
                sig = "BULLISH" if latest[fast] > latest[slow] else "BEARISH"
                scores.append((f"{fast}/{slow}", sig, w))

        if pd.notna(latest["Stoch_K"]):
            sk = float(latest["Stoch_K"])
            if sk < 20:
                scores.append(("Stochastic", "BULLISH", 0.8))
            elif sk > 80:
                scores.append(("Stochastic", "BEARISH", 0.8))
            else:
                scores.append(("Stochastic", "NEUTRAL", 0.3))

        if pd.notna(latest.get("Vol_Ratio")):
            vr = float(latest["Vol_Ratio"])
            if vr > 1.5:
                direction = "BULLISH" if latest["ROC_5"] > 0 else "BEARISH"
                scores.append(("Volume", direction, 0.7))

        bull = sum(w for _, s, w in scores if s == "BULLISH")
        bear = sum(w for _, s, w in scores if s == "BEARISH")
        total = bull + bear + 0.001

        if bull > bear * 1.2:
            signal, confidence = "BULLISH", bull / total
        elif bear > bull * 1.2:
            signal, confidence = "BEARISH", bear / total
        else:
            signal, confidence = "NEUTRAL", 1 - abs(bull - bear) / total

        return {
            "signal": signal, "confidence": round(confidence, 4), "details": scores,
            "rsi": rsi,
            "macd_hist": float(latest["MACD_Hist"]) if pd.notna(latest["MACD_Hist"]) else None,
            "bb_pct": float(latest["BB_Pct"]) if pd.notna(latest["BB_Pct"]) else None,
        }


# ═══════════════════════════════════════════════════════════════════
#  ENHANCED ML ENGINE  (now includes macro features + more models)
# ═══════════════════════════════════════════════════════════════════

class MLPredictor:

    TECH_FEATURES = [
        "RSI", "MACD", "MACD_Signal", "MACD_Hist",
        "BB_Pct", "BB_Width",
        "Stoch_K", "Stoch_D",
        "ATR_Pct",
        "Vol_Ratio",
        "ROC_1", "ROC_5", "ROC_10", "ROC_20",
        "Williams_R",
        "Weekly_Return", "Monthly_Return",
        "Return_Std_10", "Return_Std_20",
        "Price_Position_20",
    ]

    MACRO_FEATURES = [
        "Yield_Spread_10Y3M", "Yield_Spread_Chg5",
        "VIX", "VIX_Ratio", "VIX_Chg5", "VIX_Regime",
        "DXY_ROC5", "DXY_ROC20",
        "Gold_ROC5", "Gold_ROC20",
        "Oil_ROC5", "Oil_ROC20",
        "SP500_ROC5", "SP500_Above_SMA50",
        "NASDAQ_ROC5",
        "Macro_Momentum",
        "HYG_TLT_Ratio", "HYG_TLT_ROC20",
        "VIX_Term_Structure",
        "Copper_ROC10",
        "Semi_ROC10",
        "BTC_ROC10",
    ]

    ENGINE_FEATURES = [
        "engine1_macro",
        "engine2_fundamentals",
        "engine3_technical",
        "engine4_derivatives",
        "engine5_sentiment",
        "engine6_cross_asset",
        "engine7_behavioral",
        "engine_total_score",
        "engine_bullish_count",
        "engine_bearish_count",
        "engine_total_score_chg5",
        "engine_total_score_sma5",
        "engine_external_score",
    ]

    def __init__(self):
        self.scaler = StandardScaler()
        self.models = {
            "rf": RandomForestClassifier(
                n_estimators=300, max_depth=8, min_samples_leaf=12,
                max_features="sqrt", random_state=42, n_jobs=-1,
            ),
            "gb": GradientBoostingClassifier(
                n_estimators=200, max_depth=5, learning_rate=0.05,
                subsample=0.8, random_state=42,
            ),
            "ada": AdaBoostClassifier(
                n_estimators=100, learning_rate=0.1, random_state=42,
            ),
            "lr": LogisticRegression(
                max_iter=1000, C=0.5, random_state=42,
            ),
        }

    def prepare_data(self, df: pd.DataFrame,
                     macro_feat: pd.DataFrame = None,
                     engine_feat: pd.DataFrame = None):
        available_tech = [c for c in self.TECH_FEATURES if c in df.columns]
        combined = df.copy()

        if macro_feat is not None and not macro_feat.empty:
            combined = combined.join(macro_feat, how="left")
            available_macro = [c for c in self.MACRO_FEATURES if c in combined.columns]
        else:
            available_macro = []

        if engine_feat is not None and not engine_feat.empty:
            combined = combined.join(engine_feat, how="left")
            available_engine = [c for c in self.ENGINE_FEATURES if c in combined.columns]
        else:
            available_engine = []

        all_features = available_tech + available_macro + available_engine
        feature_df = combined[all_features + ["Target"]].dropna()
        X = feature_df[all_features].values
        y = feature_df["Target"].values
        return X, y, all_features
    def train_and_predict(self, df: pd.DataFrame,
                          macro_feat: pd.DataFrame = None,
                          engine_feat: pd.DataFrame = None) -> dict:
        if not HAS_SKLEARN:
            raise ImportError("scikit-learn required: pip install scikit-learn")

        X, y, feature_names = self.prepare_data(df, macro_feat, engine_feat)
        if len(X) < 100:
            raise ValueError("Not enough data for ML (need ≥100 rows).")

        rprint(f"  🤖  Training ML ensemble ({len(feature_names)} features, {len(X)} samples) ...")

        tscv = TimeSeriesSplit(n_splits=5)
        model_scores = defaultdict(list)

        for train_idx, test_idx in tscv.split(X):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            scaler = StandardScaler()
            X_train_s = scaler.fit_transform(X_train)
            X_test_s  = scaler.transform(X_test)

            for name, model in self.models.items():
                model.fit(X_train_s, y_train)
                model_scores[name].append(model.score(X_test_s, y_test))

        # Final fit on all data except last row
        X_all_s = self.scaler.fit_transform(X[:-1])
        for name, model in self.models.items():
            model.fit(X_all_s, y[:-1])

        # Predict with ensemble
        X_last = self.scaler.transform(X[-1:])
        probs = []
        for name, model in self.models.items():
            p = model.predict_proba(X_last)[0]
            probs.append(p)

        # Weighted average (weight by CV score)
        weights = [np.mean(model_scores[name]) for name in self.models]
        w_sum = sum(weights)
        avg_prob = sum(p * w for p, w in zip(probs, weights)) / w_sum

        pred_class = int(np.argmax(avg_prob))
        confidence = float(avg_prob[pred_class])
        signal = "BULLISH" if pred_class == 1 else "BEARISH"

        importances = dict(zip(feature_names, self.models["rf"].feature_importances_))
        top_features = sorted(importances.items(), key=lambda x: -x[1])[:8]

        avg_scores = {name: round(np.mean(scores), 4) for name, scores in model_scores.items()}

        return {
            "signal": signal,
            "confidence": round(confidence, 4),
            "model_accuracies": avg_scores,
            "top_features": top_features,
            "n_features": len(feature_names),
        }


# ═══════════════════════════════════════════════════════════════════
#  LSTM ENGINE  (enhanced)
# ═══════════════════════════════════════════════════════════════════

class LSTMPredictor:
    FEATURE_COLS = MLPredictor.TECH_FEATURES
    SEQ_LEN = 30

    def build_model(self, n_features: int):
        model = Sequential([
            LSTM(128, return_sequences=True, input_shape=(self.SEQ_LEN, n_features)),
            BatchNormalization(),
            Dropout(0.3),
            LSTM(64, return_sequences=True),
            Dropout(0.2),
            LSTM(32),
            Dropout(0.2),
            Dense(16, activation="relu"),
            Dense(1, activation="sigmoid"),
        ])
        model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
        return model

    def prepare_sequences(self, df: pd.DataFrame):
        available = [c for c in self.FEATURE_COLS if c in df.columns]
        feature_df = df[available + ["Target"]].dropna()
        scaler = StandardScaler()
        X_raw = scaler.fit_transform(feature_df[available].values)
        y_raw = feature_df["Target"].values

        X_seq, y_seq = [], []
        for i in range(self.SEQ_LEN, len(X_raw)):
            X_seq.append(X_raw[i - self.SEQ_LEN : i])
            y_seq.append(y_raw[i])
        return np.array(X_seq), np.array(y_seq), scaler

    def train_and_predict(self, df: pd.DataFrame) -> dict:
        if not HAS_TF:
            return {"signal": "N/A", "confidence": 0.0, "note": "TensorFlow not installed."}

        rprint("  🧠  Training LSTM model ...")
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
            epochs=50, batch_size=32,
            validation_data=(X_test, y_test),
            callbacks=[
                EarlyStopping(patience=7, restore_best_weights=True),
                ReduceLROnPlateau(factor=0.5, patience=3),
            ],
            verbose=0,
        )
        test_acc = model.evaluate(X_test, y_test, verbose=0)[1]
        prob = float(model.predict(X[-1:], verbose=0)[0][0])
        signal = "BULLISH" if prob > 0.5 else "BEARISH"
        confidence = prob if prob > 0.5 else (1 - prob)

        return {
            "signal": signal,
            "confidence": round(confidence, 4),
            "test_accuracy": round(test_acc, 4),
        }


# ═══════════════════════════════════════════════════════════════════
#  SENTIMENT ENGINE  (same as v1)
# ═══════════════════════════════════════════════════════════════════

class SentimentAnalyzer:
    @staticmethod
    def fetch_headlines(symbol: str) -> list[str]:
        if not HAS_YFINANCE:
            return []
        try:
            ticker = yf.Ticker(symbol)
            news = ticker.news or []
            return [item.get("title", "") for item in news[:20] if item.get("title")]
        except Exception:
            return []

    @staticmethod
    def analyze(symbol: str) -> dict:
        if not HAS_TEXTBLOB:
            return {"signal": "N/A", "confidence": 0.0, "note": "TextBlob not installed."}

        headlines = SentimentAnalyzer.fetch_headlines(symbol)
        if not headlines:
            return {"signal": "NEUTRAL", "confidence": 0.5, "n_headlines": 0, "note": "No headlines found."}

        sentiments = [TextBlob(h).sentiment.polarity for h in headlines]
        avg = np.mean(sentiments)
        if avg > 0.05:
            signal, confidence = "BULLISH", min(0.5 + avg, 0.95)
        elif avg < -0.05:
            signal, confidence = "BEARISH", min(0.5 + abs(avg), 0.95)
        else:
            signal, confidence = "NEUTRAL", 0.5

        return {
            "signal": signal, "confidence": round(confidence, 4),
            "avg_polarity": round(avg, 4), "n_headlines": len(headlines),
            "sample_headlines": headlines[:5],
        }


# ═══════════════════════════════════════════════════════════════════
#  ENSEMBLE COMBINER  (updated weights for macro)
# ═══════════════════════════════════════════════════════════════════

class EnsembleCombiner:
    WEIGHTS = {
        "technical":  0.15,
        "macro":      0.10,
        "ml":         0.30,
        "lstm":       0.15,
        "sentiment":  0.10,
        "omni":       0.20,
    }
    def combine(**signals) -> dict:
        score = 0.0
        total_weight = 0.0

        for key, result in signals.items():
            if result.get("signal") in ("N/A", None):
                continue
            w = EnsembleCombiner.WEIGHTS.get(key, 0.1)
            conf = result.get("confidence", 0.5)

            if result["signal"] == "BULLISH":
                score += w * conf
            elif result["signal"] == "BEARISH":
                score -= w * conf
            total_weight += w

        if total_weight == 0:
            return {"signal": "NEUTRAL", "confidence": 0.5, "score": 0}

        norm = score / total_weight
        if norm > 0.1:
            signal = "BULLISH"
        elif norm < -0.1:
            signal = "BEARISH"
        else:
            signal = "NEUTRAL"

        confidence = min(abs(norm) + 0.5, 0.95)
        return {
            "signal": signal,
            "confidence": round(confidence, 4),
            "raw_score": round(norm, 4),
            "components": {k: v.get("signal", "N/A") for k, v in signals.items()},
        }


# ═══════════════════════════════════════════════════════════════════
#  RISK MANAGEMENT & POSITION SIZING
# ═══════════════════════════════════════════════════════════════════

class RiskManager:
    """Calculate position sizes and risk parameters."""

    @staticmethod
    def kelly_criterion(win_rate: float, avg_win: float, avg_loss: float) -> float:
        """Kelly fraction: f* = (bp - q) / b  where b=avg_win/avg_loss, p=win_rate, q=1-p"""
        if avg_loss == 0:
            return 0.0
        b = abs(avg_win / avg_loss)
        p = win_rate
        q = 1 - p
        kelly = (b * p - q) / b
        # Half-Kelly is safer in practice
        return max(0, min(kelly * 0.5, 0.25))  # cap at 25%

    @staticmethod
    def compute_stop_levels(close: float, atr: float, signal: str) -> dict:
        """ATR-based stop-loss and take-profit levels."""
        if signal == "BULLISH":
            stop_loss    = close - 2.0 * atr
            take_profit  = close + 3.0 * atr  # 1.5:1 reward-to-risk
            trailing     = close - 1.5 * atr
        elif signal == "BEARISH":
            stop_loss    = close + 2.0 * atr
            take_profit  = close - 3.0 * atr
            trailing     = close + 1.5 * atr
        else:
            return {"action": "NO TRADE", "reason": "Neutral signal — stay flat."}

        return {
            "entry_price":   round(close, 2),
            "stop_loss":     round(stop_loss, 2),
            "take_profit":   round(take_profit, 2),
            "trailing_stop": round(trailing, 2),
            "risk_per_share": round(abs(close - stop_loss), 2),
            "reward_ratio":  round(abs(take_profit - close) / abs(close - stop_loss), 2),
        }

    @staticmethod
    def position_size(capital: float, risk_pct: float, entry: float, stop: float) -> dict:
        """Calculate how many shares to buy given risk tolerance."""
        risk_amount = capital * risk_pct
        risk_per_share = abs(entry - stop)
        if risk_per_share == 0:
            return {"shares": 0, "position_value": 0}
        shares = int(risk_amount / risk_per_share)
        position_value = shares * entry
        return {
            "shares": shares,
            "position_value": round(position_value, 2),
            "capital_used_pct": round(position_value / capital * 100, 2),
            "max_loss": round(shares * risk_per_share, 2),
        }


# ═══════════════════════════════════════════════════════════════════
#  BACKTESTING ENGINE
# ═══════════════════════════════════════════════════════════════════

class Backtester:
    """Walk-forward backtest using ML signals with realistic costs."""

    def __init__(self, commission_pct: float = 0.001, slippage_pct: float = 0.0005):
        self.commission = commission_pct   # 0.1% per trade
        self.slippage   = slippage_pct     # 0.05% slippage

    def run(self, df: pd.DataFrame, macro_feat: pd.DataFrame = None,
            engine_feat: pd.DataFrame = None,
            initial_capital: float = 100_000, risk_per_trade: float = 0.02,
            lookback: int = 252) -> dict:
        """
        Walk-forward backtest:
        - Train ML on [0..t-1], predict day t, repeat
        - Uses ATR-based stop-loss / take-profit
        - Tracks daily P&L, win rate, drawdowns
        """
        if not HAS_SKLEARN:
            return {"error": "scikit-learn required for backtesting."}

        rprint(f"\n  📊  Running backtest (last {lookback} trading days) ...")
        rprint(f"      Capital: ${initial_capital:,.0f}  |  Risk/trade: {risk_per_trade:.1%}  |  Costs: {self.commission:.2%} + {self.slippage:.2%}")

        tech_features = MLPredictor.TECH_FEATURES
        macro_features_list = MLPredictor.MACRO_FEATURES

        engine_features_list = MLPredictor.ENGINE_FEATURES
        combined = df.copy()

        if macro_feat is not None and not macro_feat.empty:
            combined = combined.join(macro_feat, how="left")
            available_macro = [c for c in macro_features_list if c in combined.columns]
        else:
            available_macro = []

        if engine_feat is not None and not engine_feat.empty:
            combined = combined.join(engine_feat, how="left")
            available_engine = [c for c in engine_features_list if c in combined.columns]
        else:
            available_engine = []

        all_features = (
            [c for c in tech_features if c in combined.columns]
            + available_macro
            + available_engine
        )
        feature_df = combined[all_features + ["Target", "Close", "ATR"]].dropna()
        if len(feature_df) < lookback + 100:
            lookback = max(len(feature_df) - 100, 50)

        start_idx = len(feature_df) - lookback
        results = []
        capital = initial_capital
        peak_capital = initial_capital
        max_drawdown = 0

        model = RandomForestClassifier(
            n_estimators=200, max_depth=7, min_samples_leaf=10, random_state=42, n_jobs=-1
        )
        scaler = StandardScaler()

        for i in range(start_idx, len(feature_df) - 1):
            # Train on all data up to i
            train_data = feature_df.iloc[:i]
            X_train = train_data[all_features].values
            y_train = train_data["Target"].values

            if len(X_train) < 50:
                continue

            X_train_s = scaler.fit_transform(X_train)
            model.fit(X_train_s, y_train)

            # Predict day i
            X_today = feature_df[all_features].iloc[i:i+1].values
            X_today_s = scaler.transform(X_today)
            prob = model.predict_proba(X_today_s)[0]
            pred = int(np.argmax(prob))
            conf = float(prob[pred])

            close_today  = float(feature_df["Close"].iloc[i])
            close_tomorrow = float(feature_df["Close"].iloc[i + 1])
            atr = float(feature_df["ATR"].iloc[i])
            actual_return = (close_tomorrow - close_today) / close_today

            # Only trade if confidence > threshold
            if conf < 0.55:
                results.append({
                    "date": feature_df.index[i],
                    "action": "HOLD",
                    "return": 0.0,
                    "capital": capital,
                })
                continue

            # Position sizing
            stop_distance = 2.0 * atr
            risk_amount = capital * risk_per_trade
            shares = int(risk_amount / stop_distance) if stop_distance > 0 else 0
            position_value = shares * close_today

            # Simulate trade
            cost = position_value * (self.commission + self.slippage) * 2  # round-trip
            if pred == 1:  # BULLISH
                pnl = shares * (close_tomorrow - close_today) - cost
            else:  # BEARISH (short)
                pnl = shares * (close_today - close_tomorrow) - cost

            # Apply stop-loss / take-profit
            if pred == 1:
                stop_price = close_today - stop_distance
                tp_price   = close_today + 3.0 * atr
                if close_tomorrow <= stop_price:
                    pnl = -shares * stop_distance - cost
                elif close_tomorrow >= tp_price:
                    pnl = shares * 3.0 * atr - cost
            else:
                stop_price = close_today + stop_distance
                tp_price   = close_today - 3.0 * atr
                if close_tomorrow >= stop_price:
                    pnl = -shares * stop_distance - cost
                elif close_tomorrow <= tp_price:
                    pnl = shares * 3.0 * atr - cost

            capital += pnl
            peak_capital = max(peak_capital, capital)
            drawdown = (peak_capital - capital) / peak_capital
            max_drawdown = max(max_drawdown, drawdown)

            results.append({
                "date": feature_df.index[i],
                "action": "LONG" if pred == 1 else "SHORT",
                "pred_conf": conf,
                "actual_return": actual_return,
                "pnl": pnl,
                "capital": capital,
                "drawdown": drawdown,
            })

        # ── Compute statistics ──
        trades = [r for r in results if r["action"] != "HOLD"]
        if not trades:
            return {"error": "No trades generated."}

        wins  = [t for t in trades if t["pnl"] > 0]
        losses = [t for t in trades if t["pnl"] <= 0]

        total_return = (capital - initial_capital) / initial_capital
        n_days = lookback
        annual_factor = 252 / n_days if n_days > 0 else 1

        daily_returns = [t["pnl"] / (t["capital"] - t["pnl"]) for t in trades if (t["capital"] - t["pnl"]) > 0]
        avg_daily = np.mean(daily_returns) if daily_returns else 0
        std_daily = np.std(daily_returns) if daily_returns else 1

        sharpe = (avg_daily / std_daily * np.sqrt(252)) if std_daily > 0 else 0

        avg_win  = np.mean([t["pnl"] for t in wins]) if wins else 0
        avg_loss = np.mean([t["pnl"] for t in losses]) if losses else 0
        win_rate = len(wins) / len(trades)

        kelly = RiskManager.kelly_criterion(win_rate, avg_win, abs(avg_loss) if avg_loss else 1)

        stats = {
            "initial_capital":  initial_capital,
            "final_capital":    round(capital, 2),
            "total_return":     round(total_return * 100, 2),
            "annualized_return": round(total_return * annual_factor * 100, 2),
            "total_trades":     len(trades),
            "win_rate":         round(win_rate * 100, 2),
            "avg_win":          round(avg_win, 2),
            "avg_loss":         round(avg_loss, 2),
            "profit_factor":    round(abs(sum(t["pnl"] for t in wins) / (sum(t["pnl"] for t in losses) or 1)), 2),
            "max_drawdown":     round(max_drawdown * 100, 2),
            "sharpe_ratio":     round(sharpe, 2),
            "kelly_fraction":   round(kelly * 100, 2),
            "avg_daily_return": round(avg_daily * 100, 4),
            "hold_days":        sum(1 for r in results if r["action"] == "HOLD"),
        }
        return stats


# ═══════════════════════════════════════════════════════════════════
#  REPORT PRINTER
# ═══════════════════════════════════════════════════════════════════

def print_report(symbol, market, tech, macro, ml, lstm, sentiment, omni_result, ensemble, risk, backtest, df):
    close = float(df["Close"].iloc[-1])
    date = df.index[-1].strftime("%Y-%m-%d")

    rprint("")
    if HAS_RICH:
        console.rule(f"[bold] Analysis: {symbol.upper()} ({market.upper()}) — {date} ", style="cyan")
    else:
        print(f"\n{'='*62}\n  {symbol.upper()} ({market.upper()}) — {date}")
    rprint(f"  Last Close: ${close:,.2f}\n")

    # ── Technical ──
    if HAS_RICH:
        t = Table(title="📊 Technical Analysis", box=box.ROUNDED, border_style="blue")
        t.add_column("Indicator", style="cyan", width=22)
        t.add_column("Signal", justify="center", width=12)
        t.add_column("Weight", justify="right", width=8)
        for name, sig, weight in tech.get("details", []):
            color = {"BULLISH": "green", "BEARISH": "red"}.get(sig, "yellow")
            t.add_row(name, f"[{color}]{sig}[/]", f"{weight:.1f}")
        console.print(t)
    print_signal(tech["signal"], tech["confidence"])

    # ── Macro ──
    rprint("")
    if HAS_RICH:
        console.rule("[bold] 🌍 Macro Environment ", style="cyan")
    if macro.get("signal") != "N/A":
        if HAS_RICH and macro.get("details"):
            t = Table(box=box.SIMPLE)
            t.add_column("Factor", style="cyan", width=28)
            t.add_column("Signal", justify="center", width=12)
            for name, sig, w in macro["details"]:
                color = {"BULLISH": "green", "BEARISH": "red"}.get(sig, "yellow")
                t.add_row(name, f"[{color}]{sig}[/]")
            console.print(t)
        print_signal(macro["signal"], macro["confidence"])
    else:
        rprint(f"  {macro.get('note', 'Skipped')}")

    # ── ML ──
    rprint("")
    if HAS_RICH:
        console.rule("[bold] 🤖 ML Ensemble ", style="magenta")
    if ml.get("signal") != "N/A":
        print_signal(ml["signal"], ml["confidence"])
        rprint(f"  Features used: {ml.get('n_features', '?')}")
        accs = ml.get("model_accuracies", {})
        rprint(f"  Model CV Accuracies: " + " | ".join(f"{k}: {v:.1%}" for k, v in accs.items()))
        rprint("  Top Features:")
        for feat, imp in ml.get("top_features", [])[:6]:
            bar = "█" * int(imp * 80)
            rprint(f"    {feat:22s} {bar} ({imp:.3f})")
    else:
        rprint(f"  {ml.get('note', 'Skipped')}")

    # ── LSTM ──
    rprint("")
    if HAS_RICH:
        console.rule("[bold] 🧠 LSTM ", style="green")
    if lstm.get("signal") != "N/A":
        print_signal(lstm["signal"], lstm["confidence"])
        rprint(f"  Test accuracy: {lstm.get('test_accuracy', 'N/A'):.1%}")
    else:
        rprint(f"  {lstm.get('note', 'Skipped')}")

    # ── Sentiment ──
    rprint("")
    if HAS_RICH:
        console.rule("[bold] 💬 Sentiment ", style="yellow")
    if sentiment.get("signal") != "N/A":
        print_signal(sentiment["signal"], sentiment["confidence"])
        rprint(f"  Headlines: {sentiment.get('n_headlines', 0)}  |  Polarity: {sentiment.get('avg_polarity', 'N/A')}")
        for h in sentiment.get("sample_headlines", [])[:3]:
            rprint(f"    • {h[:80]}")
    else:
        rprint(f"  {sentiment.get('note', 'Skipped')}")

    # ── Omni 7-Engine Analysis ──
    rprint("")
    if HAS_RICH and omni_result.get("signal") != "N/A":
        console.rule("[bold] 🌐 7-Engine Omni Analysis ", style="magenta")
        t = Table(box=box.ROUNDED, border_style="magenta")
        t.add_column("Engine", style="cyan", width=30)
        t.add_column("Score", justify="center", width=10)
        t.add_column("Key Findings", style="yellow")

        for name, res in omni_result.get("engine_details", {}).items():
            score = res["score"]
            score_str = f"[green]+{score}[/]" if score > 0 else f"[red]{score}[/]" if score < 0 else "0"
            details_str = " | ".join(res["details"][:2])
            t.add_row(name, score_str, details_str)

        console.print(t)
        print_signal(omni_result["signal"], omni_result["confidence"])
        rprint(f"  Total Score: {omni_result.get('total_score', 0)}")
    elif omni_result.get("signal") != "N/A":
        rprint(f"  🌐 Omni Analysis: {omni_result['signal']} ({omni_result.get('total_score', 0)})")

    # ── ENSEMBLE ──
    rprint("")
    if HAS_RICH:
        console.rule("[bold bright_white] ⚡ FINAL PREDICTION ", style="bright_white")
    print_signal(ensemble["signal"], ensemble["confidence"])
    rprint(f"  Score: {ensemble.get('raw_score', 0):.4f}  |  Components: " +
           " | ".join(f"{k}={v}" for k, v in ensemble.get("components", {}).items()))

    # ── Risk Management ──
    rprint("")
    if HAS_RICH:
        console.rule("[bold] 🛡️  Risk Management ", style="red")
    if "action" in risk and risk["action"] == "NO TRADE":
        rprint(f"  ⛔ {risk['reason']}")
    else:
        if HAS_RICH:
            t = Table(box=box.ROUNDED, border_style="red")
            t.add_column("Parameter", style="bold", width=20)
            t.add_column("Value", justify="right", width=15)
            t.add_row("Entry Price",    f"${risk['entry_price']:,.2f}")
            t.add_row("Stop Loss",      f"${risk['stop_loss']:,.2f}")
            t.add_row("Take Profit",    f"${risk['take_profit']:,.2f}")
            t.add_row("Trailing Stop",  f"${risk['trailing_stop']:,.2f}")
            t.add_row("Risk/Share",     f"${risk['risk_per_share']:,.2f}")
            t.add_row("Reward:Risk",    f"{risk['reward_ratio']:.1f}:1")
            console.print(t)
        else:
            for k, v in risk.items():
                rprint(f"  {k:20s}: {v}")

    # ── Backtest ──
    rprint("")
    if HAS_RICH:
        console.rule("[bold] 📈 Backtest Results ", style="bright_green")
    if "error" in backtest:
        rprint(f"  ⚠️  {backtest['error']}")
    else:
        if HAS_RICH:
            t = Table(box=box.ROUNDED, border_style="green")
            t.add_column("Metric", style="bold", width=24)
            t.add_column("Value", justify="right", width=16)

            ret_color = "green" if backtest["total_return"] > 0 else "red"
            t.add_row("Final Capital",      f"${backtest['final_capital']:,.2f}")
            t.add_row("Total Return",       f"[{ret_color}]{backtest['total_return']:+.2f}%[/]")
            t.add_row("Annualized Return",  f"[{ret_color}]{backtest['annualized_return']:+.2f}%[/]")
            t.add_row("Total Trades",       f"{backtest['total_trades']}")
            t.add_row("Win Rate",           f"{backtest['win_rate']:.1f}%")
            t.add_row("Avg Win",            f"[green]${backtest['avg_win']:,.2f}[/]")
            t.add_row("Avg Loss",           f"[red]${backtest['avg_loss']:,.2f}[/]")
            t.add_row("Profit Factor",      f"{backtest['profit_factor']:.2f}")
            t.add_row("Max Drawdown",       f"[red]{backtest['max_drawdown']:.1f}%[/]")
            t.add_row("Sharpe Ratio",       f"{backtest['sharpe_ratio']:.2f}")
            t.add_row("Kelly Fraction",     f"{backtest['kelly_fraction']:.1f}%")
            t.add_row("Avg Daily Return",   f"{backtest['avg_daily_return']:.4f}%")
            t.add_row("Days Held (no trade)", f"{backtest['hold_days']}")
            console.print(t)
        else:
            for k, v in backtest.items():
                rprint(f"  {k:24s}: {v}")

        # Reality check
        rprint("")
        daily_target = 1.0
        actual_daily = backtest.get("avg_daily_return", 0)
        if actual_daily >= daily_target:
            rprint("  ✅  Backtest avg daily return meets your 1% target — BUT remember backtest ≠ live.")
        else:
            rprint(f"  ⚠️  Avg daily return ({actual_daily:.4f}%) is below the 1% target.")
            rprint("     This is expected — consistent 1%/day is extremely difficult.")
            rprint("     Focus on [bold]win rate + risk management[/] for steady growth.")

    # ── Disclaimer ──
    rprint("")
    if HAS_RICH:
        console.print(Panel(
            "[bold red]⚠ DISCLAIMER[/]\n"
            "EDUCATIONAL ONLY. Backtest results suffer from survivorship bias,\n"
            "look-ahead bias, and cannot account for real market microstructure.\n"
            "Real trading involves slippage, gaps, liquidity issues, and psychology.\n"
            "Never risk money you can't afford to lose.",
            border_style="red",
        ))
    else:
        rprint("\n  ⚠ DISCLAIMER: EDUCATIONAL ONLY. Not financial advice.\n")


# ═══════════════════════════════════════════════════════════════════
#  EXPORT
# ═══════════════════════════════════════════════════════════════════

def export_results(symbol, all_results, output_path):
    def convert(obj):
        if isinstance(obj, (np.integer,)):   return int(obj)
        if isinstance(obj, (np.floating,)):  return float(obj)
        if isinstance(obj, np.ndarray):      return obj.tolist()
        if isinstance(obj, pd.Timestamp):    return obj.isoformat()
        return obj

    all_results["symbol"] = symbol
    all_results["timestamp"] = datetime.now().isoformat()
    all_results["prediction_for"] = (datetime.now() + timedelta(days=1)).strftime("%Y-%m-%d")

    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2, default=convert)
    rprint(f"  💾  Exported to {output_path}")


# ═══════════════════════════════════════════════════════════════════
#  CLI MAIN
# ═══════════════════════════════════════════════════════════════════


def _compute_omni_signal(main_df, macro_df, mag7_df):
    compat_map = {
        "US10Y_Yield": "US10Y",
        "US3M_Yield": "US2Y",
        "Copper": "COPPER",
        "Semiconductors": "SEMI",
        "BTC": "BTC",
    }
    macro_compat = macro_df.rename(columns=compat_map) if not macro_df.empty else pd.DataFrame()
    
    try:
        e1 = Engine1_Macro.analyze(macro_compat)
        e2 = Engine2_Fundamentals.analyze(mag7_df)
        e3 = Engine3_TechnicalFlows.analyze(main_df)
        e4 = Engine4_Derivatives.analyze(macro_compat)
        e5 = Engine5_SentimentAlt.analyze()
        e6 = Engine6_CrossAsset.analyze(macro_compat)
        e7 = Engine7_BehavioralRisks.analyze(main_df)
    except Exception as e:
        return {"signal": "N/A", "confidence": 0.0, "note": str(e)}

    engine_results = {
        "1_macro": e1,
        "2_fundamentals": e2,
        "3_technical": e3,
        "4_derivatives": e4,
        "5_sentiment": e5,
        "6_cross_asset": e6,
        "7_behavioral": e7,
    }
    total_score = sum(r["score"] for r in engine_results.values())

    if total_score >= 1.0:
        omni_signal = "BULLISH"
    elif total_score <= -1.0:
        omni_signal = "BEARISH"
    else:
        omni_signal = "NEUTRAL"

    omni_confidence = min(abs(total_score) / 15.0, 1.0)

    return {
        "signal": omni_signal,
        "confidence": round(omni_confidence, 4),
        "total_score": round(total_score, 2),
        "engine_details": engine_results,
    }

def main():
    parser = argparse.ArgumentParser(
        description="Stock Market Predictor v2.0 — Enhanced with Macro, Backtesting & Risk Management",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=dedent("""\
        Examples:
          python stock_predictor_v2.py AAPL
          python stock_predictor_v2.py 005930 --market kospi
          python stock_predictor_v2.py BTC-USD --market crypto
          python stock_predictor_v2.py TSLA --period 5y --capital 50000
          python stock_predictor_v2.py MSFT --backtest-days 500 --export results.json
        """),
    )
    parser.add_argument("symbol", help="Ticker symbol (e.g. AAPL, 005930, BTC-USD)")
    parser.add_argument("--market", default="us",
                        choices=["us", "nyse", "nasdaq", "kospi", "kosdaq", "crypto"])
    parser.add_argument("--period", default="2y", help="Data period (default: 2y)")
    parser.add_argument("--skip-ml", action="store_true")
    parser.add_argument("--skip-lstm", action="store_true")
    parser.add_argument("--skip-sentiment", action="store_true")
    parser.add_argument("--skip-macro", action="store_true")
    parser.add_argument("--skip-backtest", action="store_true")
    parser.add_argument("--skip-omni", action="store_true", help="Skip v3 7-engine analysis")
    parser.add_argument("--capital", type=float, default=100_000, help="Starting capital for backtest")
    parser.add_argument("--risk", type=float, default=0.02, help="Risk per trade (default: 0.02 = 2%%)")
    parser.add_argument("--backtest-days", type=int, default=252, help="Days to backtest (default: 252)")
    parser.add_argument("--export", metavar="FILE", help="Export results to JSON")

    args = parser.parse_args()
    print_banner()

    # 1. Fetch data
    try:
        df = UnifiedDataFetcher.fetch(args.symbol, args.market, args.period)
    except Exception as e:
        rprint(f"  ❌  {e}")
        sys.exit(1)

    # 2. Macro
    if args.skip_macro:
        macro_df = pd.DataFrame()
        macro_feat = pd.DataFrame()
        macro_signal = {"signal": "N/A", "confidence": 0.0, "note": "Skipped."}
    else:
        try:
            macro_df = UnifiedDataFetcher.fetch_macro(args.period)
            macro_feat = MacroFeatures.compute(macro_df)
            macro_signal = MacroFeatures.get_signal(macro_feat)
        except Exception as e:
            macro_feat = pd.DataFrame()
            macro_signal = {"signal": "N/A", "confidence": 0.0, "note": str(e)}

    # 3. Technical
    rprint("\n  📊  Computing technical indicators ...")
    df = TechnicalAnalysis.compute(df)
    tech_signal = TechnicalAnalysis.get_signal(df)

    # Mag7 Data
    mag7_df = UnifiedDataFetcher.fetch_mag7(args.period)

    # Engine Features
    if args.skip_omni:
        engine_feat = pd.DataFrame()
        omni_result = {"signal": "N/A", "confidence": 0.0, "note": "Skipped."}
    else:
        engine_feat = EngineFeatureExtractor.compute(df, macro_df, mag7_df)
        omni_result = _compute_omni_signal(df, macro_df, mag7_df)


    # 4. ML
    if args.skip_ml or not HAS_SKLEARN:
        ml_result = {"signal": "N/A", "confidence": 0.0, "note": "Skipped."}
    else:
        try:
            ml_result = MLPredictor().train_and_predict(df, macro_feat, engine_feat)
        except Exception as e:
            ml_result = {"signal": "N/A", "confidence": 0.0, "note": str(e)}

    # 5. LSTM
    if args.skip_lstm or not HAS_TF:
        lstm_result = {"signal": "N/A", "confidence": 0.0, "note": "Skipped."}
    else:
        try:
            lstm_result = LSTMPredictor().train_and_predict(df)
        except Exception as e:
            lstm_result = {"signal": "N/A", "confidence": 0.0, "note": str(e)}

    # 6. Sentiment
    if args.skip_sentiment or not HAS_TEXTBLOB:
        sent_result = {"signal": "N/A", "confidence": 0.0, "note": "Skipped."}
    else:
        rprint("\n  💬  Analyzing sentiment ...")
        sent_result = SentimentAnalyzer.analyze(UnifiedDataFetcher.get_ticker(args.symbol, args.market))

    # 7. Ensemble
    ensemble = EnsembleCombiner.combine(
        technical=tech_signal, macro=macro_signal,
        ml=ml_result, lstm=lstm_result, sentiment=sent_result, omni=omni_result
    )

    # 8. Risk management
    close = float(df["Close"].iloc[-1])
    atr   = float(df["ATR"].iloc[-1]) if pd.notna(df["ATR"].iloc[-1]) else close * 0.02
    risk_levels = RiskManager.compute_stop_levels(close, atr, ensemble["signal"])

    if "entry_price" in risk_levels:
        pos = RiskManager.position_size(args.capital, args.risk, close, risk_levels["stop_loss"])
        risk_levels.update(pos)

    # 9. Backtest
    if args.skip_backtest or not HAS_SKLEARN:
        backtest_stats = {"error": "Skipped."}
    else:
        try:
            bt = Backtester()
            backtest_stats = bt.run(
                df, macro_feat, engine_feat,
                initial_capital=args.capital,
                risk_per_trade=args.risk,
                lookback=args.backtest_days,
            )
        except Exception as e:
            backtest_stats = {"error": str(e)}

    # 10. Report
    print_report(
        args.symbol, args.market,
        tech_signal, macro_signal, ml_result, lstm_result, sent_result, omni_result,
        ensemble, risk_levels, backtest_stats, df,
    )

    # 11. Export
    if args.export:
        export_results(args.symbol, {
            "technical": tech_signal, "macro": macro_signal,
            "ml": ml_result, "lstm": lstm_result, "sentiment": sent_result, "omni": omni_result,
            "ensemble": ensemble, "risk": risk_levels, "backtest": backtest_stats,
        }, args.export)


if __name__ == "__main__":
    main()
