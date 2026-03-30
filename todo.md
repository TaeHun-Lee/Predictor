# Stock Predictor v4: v2 + v3 통합 설계서

> **목표**: v3의 7엔진 룰 기반 진단 점수를 v2의 ML 파이프라인에 피처로 주입하여,
> 거시적 시장 체제 인식과 정량적 ML 예측을 결합한 단일 시스템을 구축한다.

---

## 1. 아키텍처 개요

```
┌─────────────────────────────────────────────────────────────────┐
│                        v4 통합 파이프라인                         │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌──────────────┐    ┌──────────────────┐                       │
│  │ DataFetcher  │───▶│ OmniDataFetcher  │  ← v3에서 가져옴       │
│  │ (v2 기존)    │    │ (Mag7 + 추가거시) │                       │
│  └──────┬───────┘    └────────┬─────────┘                       │
│         │                     │                                  │
│         ▼                     ▼                                  │
│  ┌──────────────┐    ┌──────────────────┐                       │
│  │ Technical    │    │  Engine1~7       │  ← v3 엔진 그대로 사용  │
│  │ Analysis(v2) │    │  (7개 룰 엔진)   │                       │
│  └──────┬───────┘    └────────┬─────────┘                       │
│         │                     │                                  │
│         │                     ▼                                  │
│         │            ┌──────────────────┐                       │
│         │            │ EngineFeature    │  ← [신규] 엔진→피처 변환│
│         │            │ Extractor        │                       │
│         │            └────────┬─────────┘                       │
│         │                     │                                  │
│         ▼                     ▼                                  │
│  ┌──────────────────────────────────────┐                       │
│  │           MLPredictor (확장)          │                       │
│  │  TECH_FEATURES (20개, v2 기존)       │                       │
│  │ + MACRO_FEATURES (16개, v2 기존)     │                       │
│  │ + ENGINE_FEATURES (14개, v3 신규)    │  ← 총 50개 피처        │
│  └──────────────┬───────────────────────┘                       │
│                 │                                                │
│         ┌───────┼───────┐                                       │
│         ▼       ▼       ▼                                       │
│  ┌─────────┐┌────────┐┌──────────┐                              │
│  │Ensemble ││ Risk   ││Backtester│                              │
│  │Combiner ││Manager ││(v2 기존) │                              │
│  └─────────┘└────────┘└──────────┘                              │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## 2. 파일 구조

```
Predictor/
├── v4/
│   ├── stock_predictor_v4.py      # 메인 통합 파일 (단일 파일)
│   └── requirements.txt           # 의존성 (v2와 동일)
├── v1/ (기존 유지)
├── v2/ (기존 유지)
├── v3/ (기존 유지)
└── todo.md (본 문서)
```

**원칙**: v2의 `stock_predictor_v2.py`를 복사한 뒤 수정한다. v1/v2/v3 코드는 변경하지 않는다.

---

## 3. 단계별 구현 명세

### STEP 1: 데이터 수집 계층 통합 (`UnifiedDataFetcher`)

**목적**: v2의 `DataFetcher` + v3의 `OmniDataFetcher`를 하나로 합쳐, 한 번의 호출로 메인 종목 + v2 거시 + v3 추가 거시(Mag7, 구리, 반도체, VIX3M, 닛케이, BTC) 데이터를 모두 가져온다.

**구현 방법**:

1. v2의 `DataFetcher` 클래스를 그대로 복사한다.
2. `MACRO_TICKERS` 딕셔너리에 v3의 추가 티커를 병합한다.
3. `fetch_mag7()` 정적 메서드를 신규 추가한다.

**코드 명세**:

```python
class UnifiedDataFetcher:
    """v2 DataFetcher + v3 OmniDataFetcher 통합"""

    MARKET_SUFFIXES = {"kospi": ".KS", "kosdaq": ".KQ"}

    # v2 기존 9개 + v3 추가 5개 = 총 14개 거시 티커
    MACRO_TICKERS = {
        # --- v2 기존 ---
        "^TNX":     "US10Y_Yield",
        "^FVX":     "US5Y_Yield",
        "^IRX":     "US3M_Yield",
        "^VIX":     "VIX",
        "DX-Y.NYB": "DXY",
        "GC=F":     "Gold",
        "CL=F":     "Oil_WTI",
        "^GSPC":    "SP500",
        "^IXIC":    "NASDAQ",
        # --- v3 추가 ---
        "HYG":      "HYG",          # 하이일드 채권 ETF
        "TLT":      "TLT",          # 장기국채 ETF
        "^VIX3M":   "VIX3M",        # 3개월 VIX (기간구조용)
        "HG=F":     "Copper",       # 구리 선물 (Dr. Copper)
        "SOXX":     "Semiconductors", # 반도체 ETF
        "BTC-USD":  "BTC",          # 비트코인
    }

    MAG7_TICKERS = ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA"]

    @staticmethod
    def get_ticker(symbol: str, market: str = "us") -> str:
        """v2 DataFetcher.get_ticker 그대로 복사"""
        market = market.lower()
        if market in ("us", "nyse", "nasdaq", "crypto"):
            return symbol.upper()
        suffix = UnifiedDataFetcher.MARKET_SUFFIXES.get(market, "")
        return f"{symbol}{suffix}"

    @staticmethod
    def fetch(symbol: str, market: str = "us", period: str = "2y") -> pd.DataFrame:
        """메인 종목 OHLCV 다운로드. v2 DataFetcher.fetch 그대로."""
        ticker = UnifiedDataFetcher.get_ticker(symbol, market)
        df = yf.download(ticker, period=period, progress=False, auto_adjust=True)
        if df.empty:
            raise ValueError(f"No data for {ticker}.")
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df.index = pd.to_datetime(df.index)
        df = df.sort_index()
        return df

    @staticmethod
    def fetch_macro(period: str = "2y") -> pd.DataFrame:
        """거시 지표 14개 일괄 다운로드. Close 컬럼만 추출."""
        tickers = list(UnifiedDataFetcher.MACRO_TICKERS.keys())
        raw = yf.download(tickers, period=period, progress=False, auto_adjust=True)
        if raw.empty:
            return pd.DataFrame()
        close = raw["Close"]
        if isinstance(close, pd.Series):
            close = close.to_frame()
        # yfinance가 반환하는 컬럼명(야후 티커)을 우리 내부명으로 매핑
        rename_map = {}
        for col in close.columns:
            for yahoo_ticker, internal_name in UnifiedDataFetcher.MACRO_TICKERS.items():
                if col == yahoo_ticker:
                    rename_map[col] = internal_name
        close = close.rename(columns=rename_map)
        return close

    @staticmethod
    def fetch_mag7(period: str = "2y") -> pd.DataFrame:
        """Mag7 종가 데이터 다운로드. 반환: DataFrame(columns=티커명, values=Close)"""
        df = yf.download(
            UnifiedDataFetcher.MAG7_TICKERS,
            period=period, progress=False, auto_adjust=True
        )["Close"]
        return df
```

**검증 기준**: `fetch_macro()` 반환 DataFrame의 컬럼이 정확히 `["US10Y_Yield", "US5Y_Yield", "US3M_Yield", "VIX", "DXY", "Gold", "Oil_WTI", "SP500", "NASDAQ", "HYG", "TLT", "VIX3M", "Copper", "Semiconductors", "BTC"]` (데이터 가용성에 따라 일부 누락 가능)

---

### STEP 2: v3 엔진 1~7 이식

**목적**: v3의 Engine1~7 클래스를 v4 파일 안에 그대로 복사한다. 변경 없음.

**구현 방법**:

v3의 `stock_predictor_v3.py`에서 아래 7개 클래스를 그대로 복사하여 v4 파일에 붙여넣는다:

- `Engine1_Macro` (92~122줄)
- `Engine2_Fundamentals` (124~145줄)
- `Engine3_TechnicalFlows` (147~168줄)
- `Engine4_Derivatives` (170~189줄)
- `Engine5_SentimentAlt` (191~203줄)
- `Engine6_CrossAsset` (205~225줄)
- `Engine7_BehavioralRisks` (227~248줄)

**주의사항**:
- 각 엔진의 `analyze()` 메서드 시그니처와 반환 형식을 절대 변경하지 않는다.
- 반환 형식: `{"score": float, "details": list[str]}`
- `Engine3_TechnicalFlows.analyze(df)`는 `df`에 `"Close"`와 `"Volume"` 컬럼이 필요하다.
- `Engine5_SentimentAlt.analyze()`는 인자 없음.

---

### STEP 3: 엔진 피처 변환기 신규 생성 (`EngineFeatureExtractor`)

**목적**: v3의 7개 엔진 점수를 ML이 학습할 수 있는 수치형 피처 DataFrame으로 변환한다.

**설계 원칙**:
- 각 엔진의 `score` 값을 그대로 피처로 사용한다.
- 엔진 점수의 조합으로 파생 피처를 추가 생성한다.
- 일별로 계산하여 시계열 DataFrame을 반환한다 (백테스팅에서 사용 가능하도록).

**구현 방법**:

```python
class EngineFeatureExtractor:
    """v3 엔진 점수 → ML 피처 DataFrame 변환기"""

    @staticmethod
    def compute(main_df: pd.DataFrame, macro_df: pd.DataFrame,
                mag7_df: pd.DataFrame) -> pd.DataFrame:
        """
        매 거래일에 대해 rolling window 방식으로 엔진 점수를 계산한다.

        Parameters:
            main_df:  메인 종목 OHLCV DataFrame (index=DatetimeIndex)
            macro_df: 거시 지표 DataFrame (UnifiedDataFetcher.fetch_macro 반환값)
            mag7_df:  Mag7 종가 DataFrame (UnifiedDataFetcher.fetch_mag7 반환값)

        Returns:
            DataFrame(index=DatetimeIndex, columns=ENGINE_FEATURE_NAMES)

        구현 로직:
            1. main_df의 각 날짜 i에 대해 (i >= 60, 충분한 lookback 필요):
               - macro_slice = macro_df.loc[:date_i]
               - mag7_slice  = mag7_df.loc[:date_i]
               - main_slice  = main_df.loc[:date_i]
               를 준비한다.
            2. 각 엔진의 analyze()를 호출하여 score를 추출한다.
            3. 결과를 row로 축적하여 DataFrame을 만든다.

        성능 최적화:
            - 전체 날짜에 대해 매번 호출하면 느리므로,
              최근 N일(= backtest_days + 여유분)에 대해서만 계산한다.
            - 기본값: 최근 300 거래일
        """
        results = []
        dates = main_df.index
        # 최소 60거래일의 lookback이 필요 (macro pct_change(20) 등에서 사용)
        start_idx = max(60, len(dates) - 300)

        for i in range(start_idx, len(dates)):
            date_i = dates[i]
            main_slice = main_df.iloc[:i+1]
            macro_slice = macro_df.loc[:date_i] if not macro_df.empty else pd.DataFrame()
            mag7_slice = mag7_df.loc[:date_i] if not mag7_df.empty else pd.DataFrame()

            row = {"date": date_i}

            # 엔진 1: 거시경제
            try:
                e1 = Engine1_Macro.analyze(macro_slice)
                row["engine1_macro"] = e1["score"]
            except Exception:
                row["engine1_macro"] = 0.0

            # 엔진 2: 펀더멘털
            try:
                e2 = Engine2_Fundamentals.analyze(mag7_slice)
                row["engine2_fundamentals"] = e2["score"]
            except Exception:
                row["engine2_fundamentals"] = 0.0

            # 엔진 3: 기술적/수급
            try:
                e3 = Engine3_TechnicalFlows.analyze(main_slice)
                row["engine3_technical"] = e3["score"]
            except Exception:
                row["engine3_technical"] = 0.0

            # 엔진 4: 파생상품
            try:
                e4 = Engine4_Derivatives.analyze(macro_slice)
                row["engine4_derivatives"] = e4["score"]
            except Exception:
                row["engine4_derivatives"] = 0.0

            # 엔진 5: 심리 (정적 — 매일 동일한 0점 반환)
            try:
                e5 = Engine5_SentimentAlt.analyze()
                row["engine5_sentiment"] = e5["score"]
            except Exception:
                row["engine5_sentiment"] = 0.0

            # 엔진 6: 교차자산
            try:
                e6 = Engine6_CrossAsset.analyze(macro_slice)
                row["engine6_cross_asset"] = e6["score"]
            except Exception:
                row["engine6_cross_asset"] = 0.0

            # 엔진 7: 행동/구조적 리스크
            try:
                e7 = Engine7_BehavioralRisks.analyze(main_slice)
                row["engine7_behavioral"] = e7["score"]
            except Exception:
                row["engine7_behavioral"] = 0.0

            results.append(row)

        feat_df = pd.DataFrame(results).set_index("date")

        # --- 파생 피처 ---

        # 7엔진 종합 점수 (v3의 total_score와 동일)
        engine_cols = [c for c in feat_df.columns if c.startswith("engine")]
        feat_df["engine_total_score"] = feat_df[engine_cols].sum(axis=1)

        # 긍정/부정 엔진 수 (방향 합의도)
        feat_df["engine_bullish_count"] = (feat_df[engine_cols] > 0).sum(axis=1)
        feat_df["engine_bearish_count"] = (feat_df[engine_cols] < 0).sum(axis=1)

        # 종합 점수의 5일 변화율 (모멘텀)
        feat_df["engine_total_score_chg5"] = feat_df["engine_total_score"].diff(5)

        # 종합 점수의 5일 이동평균 (평활화)
        feat_df["engine_total_score_sma5"] = feat_df["engine_total_score"].rolling(5).mean()

        # 거시 + 교차자산 합산 (외부 환경 점수)
        feat_df["engine_external_score"] = (
            feat_df["engine1_macro"] +
            feat_df["engine4_derivatives"] +
            feat_df["engine6_cross_asset"]
        )

        return feat_df
```

**반환 피처 목록** (14개):

| # | 피처명 | 설명 | 값 범위 |
|---|--------|------|---------|
| 1 | `engine1_macro` | 거시경제 엔진 점수 | -4.5 ~ +2.0 |
| 2 | `engine2_fundamentals` | 펀더멘털 엔진 점수 | -2.0 ~ +2.0 |
| 3 | `engine3_technical` | 기술적/수급 엔진 점수 | -1.5 ~ +2.5 |
| 4 | `engine4_derivatives` | 파생상품 엔진 점수 | -2.5 ~ +1.0 |
| 5 | `engine5_sentiment` | 심리 엔진 점수 | 현재 항상 0 |
| 6 | `engine6_cross_asset` | 교차자산 엔진 점수 | -1.5 ~ +2.5 |
| 7 | `engine7_behavioral` | 행동/구조적 리스크 점수 | -1.5 ~ +1.0 |
| 8 | `engine_total_score` | 7엔진 합산 점수 | -15 ~ +15 |
| 9 | `engine_bullish_count` | 양수 엔진 개수 | 0 ~ 7 |
| 10 | `engine_bearish_count` | 음수 엔진 개수 | 0 ~ 7 |
| 11 | `engine_total_score_chg5` | 합산 점수 5일 변화 | float |
| 12 | `engine_total_score_sma5` | 합산 점수 5일 이동평균 | float |
| 13 | `engine_external_score` | 외부환경 점수(1+4+6) | float |
| 14 | (engine5_sentiment은 현재 0이므로 실질 13개 활성) | | |

**주의**: `engine5_sentiment`은 v3에서 외부 API 미구현으로 항상 0을 반환한다. ML은 이를 학습 시 자동으로 무시할 것이므로 제거하지 않고 피처로 포함한다 (향후 API 연동 시 즉시 활성화 가능).

---

### STEP 4: MLPredictor 확장

**목적**: v2의 `MLPredictor`에 엔진 피처 14개를 추가 피처로 주입한다.

**변경 사항**:

#### 4-1. `ENGINE_FEATURES` 상수 추가

```python
class MLPredictor:

    TECH_FEATURES = [  # v2 그대로 유지 (20개)
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

    MACRO_FEATURES = [  # v2 그대로 유지 (16개)
        "Yield_Spread_10Y3M", "Yield_Spread_Chg5",
        "VIX", "VIX_Ratio", "VIX_Chg5", "VIX_Regime",
        "DXY_ROC5", "DXY_ROC20",
        "Gold_ROC5", "Gold_ROC20",
        "Oil_ROC5", "Oil_ROC20",
        "SP500_ROC5", "SP500_Above_SMA50",
        "NASDAQ_ROC5",
        "Macro_Momentum",
    ]

    ENGINE_FEATURES = [  # 신규 추가 (14개)
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
```

#### 4-2. `prepare_data()` 메서드 수정

```python
def prepare_data(self, df: pd.DataFrame,
                 macro_feat: pd.DataFrame = None,
                 engine_feat: pd.DataFrame = None):    # ← 인자 추가
    """
    변경점:
    - 기존: df + macro_feat 조인
    - 신규: df + macro_feat + engine_feat 3-way 조인
    """
    available_tech = [c for c in self.TECH_FEATURES if c in df.columns]

    combined = df.copy()

    # 거시 피처 조인
    if macro_feat is not None and not macro_feat.empty:
        combined = combined.join(macro_feat, how="left")
    available_macro = [c for c in self.MACRO_FEATURES if c in combined.columns]

    # 엔진 피처 조인  ← 신규
    if engine_feat is not None and not engine_feat.empty:
        combined = combined.join(engine_feat, how="left")
    available_engine = [c for c in self.ENGINE_FEATURES if c in combined.columns]

    all_features = available_tech + available_macro + available_engine
    feature_df = combined[all_features + ["Target"]].dropna()
    X = feature_df[all_features].values
    y = feature_df["Target"].values
    return X, y, all_features
```

#### 4-3. `train_and_predict()` 메서드 수정

```python
def train_and_predict(self, df: pd.DataFrame,
                      macro_feat: pd.DataFrame = None,
                      engine_feat: pd.DataFrame = None) -> dict:   # ← 인자 추가
    """
    변경점: prepare_data 호출 시 engine_feat 전달
    나머지 로직(모델 학습, 교차검증, 예측)은 v2와 100% 동일
    """
    X, y, feature_names = self.prepare_data(df, macro_feat, engine_feat)
    # ... 이하 v2 코드 그대로 ...
```

**검증 기준**: `feature_names` 리스트에 `engine1_macro` 등 엔진 피처가 포함되어야 하며, `n_features`가 약 50개 (20 + 16 + 14, 데이터 가용성에 따라 변동)여야 한다.

---

### STEP 5: Backtester 확장

**목적**: 백테스팅 시에도 엔진 피처를 ML에 공급한다.

**변경 사항**: `Backtester.run()` 메서드의 시그니처와 피처 준비 로직 수정

```python
class Backtester:

    def run(self, df: pd.DataFrame,
            macro_feat: pd.DataFrame = None,
            engine_feat: pd.DataFrame = None,     # ← 인자 추가
            initial_capital: float = 100_000,
            risk_per_trade: float = 0.02,
            lookback: int = 252) -> dict:
        """
        변경점:
        1. all_features 목록에 엔진 피처 추가
        2. combined DataFrame에 engine_feat 조인
        나머지 walk-forward 로직은 v2와 동일
        """
        tech_features = MLPredictor.TECH_FEATURES
        macro_features_list = MLPredictor.MACRO_FEATURES
        engine_features_list = MLPredictor.ENGINE_FEATURES  # ← 추가

        combined = df.copy()

        if macro_feat is not None and not macro_feat.empty:
            combined = combined.join(macro_feat, how="left")
        available_macro = [c for c in macro_features_list if c in combined.columns]

        # 엔진 피처 조인  ← 추가
        if engine_feat is not None and not engine_feat.empty:
            combined = combined.join(engine_feat, how="left")
        available_engine = [c for c in engine_features_list if c in combined.columns]

        all_features = (
            [c for c in tech_features if c in combined.columns]
            + available_macro
            + available_engine    # ← 추가
        )

        # 이하 v2의 walk-forward 루프 그대로
        # ...
```

**주의**: 엔진 피처는 `EngineFeatureExtractor.compute()`에서 이미 날짜별로 계산되어 있으므로, 백테스트의 walk-forward 루프에서 미래 데이터 누출(look-ahead bias)이 발생하지 않는다. 각 날짜 i의 엔진 피처는 해당 날짜까지의 데이터만으로 계산된 값이다.

---

### STEP 6: EnsembleCombiner 가중치 재설계

**목적**: v3 엔진 진단이 ML에 이미 반영되므로, 앙상블 가중치를 재조정한다. 또한 v3의 종합 점수를 별도 시그널로 앙상블에 참여시킨다.

**변경 사항**:

```python
class EnsembleCombiner:
    WEIGHTS = {
        "technical":  0.15,     # v2: 0.20 → 0.15 (엔진3과 중복 고려)
        "macro":      0.10,     # v2: 0.15 → 0.10 (엔진1과 중복 고려)
        "ml":         0.30,     # v2: 0.30 → 유지 (핵심 예측 모델)
        "lstm":       0.15,     # v2: 0.20 → 0.15
        "sentiment":  0.10,     # v2: 0.15 → 0.10
        "omni":       0.20,     # 신규: v3 7엔진 종합 시그널
    }
```

**`omni` 시그널 생성 로직** (main 함수에서):

```python
# 7엔진 종합 시그널을 v3 방식으로 생성
engine_results = {
    "1_macro": Engine1_Macro.analyze(macro_df_latest),
    "2_fundamentals": Engine2_Fundamentals.analyze(mag7_df_latest),
    "3_technical": Engine3_TechnicalFlows.analyze(main_df),
    "4_derivatives": Engine4_Derivatives.analyze(macro_df_latest),
    "5_sentiment": Engine5_SentimentAlt.analyze(),
    "6_cross_asset": Engine6_CrossAsset.analyze(macro_df_latest),
    "7_behavioral": Engine7_BehavioralRisks.analyze(main_df),
}
total_score = sum(r["score"] for r in engine_results.values())

# v3 기준으로 시그널 변환
if total_score >= 1.0:
    omni_signal = "BULLISH"
elif total_score <= -1.0:
    omni_signal = "BEARISH"
else:
    omni_signal = "NEUTRAL"

omni_confidence = min(abs(total_score) / 15.0, 1.0)

omni_result = {
    "signal": omni_signal,
    "confidence": round(omni_confidence, 4),
    "total_score": round(total_score, 2),
    "engine_details": engine_results,
}
```

---

### STEP 7: MacroFeatures 확장

**목적**: v2의 `MacroFeatures.compute()`에 v3 추가 티커(HYG, TLT, VIX3M, Copper, Semiconductors, BTC)에서 파생되는 피처를 추가한다.

**추가 피처 (기존 16개에 8개 추가 → 총 24개)**:

```python
# MacroFeatures.compute() 내부에 추가할 코드:

# ── HYG/TLT 비율 (신용 경색 프록시, v3 Engine1에서 사용) ──
if "HYG" in macro_df and "TLT" in macro_df:
    feat["HYG_TLT_Ratio"] = macro_df["HYG"] / macro_df["TLT"]
    feat["HYG_TLT_ROC20"] = feat["HYG_TLT_Ratio"].pct_change(20) * 100

# ── VIX 기간구조 (v3 Engine4에서 사용) ──
if "VIX" in macro_df and "VIX3M" in macro_df:
    feat["VIX_Term_Structure"] = macro_df["VIX"] / macro_df["VIX3M"]
    # 1.0 초과 = 백워데이션(패닉), 0.8 미만 = 콘탱고(안정)

# ── 구리 (실물 경기 척도) ──
if "Copper" in macro_df:
    feat["Copper_ROC10"] = macro_df["Copper"].pct_change(10) * 100

# ── 반도체 (테크 선행지표) ──
if "Semiconductors" in macro_df:
    feat["Semi_ROC10"] = macro_df["Semiconductors"].pct_change(10) * 100

# ── BTC (위험선호도) ──
if "BTC" in macro_df:
    feat["BTC_ROC10"] = macro_df["BTC"].pct_change(10) * 100
```

**`MACRO_FEATURES` 상수에도 추가**:

```python
MACRO_FEATURES = [
    # ... 기존 16개 ...
    "HYG_TLT_Ratio", "HYG_TLT_ROC20",    # 신용
    "VIX_Term_Structure",                   # VIX 기간구조
    "Copper_ROC10",                         # 실물경기
    "Semi_ROC10",                           # 테크 선행
    "BTC_ROC10",                            # 위험선호도
]
```

---

### STEP 8: 리포트 출력 확장 (`print_report`)

**목적**: v3 엔진 분석 결과를 별도 섹션으로 출력한다.

**추가할 리포트 섹션** (Ensemble 섹션 바로 앞에 삽입):

```python
# ── Omni 7-Engine Analysis ──
if HAS_RICH:
    console.rule("[bold] 🌐 7-Engine Omni Analysis ", style="magenta")
    t = Table(box=box.ROUNDED, border_style="magenta")
    t.add_column("Engine", style="cyan", width=30)
    t.add_column("Score", justify="center", width=10)
    t.add_column("Key Findings", style="yellow")

    for name, res in omni_result["engine_details"].items():
        score = res["score"]
        score_str = f"[green]+{score}[/]" if score > 0 else f"[red]{score}[/]" if score < 0 else "0"
        details_str = " | ".join(res["details"][:2])  # 주요 2개만 표시
        t.add_row(name, score_str, details_str)

    console.print(t)

print_signal(omni_result["signal"], omni_result["confidence"])
rprint(f"  Total Score: {omni_result['total_score']}")
```

---

### STEP 9: main() 함수 통합 흐름

**전체 실행 순서** (v2의 main()을 아래 순서로 재구성):

```python
def main():
    # 0. CLI 인자 파싱 (v2와 동일 + --skip-omni 옵션 추가)
    parser.add_argument("--skip-omni", action="store_true",
                        help="Skip v3 7-engine analysis")

    # 1. 메인 종목 데이터 fetch
    df = UnifiedDataFetcher.fetch(args.symbol, args.market, args.period)

    # 2. 거시 데이터 fetch
    macro_df = UnifiedDataFetcher.fetch_macro(args.period)
    macro_feat = MacroFeatures.compute(macro_df)          # 확장된 24개 피처
    macro_signal = MacroFeatures.get_signal(macro_feat)

    # 3. Mag7 데이터 fetch  ← 신규
    mag7_df = UnifiedDataFetcher.fetch_mag7(args.period)

    # 4. 기술적 분석
    df = TechnicalAnalysis.compute(df)
    tech_signal = TechnicalAnalysis.get_signal(df)

    # 5. 엔진 피처 계산  ← 신규
    if not args.skip_omni:
        engine_feat = EngineFeatureExtractor.compute(df, macro_df, mag7_df)
    else:
        engine_feat = pd.DataFrame()

    # 6. ML 예측 (엔진 피처 포함)
    ml_result = MLPredictor().train_and_predict(df, macro_feat, engine_feat)

    # 7. LSTM 예측 (v2 그대로)
    lstm_result = LSTMPredictor().train_and_predict(df)

    # 8. 감성 분석 (v2 그대로)
    sent_result = SentimentAnalyzer.analyze(ticker)

    # 9. Omni 시그널 생성  ← 신규
    if not args.skip_omni:
        omni_result = _compute_omni_signal(df, macro_df, mag7_df)
    else:
        omni_result = {"signal": "N/A", "confidence": 0.0, "note": "Skipped."}

    # 10. 앙상블 결합 (omni 포함)
    ensemble = EnsembleCombiner.combine(
        technical=tech_signal,
        macro=macro_signal,
        ml=ml_result,
        lstm=lstm_result,
        sentiment=sent_result,
        omni=omni_result,            # ← 신규
    )

    # 11. 리스크 관리 (v2 그대로)
    risk_levels = RiskManager.compute_stop_levels(close, atr, ensemble["signal"])

    # 12. 백테스팅 (엔진 피처 포함)
    backtest_stats = Backtester().run(
        df, macro_feat, engine_feat,   # ← engine_feat 추가
        initial_capital=args.capital,
        risk_per_trade=args.risk,
        lookback=args.backtest_days,
    )

    # 13. 리포트 출력
    print_report(...)

    # 14. JSON 내보내기 (omni_result 포함)
```

---

## 4. 주의사항 및 엣지 케이스

### 4-1. 데이터 누출 방지 (Look-ahead Bias)

- `EngineFeatureExtractor.compute()`에서 날짜 i의 피처는 반드시 `[:i+1]` 슬라이스만 사용한다.
- Backtester의 walk-forward 루프에서 `engine_feat.iloc[:i]`로 학습하고 `engine_feat.iloc[i]`로 예측해야 한다. 이는 이미 `combined.iloc[:i]`로 처리되므로 추가 조치 불필요.

### 4-2. 결측값 처리

- 엔진 피처에서 예외 발생 시 `score = 0.0`으로 대체한다 (각 엔진 호출부의 try-except).
- `combined.join(engine_feat, how="left")` 후 `dropna()`에서 엔진 피처가 NaN인 초기 행은 자동 제거된다.
- Mag7 또는 거시 데이터가 완전히 실패할 경우, 엔진 피처 전체가 0이 되어도 ML은 기존 TECH+MACRO 피처로 동작한다.

### 4-3. 성능 최적화

- `EngineFeatureExtractor.compute()`는 최대 300일 루프로 제한한다.
- Engine5 (심리)는 항상 0을 반환하므로 사실상 연산 비용 없음.
- 각 엔진의 analyze()는 pandas 연산만 사용하여 개별 호출은 < 1ms.
- 전체 300일 루프: 약 1~3초 예상.

### 4-4. v3 Engine 수정 금지

- v3의 Engine1~7 코드는 그대로 복사하며, 입출력 인터페이스를 변경하지 않는다.
- 향후 v3가 업데이트되면 v4에도 동일하게 반영할 수 있도록 원본 유지.

---

## 5. 피처 전체 목록 (최종)

| 그룹 | 개수 | 피처명 |
|------|------|--------|
| **TECH** (v2) | 20 | RSI, MACD, MACD_Signal, MACD_Hist, BB_Pct, BB_Width, Stoch_K, Stoch_D, ATR_Pct, Vol_Ratio, ROC_1, ROC_5, ROC_10, ROC_20, Williams_R, Weekly_Return, Monthly_Return, Return_Std_10, Return_Std_20, Price_Position_20 |
| **MACRO** (v2+확장) | 22 | Yield_Spread_10Y3M, Yield_Spread_Chg5, VIX, VIX_SMA10, VIX_Ratio, VIX_Chg5, VIX_Regime, DXY_ROC5, DXY_ROC20, Gold_ROC5, Gold_ROC20, Oil_ROC5, Oil_ROC20, SP500_ROC5, SP500_Above_SMA50, NASDAQ_ROC5, Macro_Momentum, HYG_TLT_Ratio, HYG_TLT_ROC20, VIX_Term_Structure, Copper_ROC10, Semi_ROC10, BTC_ROC10 |
| **ENGINE** (v3 신규) | 13 | engine1_macro, engine2_fundamentals, engine3_technical, engine4_derivatives, engine5_sentiment, engine6_cross_asset, engine7_behavioral, engine_total_score, engine_bullish_count, engine_bearish_count, engine_total_score_chg5, engine_total_score_sma5, engine_external_score |
| **합계** | **55** | |

---

## 6. CLI 사용 예시

```bash
# 기본 실행 (모든 기능 활성화)
python v4/stock_predictor_v4.py AAPL

# v3 엔진 분석 비활성화 (v2 동작과 동일)
python v4/stock_predictor_v4.py AAPL --skip-omni

# 한국 시장
python v4/stock_predictor_v4.py 005930 --market kospi

# 풀 옵션
python v4/stock_predictor_v4.py QQQ --period 3y --capital 200000 --backtest-days 500 --export results.json
```

---

## 7. 구현 체크리스트

- [ ] **STEP 1**: `UnifiedDataFetcher` 클래스 작성 (v2 DataFetcher 복사 + v3 티커 병합 + fetch_mag7 추가)
- [ ] **STEP 2**: Engine1~7 클래스 v3에서 그대로 복사
- [ ] **STEP 3**: `EngineFeatureExtractor` 클래스 신규 작성 (일별 엔진 점수 → DataFrame 변환)
- [ ] **STEP 4**: `MLPredictor` 확장 (ENGINE_FEATURES 상수 추가, prepare_data/train_and_predict에 engine_feat 인자 추가)
- [ ] **STEP 5**: `Backtester.run()` 확장 (engine_feat 인자 추가, all_features에 엔진 피처 포함)
- [ ] **STEP 6**: `EnsembleCombiner.WEIGHTS` 재조정 (omni 0.20 추가) + omni 시그널 생성 로직
- [ ] **STEP 7**: `MacroFeatures.compute()` 확장 (HYG/TLT, VIX3M, Copper, Semi, BTC 피처 6개 추가)
- [ ] **STEP 8**: `print_report()` 확장 (7엔진 분석 섹션 추가)
- [ ] **STEP 9**: `main()` 재구성 (통합 흐름으로 재작성)
- [ ] **테스트**: `python v4/stock_predictor_v4.py AAPL` 실행하여 전체 파이프라인 정상 동작 확인
- [ ] **테스트**: `--skip-omni` 플래그로 v2 호환 모드 동작 확인
- [ ] **테스트**: `--export results.json`으로 JSON 출력에 omni_result 포함 확인
