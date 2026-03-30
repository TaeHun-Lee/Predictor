import re
import os

with open("v2/stock_predictor_v2.py", "r", encoding="utf-8") as f:
    v2_code = f.read()

with open("v3/stock_predictor_v3.py", "r", encoding="utf-8") as f:
    v3_code = f.read()

# Extract v3 Engines 1~7
engine_pattern = re.compile(r'(class Engine1_Macro:.*?)(?=\n# ══+)', re.DOTALL)
engine_match = engine_pattern.search(v3_code)
v3_engines = engine_match.group(1)

# Now start building v4_code
v4_lines = []

v2_lines = v2_code.split("\n")

# Process v2 lines and inject/replace
i = 0
while i < len(v2_lines):
    line = v2_lines[i]
    
    # Replace header
    if "STOCK MARKET PREDICTOR v2.0" in line:
        v4_lines.append(line.replace("v2.0", "v4.0"))
        i += 1
        continue
        
    if "Technical · Macro · ML · LSTM · Sentiment · Backtesting" in line:
        v4_lines.append(line.replace("Sentiment · Backtesting", "Sentiment · Omni · Backtesting"))
        i += 1
        continue
    
    # unified DataFetcher replace
    if "class DataFetcher:" in line:
        v4_lines.append('''class UnifiedDataFetcher:
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
            return pd.DataFrame()''')
        # skip until MacroFeatures
        while "class MacroFeatures:" not in v2_lines[i]:
            i += 1
        
        # Inject v3 engines and EngineFeatureExtractor before MacroFeatures
        v4_lines.append("\n\n# ═══════════════════════════════════════════════════════════════════")
        v4_lines.append("#  v3 ENGINE 1~7")
        v4_lines.append("# ═══════════════════════════════════════════════════════════════════\n")
        v4_lines.append(v3_engines)
        
        v4_lines.append('''
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
''')
        # Do not i += 1 here, just continue so the outer loop processes 'class MacroFeatures:'
        continue
    
    # Inject expanded MacroFeatures
    if "class MacroFeatures:" in line:
        # copy it, but replace compute
        v4_lines.append("class MacroFeatures:")
        i += 1
        while "def compute" not in v2_lines[i]:
            v4_lines.append(v2_lines[i])
            i += 1
        v4_lines.append('''    @staticmethod
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

        return feat''')
        # skip old compute
        while "def get_signal" not in v2_lines[i]:
            i += 1
        continue
    
    # MLPredictor updates
    if "class MLPredictor:" in line:
        v4_lines.append(line)
        i += 1
        while "def __init__(self):" not in v2_lines[i]:
            if "MACRO_FEATURES = [" in v2_lines[i]:
                v4_lines.append('''    MACRO_FEATURES = [
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
    ]''')
                while "]" not in v2_lines[i]:
                    i+=1
            else:
                v4_lines.append(v2_lines[i])
            i += 1
        continue
    
    if "def prepare_data(self, df: pd.DataFrame, macro_feat: pd.DataFrame = None):" in line:
        v4_lines.append('''    def prepare_data(self, df: pd.DataFrame,
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
        return X, y, all_features''')
        while "def train_and_predict" not in v2_lines[i]:
            i += 1
        continue
        
    if "def train_and_predict(self, df: pd.DataFrame, macro_feat: pd.DataFrame = None) -> dict:" in line:
        v4_lines.append('''    def train_and_predict(self, df: pd.DataFrame,
                          macro_feat: pd.DataFrame = None,
                          engine_feat: pd.DataFrame = None) -> dict:''')
        i += 1
        while "X, y, feature_names = self.prepare_data(df, macro_feat)" not in v2_lines[i]:
            v4_lines.append(v2_lines[i])
            i += 1
        v4_lines.append("        X, y, feature_names = self.prepare_data(df, macro_feat, engine_feat)")
        i += 1
        continue
        
    # EnsembleCombiner updates
    if "class EnsembleCombiner:" in line:
        v4_lines.append("class EnsembleCombiner:")
        i += 1
        while "def combine" not in v2_lines[i]:
            if "WEIGHTS = {" in v2_lines[i]:
                v4_lines.append('''    WEIGHTS = {
        "technical":  0.15,
        "macro":      0.10,
        "ml":         0.30,
        "lstm":       0.15,
        "sentiment":  0.10,
        "omni":       0.20,
    }''')
                while "}" not in v2_lines[i]:
                    i+=1
            i += 1
        continue

    # Backtester updates
    if "def run(self, df: pd.DataFrame, macro_feat: pd.DataFrame = None," in line:
        v4_lines.append('''    def run(self, df: pd.DataFrame, macro_feat: pd.DataFrame = None,
            engine_feat: pd.DataFrame = None,
            initial_capital: float = 100_000, risk_per_trade: float = 0.02,
            lookback: int = 252) -> dict:''')
        # skip past the signature
        while "def run(" not in v2_lines[i]: i+=1
        while ")" not in v2_lines[i]: i+=1
        i += 1
        continue

    if "if macro_feat is not None and not macro_feat.empty:" in line and "class Backtester" in "\n".join(v2_lines[max(0, i-50):i]):
        v4_lines.append('''        engine_features_list = MLPredictor.ENGINE_FEATURES
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
        )''')
        while "feature_df = combined" not in v2_lines[i]:
            i += 1
        continue
        
    # print_report signature
    if "def print_report(symbol, market, tech, macro, ml, lstm, sentiment, ensemble, risk, backtest, df):" in line:
        v4_lines.append("def print_report(symbol, market, tech, macro, ml, lstm, sentiment, omni_result, ensemble, risk, backtest, df):")
        i += 1
        continue
        
    if "# ── ENSEMBLE ──" in line:
        v4_lines.append('''    # ── Omni 7-Engine Analysis ──
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
''')
        v4_lines.append(line)
        i += 1
        continue

    # main()
    if "def main():" in line:
        v4_lines.append('''
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
''')
        v4_lines.append(line)
        i += 1
        continue

    if "parser.add_argument(\"--skip-backtest\", action=\"store_true\")" in line:
        v4_lines.append(line)
        v4_lines.append("    parser.add_argument(\"--skip-omni\", action=\"store_true\", help=\"Skip v3 7-engine analysis\")")
        i += 1
        continue

    # main data fetching
    if "df = DataFetcher.fetch(args.symbol, args.market, args.period)" in line:
        v4_lines.append("        df = UnifiedDataFetcher.fetch(args.symbol, args.market, args.period)")
        i += 1
        continue

    if "macro_df = DataFetcher.fetch_macro(args.period)" in line:
        v4_lines.append("            macro_df = UnifiedDataFetcher.fetch_macro(args.period)")
        i += 1
        continue

    if "tech_signal = TechnicalAnalysis.get_signal(df)" in line:
        v4_lines.append(line)
        v4_lines.append('''
    # Mag7 Data
    mag7_df = UnifiedDataFetcher.fetch_mag7(args.period)

    # Engine Features
    if args.skip_omni:
        engine_feat = pd.DataFrame()
        omni_result = {"signal": "N/A", "confidence": 0.0, "note": "Skipped."}
    else:
        engine_feat = EngineFeatureExtractor.compute(df, macro_df, mag7_df)
        omni_result = _compute_omni_signal(df, macro_df, mag7_df)
''')
        i += 1
        continue

    if "ml_result = MLPredictor().train_and_predict(df, macro_feat)" in line:
        v4_lines.append("            ml_result = MLPredictor().train_and_predict(df, macro_feat, engine_feat)")
        i += 1
        continue

    if "ensemble = EnsembleCombiner.combine(" in line:
        v4_lines.append('''    ensemble = EnsembleCombiner.combine(
        technical=tech_signal, macro=macro_signal,
        ml=ml_result, lstm=lstm_result, sentiment=sent_result, omni=omni_result
    )''')
        while ")" not in v2_lines[i]:
            i += 1
        i += 1
        continue
        
    if "df, macro_feat," in line and "initial_capital=" in v2_lines[i+1]:
        v4_lines.append("                df, macro_feat, engine_feat,")
        i += 1
        continue
        
    if "tech_signal, macro_signal, ml_result, lstm_result, sent_result," in line and "ensemble, risk_levels, backtest_stats, df," in v2_lines[i+1]:
        v4_lines.append("        tech_signal, macro_signal, ml_result, lstm_result, sent_result, omni_result,")
        i += 1
        continue
        
    if "\"ml\": ml_result, \"lstm\": lstm_result, \"sentiment\": sent_result," in line:
        v4_lines.append("            \"ml\": ml_result, \"lstm\": lstm_result, \"sentiment\": sent_result, \"omni\": omni_result,")
        i += 1
        continue

    if "DataFetcher.get_ticker" in line:
        v4_lines.append(line.replace("DataFetcher", "UnifiedDataFetcher"))
        i += 1
        continue

    v4_lines.append(line)
    i += 1

with open("v4/stock_predictor_v4.py", "w", encoding="utf-8") as f:
    f.write("\n".join(v4_lines))
