# 글로벌 시장 옴니 예측기 v5: 7대 엔진 피처 고도화 명세서

본 문서는 `market_predictor_v4.py`를 계승하여, 제공된 7대 시장 분석 요소를 정량적 데이터로 변환하고 모델에 주입하기 위한 **최종 구현 가이드**이다.

---

## 0. 설계 원칙 (Core Principles)
- **Data Source**: Yahoo Finance (`yf`)를 기본으로 하되, 모든 티커는 `UnifiedDataFetcher`에서 관리한다.
- **Target Variable**: 분석 대상 지수(예: `^GSPC`)의 **T+1일 수익률 방향** (1: 상승, 0: 하락).
- **Anti-Bias**: 모든 피처는 과거 시점 데이터만 사용하며, `shift(1)` 처리를 통해 미래 데이터 누출(Look-ahead Bias)을 원천 차단한다.

---

## 1. 우선순위별 구현 항목 (Priority Roadmap)

### [P0] 거시 유동성 및 파생상품 체제 (결정력 70%)
1.  **금리 체인**: US10Y, US2Y, US3M 수익률 기반 장단기 금리차 구현.
2.  **신용 경색 지표**: `HYG(하이일드)`와 `TLT(국채)`의 상대 강도 및 모멘텀.
3.  **VIX 기간 구조**: `VIX(현물/단기)` / `VIX3M(중기)` 비율을 통한 시장 공포의 기간적 구조 분석.
4.  **달러 인덱스 (DXY)**: 글로벌 유동성 흡수/공급 척도로서의 ROC(변화율).

### [P1] 교차 자산 선행 지표 및 지수 내부 건전성
1.  **닥터 코퍼 (Dr. Copper)**: 구리 선물(`HG=F`) 가격 변화율을 통한 실물 경기 선행 진단.
2.  **테크/반도체 리더십**: `SOXX(반도체 ETF)`의 지수 대비 상대 강도(Relative Strength).
3.  **위험 자산 선호도**: 비트코인(`BTC-USD`)의 단기 변동성 및 지수 상관관계 변화.
4.  **메가캡 Breadth**: Magnificent 7 종목 중 지수 추세와 동조화되는 종목 수 산출.

### [P2] 시스템 수급 및 행동재무학 (계절성)
1.  **거래량 강도 (Volume Force)**: 가격 돌파 시 거래량 동반 여부 정량화.
2.  **캘린더 피처**: 월말 리밸런싱(25일~말일), 산타 랠리 구간, 요일 효과를 바이너리(0/1) 피처로 변환.
3.  **변동성 군집 (Clustering)**: 단기 변동성과 중기 변동성의 비율을 통한 급락 위험도 측정.

---

## 2. 엔진별 세부 피처 생성 공식 (Detailed Logic)

### Engine 1: 거시경제 및 통화 정책
- `Yield_Curve_10Y2Y`: `US10Y - US2Y` (침체 전조)
- `Credit_Spread_Proxy`: `(HYG / TLT) / SMA(HYG / TLT, 20) - 1` (신용 리스크 가속도)
- `DXY_Trend`: `DXY`의 5일 및 20일 ROC 평균 (자금 이탈 속도)

### Engine 2: 펀더멘털 및 시장 가중치
- `Mag7_Alignment`: Mag7 종목 중 종가가 5일 이동평균선 위에 있는 종목 수 (0~7)
- `Index_Valuation_Proxy`: `Close / SMA(Close, 200)` (역사적 평균 대비 과열도)

### Engine 3: 기술적 지표 및 수급
- `MFI_Signal`: Money Flow Index (가격+거래량)
- `Relative_Volume_Shock`: `Volume / SMA(Volume, 20)` (기계적 수급 폭발 여부)

### Engine 4: 파생상품 및 마이크로 구조
- `Term_Structure_Premium`: `VIX / VIX3M` (1.0 초과 시 '백워데이션'으로 강력 약세 신호)
- `VIX_Regime_Change`: `VIX / SMA(VIX, 10)` (변동성 체제 전환 여부)

### Engine 5: 심리 및 대체 데이터 (프록시 기반)
- `Contrarian_Index`: (지수 RSI + VIX 역전값 + 탐욕지수 프록시)의 합산
- `Event_Window_Flag`: 주요 지표 발표일이나 리밸런싱 기간 여부 (바이너리)

### Engine 6: 교차 자산 및 섹터 연결성
- `Copper_Gold_Alpha`: `HG=F_ROC(10) - GC=F_ROC(10)` (경기 확장 vs 안전 자산 선호 비교)
- `Semi_Lead_Lag`: `SOXX_ROC(5) - Index_ROC(5)` (테크 섹터의 시장 견인력)

### Engine 7: 구조적 및 행동 리스크
- `Tail_Risk_Score`: `StdDev(Return, 5) / StdDev(Return, 60)` (꼬리 위험 증폭 여부)
- `Exhaustion_Score`: 최근 14일 중 상승/하락 일수의 극단적 쏠림 정도

---

## 3. 구현 단계 (Implementation Steps)

1.  **UnifiedDataFetcher 확장**: `TICKERS` 딕셔너리에 매크로, 상품, 섹터 ETF 전 종목 추가.
2.  **EngineFeatureExtractor 고도화**: 상기 7대 엔진별 공식을 코드로 구현하여 `DataFrame` 생성.
3.  **MLPredictor 최적화**: 
    - `RandomForest`, `XGBoost`, `LightGBM` 등 앙상블 모델 강화.
    - 피처 중요도 분석을 통해 기여도가 낮은 시장 변수 자동 필터링.
4.  **시장 체제(Regime) 출력**: 
    - 최종 리포트에 "현재 시장 국면" 자동 정의 (예: `안전 자산 선호`, `유동성 장세`, `변동성 전이` 등).

---

## 4. 최종 검증 항목 (Verification)
- [ ] 데이터 결측치(`NaN`) 처리가 `ffill` 후 `bfill`로 완벽하게 수행되는가?
- [ ] 모든 피처 계산에 미래 데이터가 1%라도 포함되지 않았는가?
- [ ] 7개 엔진의 점수 합산 로직이 `-15`에서 `+15` 사이의 변별력을 가지는가?
- [ ] 백테스트 수수료가 지수 ETF(SPY, QQQ) 실제 거래 비용을 반영하고 있는가?
