# Review: Time Series Lecture Notebooks

## 2-3, 1. time_series_eda_bike_sharing_rich_lecture.ipynb

- **Cells**: 47 (code 17, markdown 30)
- **First headings**: Chapter 2-3, 1강 시계열 데이터 이해 및 분석 — Bike Sharing (Hourly), 구성, 0. 환경 준비 및 라이브러리 임포트, 안내: 경고 억제와 한글 폰트, 1. 데이터 불러오기와 기본 전처리
- **Top imports**: os(1), warnings(1), numpy(1), pandas(1), matplotlib.pyplot(1), matplotlib(1), statsmodels.tsa.seasonal(1), statsmodels.nonparametric.smoothers_lowess(1), matplotlib.patches(1)
- **Data paths referenced**: bike-sharing-demand/train.csv
- **Splitting strategy**: —
- **Scaling**: —
- **Stat tests / decomposition**: STL
- **Models**: —
- **Metrics**: —
- **Random seeds set**: No; **Torch seed**: No

## 2-3, 2-1. time_series_forecasting_classical_baselines.ipynb

- **Cells**: 21 (code 9, markdown 12)
- **First headings**: Chapter 2-3, 2-1강 시계열 예측 모델링 — Classical & Baselines (Bike Sharing), 강의 개요 및 학습 목표, 0. 환경 준비 및 라이브러리 임포트, 1. 데이터 준비 및 전처리, 2. 일반 회귀 베이스라인: 선형 회귀 vs 랜덤포레스트
- **Top imports**: sklearn.linear_model(2), os(1), warnings(1), numpy(1), pandas(1), matplotlib.pyplot(1), matplotlib(1), statsmodels.tsa.statespace.sarimax(1), prophet(1), sklearn.ensemble(1), sklearn.metrics(1), statsmodels.tsa.stattools(1)
- **Data paths referenced**: bike-sharing-demand/train.csv
- **Splitting strategy**: TimeSeriesSplit
- **Scaling**: —
- **Stat tests / decomposition**: ACF, ADF, PACF
- **Models**: ARIMA/SARIMAX, Prophet
- **Metrics**: MAE, MSE/RMSE, RMSE
- **Random seeds set**: No; **Torch seed**: No

## 2-3, 2-2. time_series_forecasting_pytorch_lstm.ipynb

- **Cells**: 16 (code 8, markdown 8)
- **First headings**: Chapter 2-3, 2-2강 시계열 예측 모델링 — PyTorch LSTM (Bike Sharing), 0. 환경 준비 및 라이브러리 임포트, 1. 데이터 준비 및 전처리, 2. 시퀀스 데이터셋 생성 및 LSTM 모델 정의, 3. 학습/검증/테스트
- **Top imports**: numpy(3), torch(3), os(2), warnings(2), pandas(2), matplotlib.pyplot(2), torch.utils.data(2), sys(1), matplotlib(1), statsmodels.tsa.seasonal(1), statsmodels.nonparametric.smoothers_lowess(1), torch.nn(1)
- **Data paths referenced**: bike-sharing-demand/train.csv
- **Splitting strategy**: —
- **Scaling**: —
- **Stat tests / decomposition**: —
- **Models**: LSTM
- **Metrics**: MAE, RMSE
- **Random seeds set**: Yes; **Torch seed**: Yes

### Findings for **2-3, 1. time_series_eda_bike_sharing_rich_lecture.ipynb**
**Potential issues / opportunities:**
- No explicit naive/seasonal-naive baseline for context.
- No walk-forward / TimeSeriesSplit CV; single split only.
- No uncertainty or prediction intervals.

**Quick wins:**
- Set `np.random.seed(42)` and `random.seed(42)` for reproducibility.
- Use `TimeSeriesSplit` or walk-forward validation.
### Findings for **2-3, 2-1. time_series_forecasting_classical_baselines.ipynb**
**Potential issues / opportunities:**
- No explicit naive/seasonal-naive baseline for context.
- No uncertainty or prediction intervals.

**Quick wins:**
- Set `np.random.seed(42)` and `random.seed(42)` for reproducibility.
### Findings for **2-3, 2-2. time_series_forecasting_pytorch_lstm.ipynb**
**Potential issues / opportunities:**
- shuffle=True may break time order for time series.
- PyTorch DataLoader uses shuffle=True (should be False for time series).
- No explicit naive/seasonal-naive baseline for context.
- No walk-forward / TimeSeriesSplit CV; single split only.
- No uncertainty or prediction intervals.
- No learning rate schedule.
- No early stopping.
- No gradient clipping.

**Quick wins:**
- Use `TimeSeriesSplit` or walk-forward validation.