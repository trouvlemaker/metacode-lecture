"""
Chapter 2-3, 2-1강 시계열 예측 모델링 — Classical & Baselines (Bike Sharing)

목표
- 시계열 예측의 기본 개념과 베이스라인을 구축하고, 회귀/시계열 전용 모델을 비교합니다.
- 데이터 준비, 특성 엔지니어링, 일반 회귀(선형회귀/랜덤포레스트), ARIMA/SARIMA, Prophet(선택)을 다룹니다.

규칙(강의용)
- 시각화는 matplotlib만 사용합니다.
- seaborn 사용 X, 색상 지정 X, 서브플롯 X (단일 도표 중심)

데이터
- Kaggle Bike Sharing Demand (시간 단위, target: count)
"""

from __future__ import annotations

import os
import warnings
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import font_manager

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error

from statsmodels.tsa.statespace.sarimax import SARIMAX

# Prophet은 선택 사항 (미설치 환경 대응)
try:
    from prophet import Prophet  # type: ignore
    _HAS_PROPHET = True
except Exception:
    _HAS_PROPHET = False


# 전역 경고 억제
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", message=r"Glyph.*missing from font.*", category=UserWarning)


def _set_korean_font() -> None:
    font_candidates = [
        "AppleGothic",
        "NanumGothic",
        "Malgun Gothic",
        "Noto Sans CJK KR",
        "Noto Sans KR",
        "DejaVu Sans",
    ]
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = font_candidates
    plt.rcParams["axes.unicode_minus"] = False


_set_korean_font()
pd.set_option("display.max_columns", 100)


# =========================================
# 1) 데이터 준비 및 전처리
# =========================================


def load_hourly_data() -> pd.DataFrame:
    """Bike Sharing Demand 시간단위 데이터 로드 및 정렬, 열 순서 고정"""
    candidates = [
        "bike-sharing-demand/train.csv",
    ]
    path = None
    for c in candidates:
        if os.path.exists(c):
            path = c
            break
    if path is None:
        raise FileNotFoundError("train.csv 경로를 찾을 수 없습니다.")

    df = pd.read_csv(path)
    df["datetime"] = pd.to_datetime(df["datetime"])  # 시계열 파싱
    df = df.sort_values("datetime").reset_index(drop=True)
    cols = [
        "datetime",
        "season",
        "holiday",
        "workingday",
        "weather",
        "temp",
        "atemp",
        "humidity",
        "windspeed",
        "casual",
        "registered",
        "count",
    ]
    df = df[cols]
    return df


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """시간 파생변수 생성 (년/월/일/요일/시간/계절 등)"""
    out = df.copy()
    out["year"] = out["datetime"].dt.year
    out["month"] = out["datetime"].dt.month
    out["day"] = out["datetime"].dt.day
    out["dayofweek"] = out["datetime"].dt.dayofweek
    out["hour"] = out["datetime"].dt.hour
    # 계절(season) 컬럼은 기존에 존재: 1~4
    return out


def split_by_time(df: pd.DataFrame, train_ratio: float = 0.8, val_ratio: float = 0.1) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """시간 순서로 train/val/test 분할"""
    n = len(df)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    train = df.iloc[:n_train]
    val = df.iloc[n_train : n_train + n_val]
    test = df.iloc[n_train + n_val :]
    return train, val, test


def to_supervised_features(df: pd.DataFrame, target: str = "count") -> Tuple[pd.DataFrame, pd.Series]:
    """회귀용 입력/타깃 생성 + 원-핫 인코딩"""
    feature_cols = [
        "season",
        "holiday",
        "workingday",
        "weather",
        "temp",
        "atemp",
        "humidity",
        "windspeed",
        "year",
        "month",
        "day",
        "dayofweek",
        "hour",
    ]
    X = df[feature_cols].copy()
    X = pd.get_dummies(X, columns=["season", "holiday", "workingday", "weather", "year", "month", "dayofweek", "hour"], drop_first=False)
    y = df[target].astype(float)
    return X, y


# =========================================
# 2) 평가 지표
# =========================================


@dataclass
class Metrics:
    mae: float
    mse: float
    rmse: float
    mape: float
    directional_accuracy: Optional[float] = None


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Metrics:
    mae = float(mean_absolute_error(y_true, y_pred))
    mse = float(mean_squared_error(y_true, y_pred))
    rmse = float(np.sqrt(mse))
    with np.errstate(divide="ignore", invalid="ignore"):
        mape = float(np.mean(np.abs((y_true - y_pred) / np.clip(np.abs(y_true), 1e-8, None))) * 100.0)

    # Directional Accuracy: sign(y_t - y_{t-1}) == sign(yhat_t - y_{t-1})
    if len(y_true) > 1:
        prev = np.concatenate([[y_true[0]], y_true[:-1]])
        actual_dir = np.sign(y_true - prev)
        pred_dir = np.sign(y_pred - prev)
        da = float(np.mean((actual_dir == pred_dir).astype(float)))
    else:
        da = np.nan
    return Metrics(mae=mae, mse=mse, rmse=rmse, mape=mape, directional_accuracy=da)


def print_metrics(name: str, m: Metrics) -> None:
    print(f"[{name}] MAE={m.mae:.3f}, MSE={m.mse:.3f}, RMSE={m.rmse:.3f}, MAPE={m.mape:.2f}%, DirAcc={m.directional_accuracy:.3f}")


# =========================================
# 3) 베이스라인 회귀 모델
# =========================================


def train_linear_regression(X_train: pd.DataFrame, y_train: pd.Series) -> LinearRegression:
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model


def train_random_forest(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    n_estimators: int = 200,
    max_depth: Optional[int] = None,
    random_state: int = 42,
) -> RandomForestRegressor:
    model = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=random_state,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    return model


def plot_actual_vs_pred(dt_index: pd.DatetimeIndex, y_true: np.ndarray, y_pred: np.ndarray, title: str) -> None:
    plt.figure(figsize=(12, 4))
    plt.plot(dt_index, y_true, label="Actual")
    plt.plot(dt_index, y_pred, label="Pred")
    plt.title(title)
    plt.xlabel("Time")
    plt.ylabel("Count")
    plt.legend()
    plt.tight_layout()
    plt.show()


# =========================================
# 4) 시계열 전용 모델: ARIMA/SARIMA, Prophet(선택)
# =========================================


def aggregate_daily(df: pd.DataFrame) -> pd.Series:
    s = df.set_index("datetime")["count"].resample("D").mean()
    return s


def fit_sarima(train: pd.Series, order: Tuple[int, int, int] = (1, 1, 1), seasonal_order: Tuple[int, int, int, int] = (1, 1, 1, 7)) -> SARIMAX:
    model = SARIMAX(train, order=order, seasonal_order=seasonal_order, enforce_stationarity=False, enforce_invertibility=False)
    res = model.fit(disp=False)
    return res


def forecast_sarima(res, steps: int) -> np.ndarray:
    fc = res.forecast(steps=steps)
    return np.asarray(fc)


def fit_prophet_daily(train: pd.Series):
    if not _HAS_PROPHET:
        print("Prophet 미설치: Prophet 모델을 건너뜁니다.")
        return None
    dfp = pd.DataFrame({"ds": train.index, "y": train.values})
    m = Prophet(
        seasonality_mode="additive",
        yearly_seasonality=False,
        weekly_seasonality=True,
        daily_seasonality=False,
    )
    m.fit(dfp)
    return m


def forecast_prophet(model, horizon: int, start_index: pd.DatetimeIndex) -> Optional[np.ndarray]:
    if model is None:
        return None
    future = pd.DataFrame({"ds": pd.date_range(start=start_index[-1] + pd.Timedelta(days=1), periods=horizon, freq="D")})
    fc = model.predict(future)
    return fc["yhat"].to_numpy()


# =========================================
# 5) 실행 절차 (메인 흐름)
# =========================================


def main() -> None:
    # 개요 출력
    print("시계열 예측의 중요성과 활용: 수요예측/재고/인력/에너지/교통 등")
    print("일반 회귀 vs 시계열 모델: IID 가정 vs 자기상관/계절성/추세 고려")

    # 데이터 로드/전처리
    df = load_hourly_data()
    df_feat = add_time_features(df)
    train_df, val_df, test_df = split_by_time(df_feat, train_ratio=0.8, val_ratio=0.1)

    # 회귀용 입력/타깃
    X_train, y_train = to_supervised_features(train_df)
    X_val, y_val = to_supervised_features(val_df)
    X_test, y_test = to_supervised_features(test_df)

    # 선형 회귀
    lin = train_linear_regression(X_train, y_train)
    lin_val_pred = lin.predict(X_val)
    lin_test_pred = lin.predict(X_test)
    lin_val_m = compute_metrics(y_val.to_numpy(), lin_val_pred)
    lin_test_m = compute_metrics(y_test.to_numpy(), lin_test_pred)
    print_metrics("Linear/VAL", lin_val_m)
    print_metrics("Linear/TEST", lin_test_m)
    plot_actual_vs_pred(val_df["datetime"].values, y_val.to_numpy(), lin_val_pred, "선형회귀 검증 예측 vs 실제")
    plot_actual_vs_pred(test_df["datetime"].values, y_test.to_numpy(), lin_test_pred, "선형회귀 테스트 예측 vs 실제")

    # 랜덤 포레스트
    rf = train_random_forest(X_train, y_train, n_estimators=300, max_depth=None)
    rf_val_pred = rf.predict(X_val)
    rf_test_pred = rf.predict(X_test)
    rf_val_m = compute_metrics(y_val.to_numpy(), rf_val_pred)
    rf_test_m = compute_metrics(y_test.to_numpy(), rf_test_pred)
    print_metrics("RF/VAL", rf_val_m)
    print_metrics("RF/TEST", rf_test_m)
    plot_actual_vs_pred(val_df["datetime"].values, y_val.to_numpy(), rf_val_pred, "랜덤포레스트 검증 예측 vs 실제")
    plot_actual_vs_pred(test_df["datetime"].values, y_test.to_numpy(), rf_test_pred, "랜덤포레스트 테스트 예측 vs 실제")

    # 특성 중요도 상위 출력 (랜덤포레스트)
    try:
        importances = pd.Series(rf.feature_importances_, index=X_train.columns).sort_values(ascending=False)
        print("랜덤포레스트 중요도 TOP 15:\n", importances.head(15))
    except Exception:
        pass

    # ARIMA/SARIMA (일 단위 집계)
    daily = aggregate_daily(df)
    n = len(daily)
    n_train = int(n * 0.8)
    n_val = int(n * 0.1)
    daily_train = daily.iloc[:n_train]
    daily_val = daily.iloc[n_train : n_train + n_val]
    daily_test = daily.iloc[n_train + n_val :]

    sarima_res = fit_sarima(daily_train)
    sar_val_fc = forecast_sarima(sarima_res, steps=len(daily_val))
    sar_test_fc = forecast_sarima(sarima_res.apply(daily_train.append(daily_val)), steps=len(daily_test)) if hasattr(sarima_res, "apply") else forecast_sarima(fit_sarima(daily_train.append(daily_val)), steps=len(daily_test))

    sar_val_m = compute_metrics(daily_val.to_numpy(), sar_val_fc)
    sar_test_m = compute_metrics(daily_test.to_numpy(), sar_test_fc)
    print_metrics("SARIMA/VAL", sar_val_m)
    print_metrics("SARIMA/TEST", sar_test_m)
    plot_actual_vs_pred(daily_val.index, daily_val.to_numpy(), sar_val_fc, "SARIMA 검증 예측 vs 실제 (일)")
    plot_actual_vs_pred(daily_test.index, daily_test.to_numpy(), sar_test_fc, "SARIMA 테스트 예측 vs 실제 (일)")

    # Prophet (옵션)
    if _HAS_PROPHET:
        prop_model = fit_prophet_daily(daily_train)
        prop_val_fc = forecast_prophet(prop_model, len(daily_val), start_index=daily_train.index)
        prop_test_fc = forecast_prophet(prop_model, len(daily_val) + len(daily_test), start_index=daily_train.index)
        if prop_val_fc is not None:
            prop_val_m = compute_metrics(daily_val.to_numpy(), prop_val_fc)
            print_metrics("Prophet/VAL", prop_val_m)
            plot_actual_vs_pred(daily_val.index, daily_val.to_numpy(), prop_val_fc, "Prophet 검증 예측 vs 실제 (일)")
        if prop_test_fc is not None:
            # 테스트는 마지막 len(test) 구간만 취함
            prop_test_fc_cut = prop_test_fc[-len(daily_test) :]
            prop_test_m = compute_metrics(daily_test.to_numpy(), prop_test_fc_cut)
            print_metrics("Prophet/TEST", prop_test_m)
            plot_actual_vs_pred(daily_test.index, daily_test.to_numpy(), prop_test_fc_cut, "Prophet 테스트 예측 vs 실제 (일)")
    else:
        print("Prophet 미설치로 스킵")

    # 간단 성능 비교 요약
    print("\n=== 모델 성능 요약(테스트) ===")
    print_metrics("Linear/TEST", lin_test_m)
    print_metrics("RF/TEST", rf_test_m)
    print_metrics("SARIMA/TEST", sar_test_m)
    if _HAS_PROPHET:
        try:
            print_metrics("Prophet/TEST", prop_test_m)  # type: ignore
        except Exception:
            pass


if __name__ == "__main__":
    main()


