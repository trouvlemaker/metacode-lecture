"""
Chapter 2-3, 2-2강 시계열 예측 모델링 — PyTorch LSTM (Bike Sharing)

목표
- 시계열 데이터를 시퀀스 학습 형태로 변환하고, LSTM 기반 딥러닝 모델을 구축/학습/평가합니다.

규칙(강의용)
- 시각화는 matplotlib만 사용 (seaborn/색상지정/서브플롯 X)
- PyTorch로 구현 (GPU 없어도 동작)

데이터
- Kaggle Bike Sharing Demand (시간 단위, target: count)
"""

from __future__ import annotations

import os
import warnings
from typing import Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader


warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)


def _set_seed(seed: int = 42) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_hourly_data() -> pd.DataFrame:
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
    return df


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["year"] = out["datetime"].dt.year
    out["month"] = out["datetime"].dt.month
    out["day"] = out["datetime"].dt.day
    out["dayofweek"] = out["datetime"].dt.dayofweek
    out["hour"] = out["datetime"].dt.hour
    return out


def split_by_time(df: pd.DataFrame, train_ratio: float = 0.8, val_ratio: float = 0.1):
    n = len(df)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    train = df.iloc[:n_train]
    val = df.iloc[n_train : n_train + n_val]
    test = df.iloc[n_train + n_val :]
    return train, val, test


def build_feature_matrix(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, list[str]]:
    feature_cols = [
        "temp",
        "atemp",
        "humidity",
        "windspeed",
        "season",
        "holiday",
        "workingday",
        "weather",
        "year",
        "month",
        "dayofweek",
        "hour",
    ]
    X = df[feature_cols].to_numpy(dtype=np.float32)
    y = df["count"].astype(np.float32).to_numpy()
    return X, y, feature_cols


def make_windows(X: np.ndarray, y: np.ndarray, window: int, horizon: int = 1) -> Tuple[np.ndarray, np.ndarray]:
    xs, ys = [], []
    for i in range(len(X) - window - horizon + 1):
        xs.append(X[i : i + window])
        ys.append(y[i + window : i + window + horizon])
    return np.asarray(xs, dtype=np.float32), np.asarray(ys, dtype=np.float32)


class SequenceDataset(Dataset):
    def __init__(self, X_seq: np.ndarray, y_seq: np.ndarray):
        self.X = X_seq
        self.y = y_seq

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int):
        return self.X[idx], self.y[idx]


class LSTMRegressor(nn.Module):
    def __init__(self, num_features: int, hidden_size: int = 64, num_layers: int = 2, dropout: float = 0.1):
        super().__init__()
        self.lstm = nn.LSTM(input_size=num_features, hidden_size=hidden_size, num_layers=num_layers, batch_first=True, dropout=dropout)
        self.head = nn.Linear(hidden_size, 1)

    def forward(self, x):  # x: (B, T, F)
        out, _ = self.lstm(x)
        last = out[:, -1, :]
        y = self.head(last)
        return y.squeeze(-1)


def train_model(model: nn.Module, train_loader: DataLoader, val_loader: DataLoader, epochs: int, lr: float, device: torch.device):
    criterion = nn.MSELoss()
    optim = torch.optim.Adam(model.parameters(), lr=lr)
    best_val = float("inf")
    best_state = None
    for ep in range(1, epochs + 1):
        model.train()
        loss_sum, n = 0.0, 0
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            pred = model(xb)
            loss = criterion(pred, yb)
            optim.zero_grad()
            loss.backward()
            optim.step()
            loss_sum += float(loss.item()) * len(xb)
            n += len(xb)
        train_loss = loss_sum / max(n, 1)

        model.eval()
        with torch.no_grad():
            val_sum, n2 = 0.0, 0
            for xb, yb in val_loader:
                xb = xb.to(device)
                yb = yb.to(device)
                pred = model(xb)
                loss = criterion(pred, yb)
                val_sum += float(loss.item()) * len(xb)
                n2 += len(xb)
        val_loss = val_sum / max(n2, 1)
        print(f"Epoch {ep:03d} - train MSE: {train_loss:.4f}, val MSE: {val_loss:.4f}")
        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)


def predict_all(model: nn.Module, loader: DataLoader, device: torch.device) -> np.ndarray:
    model.eval()
    outs: list[np.ndarray] = []
    with torch.no_grad():
        for xb, _ in loader:
            xb = xb.to(device)
            pred = model(xb)
            outs.append(pred.cpu().numpy())
    return np.concatenate(outs, axis=0)


def plot_series(dt_index: pd.DatetimeIndex, y_true: np.ndarray, y_pred: np.ndarray, title: str) -> None:
    plt.figure(figsize=(12, 4))
    plt.plot(dt_index, y_true, label="Actual")
    plt.plot(dt_index, y_pred, label="Pred")
    plt.title(title)
    plt.xlabel("Time")
    plt.ylabel("Count")
    plt.legend()
    plt.tight_layout()
    plt.show()


def main() -> None:
    _set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    df = load_hourly_data()
    df = add_time_features(df)

    train_df, val_df, test_df = split_by_time(df, 0.8, 0.1)
    X_train, y_train, feat_cols = build_feature_matrix(train_df)
    X_val, y_val, _ = build_feature_matrix(val_df)
    X_test, y_test, _ = build_feature_matrix(test_df)

    # 표준화(간단): train 평균/표준편차로 정규화
    mean = X_train.mean(axis=0, keepdims=True)
    std = X_train.std(axis=0, keepdims=True) + 1e-8
    X_train = (X_train - mean) / std
    X_val = (X_val - mean) / std
    X_test = (X_test - mean) / std

    window = 24
    Xtr_seq, ytr_seq = make_windows(X_train, y_train, window, horizon=1)
    Xva_seq, yva_seq = make_windows(X_val, y_val, window, horizon=1)
    Xte_seq, yte_seq = make_windows(X_test, y_test, window, horizon=1)

    train_loader = DataLoader(SequenceDataset(Xtr_seq, ytr_seq.squeeze(-1) if ytr_seq.ndim > 1 else ytr_seq), batch_size=128, shuffle=True)
    val_loader = DataLoader(SequenceDataset(Xva_seq, yva_seq.squeeze(-1) if yva_seq.ndim > 1 else yva_seq), batch_size=256, shuffle=False)
    test_loader = DataLoader(SequenceDataset(Xte_seq, yte_seq.squeeze(-1) if yte_seq.ndim > 1 else yte_seq), batch_size=256, shuffle=False)

    model = LSTMRegressor(num_features=Xtr_seq.shape[-1], hidden_size=64, num_layers=2, dropout=0.1).to(device)
    train_model(model, train_loader, val_loader, epochs=20, lr=1e-3, device=device)

    val_pred = predict_all(model, val_loader, device)
    test_pred = predict_all(model, test_loader, device)

    # 시각화 (검증/테스트 구간의 마지막 타임스텝만 대응)
    val_idx = val_df["datetime"].iloc[window:].values
    test_idx = test_df["datetime"].iloc[window:].values
    plot_series(val_idx, yva_seq.squeeze(-1), val_pred, "LSTM 검증 예측 vs 실제")
    plot_series(test_idx, yte_seq.squeeze(-1), test_pred, "LSTM 테스트 예측 vs 실제")


if __name__ == "__main__":
    main()


