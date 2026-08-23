"""
PyTorch Neural Network model for multi-timeframe trading.

Combines:
1) HTF directional bias (MACD & Supertrend)
2) 15-minute granular features (OI changes, CVD momentum, price dynamics)

Outputs probabilities/logits for 3 classes:
0: Flat / Cash
1: Long
2: Short
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from typing import Tuple, List, Dict, Any


FEATURE_COLS = [
    'returns_15m', 'close_to_sma',
    'oi_change_1', 'oi_change_4', 'oi_ratio',
    'cvd_diff_1', 'cvd_diff_4', 'cvd_momentum',
    'htf_macd_hist', 'htf_supertrend_dir', 'htf_bias_score'
]


class MultiTimeframeDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class MultiTimeframeTradingNN(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 64, num_classes: int = 3, dropout: float = 0.2):
        super(MultiTimeframeTradingNN, self).__init__()

        # HTF sub-network branch (taking HTF bias features: last 3 features)
        self.htf_branch = nn.Sequential(
            nn.Linear(3, 16),
            nn.ReLU(),
            nn.BatchNorm1d(16)
        )

        # 15m order flow sub-network branch (taking OI, CVD, price features: first input_dim - 3 features)
        self.ltf_branch = nn.Sequential(
            nn.Linear(input_dim - 3, 32),
            nn.ReLU(),
            nn.BatchNorm1d(32),
            nn.Dropout(dropout)
        )

        # Combined execution network
        self.combined_net = nn.Sequential(
            nn.Linear(16 + 32, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        ltf_x = x[:, :-3]
        htf_x = x[:, -3:]

        ltf_out = self.ltf_branch(ltf_x)
        htf_out = self.htf_branch(htf_x)

        combined = torch.cat([ltf_out, htf_out], dim=1)
        logits = self.combined_net(combined)
        return logits


def create_labels(df: pd.DataFrame, forward_horizon: int = 4, threshold: float = 0.002) -> pd.Series:
    """
    Creates target labels based on future N-period return:
    0: Flat / Neutral
    1: Long (+return > threshold)
    2: Short (-return < -threshold)
    """
    future_return = df['close'].shift(-forward_horizon) / df['close'] - 1.0
    labels = pd.Series(0, index=df.index)
    labels[future_return > threshold] = 1
    labels[future_return < -threshold] = 2
    return labels


def train_model(
    df_train: pd.DataFrame,
    feature_cols: List[str] = FEATURE_COLS,
    epochs: int = 15,
    batch_size: int = 128,
    lr: float = 0.001,
    forward_horizon: int = 4,
    threshold: float = 0.002
) -> Tuple[MultiTimeframeTradingNN, StandardScaler]:
    """
    Trains the multi-timeframe neural network model.
    """
    df = df_train.copy().dropna(subset=feature_cols)
    y = create_labels(df, forward_horizon=forward_horizon, threshold=threshold).values

    # Remove trailing samples where target couldn't be calculated
    X = df[feature_cols].values[:-forward_horizon]
    y = y[:-forward_horizon]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    dataset = MultiTimeframeDataset(X_scaled, y)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    input_dim = len(feature_cols)
    model = MultiTimeframeTradingNN(input_dim=input_dim)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)

    model.train()
    for epoch in range(epochs):
        total_loss = 0.0
        for batch_x, batch_y in dataloader:
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(batch_y)

    return model, scaler


def predict_signals(
    model: MultiTimeframeTradingNN,
    scaler: StandardScaler,
    df: pd.DataFrame,
    feature_cols: List[str] = FEATURE_COLS
) -> np.ndarray:
    """
    Generates predicted trading signals (0: Flat, 1: Long, 2: Short) for input dataframe.
    """
    model.eval()
    X = df[feature_cols].fillna(0).values
    X_scaled = scaler.transform(X)
    tensor_x = torch.tensor(X_scaled, dtype=torch.float32)

    with torch.no_grad():
        logits = model(tensor_x)
        probs = torch.softmax(logits, dim=1)
        preds = torch.argmax(probs, dim=1).numpy()

    return preds
