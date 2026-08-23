"""
Indicators module for multi-timeframe trading system.

Computes:
- HTF Indicators:
  - MACD (MACD line, Signal line, Histogram)
  - Supertrend (Trend direction: +1 bullish, -1 bearish)
- 15m Indicators:
  - Open Interest (OI) changes, rate of change, moving average ratios
  - Cumulative Volume Delta (CVD) changes, momentum, moving averages
  - Merged HTF directional bias aligned onto 15m timeframe
"""

import numpy as np
import pandas as pd


def compute_macd(
    df: pd.DataFrame,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
    price_col: str = "close"
) -> pd.DataFrame:
    """
    Calculates MACD Line, Signal Line, and MACD Histogram.
    """
    df = df.copy()
    ema_fast = df[price_col].ewm(span=fast, adjust=False).mean()
    ema_slow = df[price_col].ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    macd_hist = macd_line - signal_line

    df['macd'] = macd_line
    df['macd_signal'] = signal_line
    df['macd_hist'] = macd_hist
    return df


def compute_supertrend(
    df: pd.DataFrame,
    period: int = 10,
    multiplier: float = 3.0
) -> pd.DataFrame:
    """
    Calculates Supertrend indicator.
    Returns dataframe with 'supertrend' value and 'supertrend_dir' (+1 for Long, -1 for Short).
    """
    df = df.copy()
    high = df['high']
    low = df['low']
    close = df['close']

    # Average True Range (ATR)
    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=period, min_periods=1).mean()

    # Basic Upper & Lower Bands
    hl2 = (high + low) / 2
    basic_ub = hl2 + (multiplier * atr)
    basic_lb = hl2 - (multiplier * atr)

    n = len(df)
    final_ub = np.zeros(n)
    final_lb = np.zeros(n)
    supertrend = np.zeros(n)
    st_dir = np.ones(n)  # 1 for bullish, -1 for bearish

    for i in range(1, n):
        # Upper band calculation
        if basic_ub.iloc[i] < final_ub[i - 1] or close.iloc[i - 1] > final_ub[i - 1]:
            final_ub[i] = basic_ub.iloc[i]
        else:
            final_ub[i] = final_ub[i - 1]

        # Lower band calculation
        if basic_lb.iloc[i] > final_lb[i - 1] or close.iloc[i - 1] < final_lb[i - 1]:
            final_lb[i] = basic_lb.iloc[i]
        else:
            final_lb[i] = final_lb[i - 1]

        # Trend direction
        if st_dir[i - 1] == 1:
            if close.iloc[i] < final_lb[i]:
                st_dir[i] = -1
                supertrend[i] = final_ub[i]
            else:
                st_dir[i] = 1
                supertrend[i] = final_lb[i]
        else:
            if close.iloc[i] > final_ub[i]:
                st_dir[i] = 1
                supertrend[i] = final_lb[i]
            else:
                st_dir[i] = -1
                supertrend[i] = final_ub[i]

    df['supertrend'] = supertrend
    df['supertrend_dir'] = st_dir
    return df


def compute_htf_indicators(df_htf: pd.DataFrame) -> pd.DataFrame:
    """
    Computes MACD and Supertrend on HTF data to create HTF Directional Bias.
    """
    df_htf = compute_macd(df_htf)
    df_htf = compute_supertrend(df_htf)

    # Combined HTF bias: +1 if both MACD hist > 0 & Supertrend bullish, -1 if both bearish, 0 if mixed/neutral
    macd_bias = np.where(df_htf['macd_hist'] > 0, 1, -1)
    st_bias = df_htf['supertrend_dir']

    # Signal score: average of normalized indicators
    df_htf['htf_bias_score'] = (macd_bias + st_bias) / 2.0
    return df_htf


def compute_15m_features(df_15m: pd.DataFrame, df_htf: pd.DataFrame) -> pd.DataFrame:
    """
    Computes 15m order flow features (OI and CVD dynamics) and forward-fills HTF bias into 15m timeline.
    """
    df = df_15m.copy()

    # 15m Technical Features
    df['returns_15m'] = df['close'].pct_change().fillna(0)
    df['sma_20'] = df['close'].rolling(20, min_periods=1).mean()
    df['close_to_sma'] = (df['close'] - df['sma_20']) / df['sma_20']

    # Open Interest (OI) Features
    df['oi_change_1'] = df['open_interest'].pct_change(1).fillna(0)
    df['oi_change_4'] = df['open_interest'].pct_change(4).fillna(0)  # 1 hour
    df['oi_sma_20'] = df['open_interest'].rolling(20, min_periods=1).mean()
    df['oi_ratio'] = df['open_interest'] / (df['oi_sma_20'] + 1e-8)

    # Cumulative Volume Delta (CVD) Features
    df['cvd_diff_1'] = df['cvd'].diff(1).fillna(0)
    df['cvd_diff_4'] = df['cvd'].diff(4).fillna(0)
    df['cvd_sma_20'] = df['cvd'].rolling(20, min_periods=1).mean()
    df['cvd_momentum'] = (df['cvd'] - df['cvd_sma_20']) / (df['volume'].rolling(20, min_periods=1).mean() + 1e-8)

    # Merge HTF directional bias into 15m dataframe (forward fill to avoid lookahead bias)
    htf_subset = df_htf[['timestamp', 'macd', 'macd_hist', 'supertrend_dir', 'htf_bias_score']].rename(
        columns={
            'macd': 'htf_macd',
            'macd_hist': 'htf_macd_hist',
            'supertrend_dir': 'htf_supertrend_dir',
            'htf_bias_score': 'htf_bias_score'
        }
    )

    df = pd.merge_asof(
        df.sort_values('timestamp'),
        htf_subset.sort_values('timestamp'),
        on='timestamp',
        direction='backward'
    )

    df['htf_bias_score'] = df['htf_bias_score'].fillna(0)
    df['htf_supertrend_dir'] = df['htf_supertrend_dir'].fillna(0)
    df['htf_macd_hist'] = df['htf_macd_hist'].fillna(0)

    return df
