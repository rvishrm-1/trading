"""
Data module for multi-timeframe trading neural network.
Provides synthetic data generation (20 months: 10 months train, 10 months test)
and CSV loading capabilities for 15m and HTF (High Timeframe) data.
"""

from typing import Tuple, Optional
import numpy as np
import pandas as pd


def generate_synthetic_data(
    start_date: str = "2023-01-01",
    total_days: int = 600,  # ~20 months (300 days train, 300 days test)
    htf_freq: str = "4h",
    initial_price: float = 30000.0,
    seed: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Generates synthetic market data for 15-minute timeframe and High Timeframe (HTF).

    Includes:
    - OHLCV (Open, High, Low, Close, Volume)
    - Open Interest (OI)
    - Cumulative Volume Delta (CVD)
    """
    np.random.seed(seed)

    # 15-minute timestamps
    dates_15m = pd.date_range(start=start_date, periods=total_days * 24 * 4, freq="15min")
    n = len(dates_15m)

    # Simulate price random walk with momentum and regime shifts
    returns = np.random.normal(0.00005, 0.002, size=n)
    # Add a sinusoidal trend component to simulate market cycles
    cycles = 0.001 * np.sin(np.linspace(0, 10 * np.pi, n))
    returns += cycles

    price_path = initial_price * np.exp(np.cumsum(returns))

    # Construct 15m OHLCV
    noise_h = np.abs(np.random.normal(0, 0.001, size=n))
    noise_l = np.abs(np.random.normal(0, 0.001, size=n))
    noise_c = np.random.normal(0, 0.0005, size=n)

    close_15m = price_path
    open_15m = np.roll(close_15m, 1)
    open_15m[0] = initial_price
    high_15m = np.maximum(open_15m, close_15m) * (1 + noise_h)
    low_15m = np.minimum(open_15m, close_15m) * (1 - noise_l)

    # Volume and Order Flow (CVD, OI)
    base_volume = np.random.lognormal(mean=3, sigma=0.8, size=n) * 10
    # Buy volume vs Sell volume delta
    delta = base_volume * np.random.uniform(-0.5, 0.5, size=n) + (returns * base_volume * 50)
    cvd = np.cumsum(delta)

    # Open Interest simulation correlated with price changes and volume
    oi_changes = np.random.normal(0, 100, size=n) + (np.abs(returns) * 5000)
    open_interest = np.maximum(10000, 50000 + np.cumsum(oi_changes))

    df_15m = pd.DataFrame({
        'timestamp': dates_15m,
        'open': open_15m,
        'high': high_15m,
        'low': low_15m,
        'close': close_15m,
        'volume': base_volume,
        'open_interest': open_interest,
        'cvd': cvd
    })

    # Resample to HTF
    df_htf = df_15m.set_index('timestamp').resample(htf_freq).agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum',
        'open_interest': 'last',
        'cvd': 'last'
    }).reset_index().dropna()

    return df_15m, df_htf


def split_train_test_by_months(
    df: pd.DataFrame,
    train_months: int = 10,
    test_months: int = 10
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Splits dataframe into train and test sets based on exact date cutoff or month duration.
    """
    start_date = df['timestamp'].min()
    train_end = start_date + pd.DateOffset(months=train_months)
    test_end = train_end + pd.DateOffset(months=test_months)

    train_df = df[(df['timestamp'] >= start_date) & (df['timestamp'] < train_end)].copy()
    test_df = df[(df['timestamp'] >= train_end) & (df['timestamp'] <= test_end)].copy()

    # Fallback in case DateOffset overshoots/undershoots exact len
    if len(test_df) == 0:
        midpoint = len(df) // 2
        train_df = df.iloc[:midpoint].copy()
        test_df = df.iloc[midpoint:].copy()

    return train_df, test_df


def load_data_from_csv(filepath_15m: str, filepath_htf: Optional[str] = None, htf_freq: str = "4h") -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Loads custom 15m data from CSV and optionally generates or loads HTF data.
    """
    df_15m = pd.read_csv(filepath_15m)
    df_15m['timestamp'] = pd.to_datetime(df_15m['timestamp'])
    df_15m = df_15m.sort_values('timestamp').reset_index(drop=True)

    if filepath_htf:
        df_htf = pd.read_csv(filepath_htf)
        df_htf['timestamp'] = pd.to_datetime(df_htf['timestamp'])
        df_htf = df_htf.sort_values('timestamp').reset_index(drop=True)
    else:
        df_htf = df_15m.set_index('timestamp').resample(htf_freq).agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum',
            'open_interest': 'last' if 'open_interest' in df_15m else 'first',
            'cvd': 'last' if 'cvd' in df_15m else 'first'
        }).reset_index().dropna()

    return df_15m, df_htf
