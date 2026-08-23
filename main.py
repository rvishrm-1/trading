"""
Main execution script for the Multi-Timeframe Trading Neural Network.

Workflow:
1. Generate synthetic data OR fetch real market data (e.g. BTC-USD, ETH-USD) OR load from CSV.
2. Compute HTF indicators (MACD, Supertrend) and 15m order flow features (OI, CVD).
3. Split data into training period and testing period using chronological splitting.
4. Train PyTorch neural network on historical training set.
5. Predict signals on unseen out-of-sample test set.
6. Execute backtest with Stop-Loss (-1.5%) and Take-Profit (+3.0%) risk management rules.
7. Print detailed out-of-sample backtest report with all required metrics.
"""

import sys
import argparse
import pandas as pd
import numpy as np

from data import generate_synthetic_data, fetch_real_data, split_train_test_by_months, load_data_from_csv
from indicators import compute_htf_indicators, compute_15m_features
from model import train_model, predict_signals
from backtest import Backtester


def parse_args():
    parser = argparse.ArgumentParser(description="Multi-Timeframe Neural Network Trading System")
    parser.add_argument("--real", action="store_true", help="Fetch real historical market data")
    parser.add_argument("--symbol", type=str, default="BTC-USD", help="Symbol for real market data (default: BTC-USD)")
    parser.add_argument("--period", type=str, default="60d", help="Data download period for real data (default: 60d)")
    parser.add_argument("--csv_15m", type=str, default=None, help="Path to custom 15m CSV data file")
    parser.add_argument("--csv_htf", type=str, default=None, help="Path to custom HTF CSV data file")
    parser.add_argument("--htf_freq", type=str, default="4h", help="HTF frequency (default: 4h)")
    parser.add_argument("--epochs", type=int, default=15, help="Number of training epochs (default: 15)")
    parser.add_argument("--initial_capital", type=float, default=10000.0, help="Initial capital ($)")
    parser.add_argument("--stop_loss", type=float, default=0.015, help="Stop Loss fraction (default: 0.015 for 1.5%%)")
    parser.add_argument("--risk_reward", type=float, default=2.0, help="Risk Reward ratio (default: 2.0 for 2:1 R:R, TP = 3.0%%)")
    return parser.parse_args()


def main():
    args = parse_args()

    print("=" * 65)
    print(" MULTI-TIMEFRAME TRADING NEURAL NETWORK PIPELINE ")
    print(" HTF Directional Bias: MACD & Supertrend")
    print(" 15m Execution Signals: Open Interest (OI) & CVD Dynamics")
    print("=" * 65)

    # 1. Load or Generate Data
    if args.real:
        print(f"\n[1/5] Downloading REAL historical market data for {args.symbol} (period={args.period})...")
        df_15m, df_htf = fetch_real_data(symbol=args.symbol, period=args.period, interval="15m", htf_freq=args.htf_freq)
    elif args.csv_15m:
        print(f"\n[1/5] Loading data from CSV: {args.csv_15m}...")
        df_15m, df_htf = load_data_from_csv(args.csv_15m, args.csv_htf, htf_freq=args.htf_freq)
    else:
        print("\n[1/5] Generating 20 months synthetic market data (15m & 4h HTF)...")
        df_15m, df_htf = generate_synthetic_data(total_days=600, htf_freq=args.htf_freq)

    # 2. Compute Indicators
    print("\n[2/5] Computing HTF bias (MACD & Supertrend) and 15m order flow (OI & CVD)...")
    df_htf = compute_htf_indicators(df_htf)
    df_15m = compute_15m_features(df_15m, df_htf)

    # 3. Chronological Train & Test Split
    print("\n[3/5] Chronological splitting of time-series data into Train and Test sets...")
    train_df, test_df = split_train_test_by_months(df_15m, train_months=10, test_months=10)

    train_start = train_df['timestamp'].min()
    train_end = train_df['timestamp'].max()
    test_start = test_df['timestamp'].min()
    test_end = test_df['timestamp'].max()

    print(f"  Training Period:  {train_start} to {train_end} ({len(train_df)} 15m candles)")
    print(f"  Testing Period:   {test_start} to {test_end} ({len(test_df)} 15m candles)")

    # 4. Train Neural Network Model
    print(f"\n[4/5] Training PyTorch Neural Network for {args.epochs} epochs on historical train set...")
    model, scaler = train_model(train_df, epochs=args.epochs)
    print("  Model training complete.")

    # 5. Predict & Backtest on Unseen Out-of-Sample Test Set
    print("\n[5/5] Predicting signals and executing out-of-sample backtest...")
    test_signals = predict_signals(model, scaler, test_df)

    backtester = Backtester(
        initial_capital=args.initial_capital,
        stop_loss_pct=args.stop_loss,
        risk_reward_ratio=args.risk_reward
    )
    results = backtester.run(test_df, test_signals)

    # Print Full Backtest Results
    data_source_label = f"REAL HISTORICAL DATA ({args.symbol})" if args.real else ("CUSTOM CSV" if args.csv_15m else "SYNTHETIC MARKET DATA")
    tp_pct_str = f"+{args.stop_loss * args.risk_reward * 100:.2f}%"
    sl_pct_str = f"-{args.stop_loss * 100:.2f}%"

    print("\n" + "=" * 65)
    print("           OUT-OF-SAMPLE BACKTEST PROFITABILITY REPORT           ")
    print("=" * 65)
    print(f" Asset / Dataset:          {data_source_label}")
    print(f" Training Date Range:      {train_start} to {train_end}")
    print(f" Testing Date Range:       {test_start} to {test_end}")
    print(f" Take-Profit (TP):         {tp_pct_str}")
    print(f" Stop-Loss (SL):           {sl_pct_str}")
    print(f" Risk / Reward Ratio:      {args.risk_reward:.1f}:1")
    print("-" * 65)
    print(f" Initial Portfolio Value:  ${results['initial_capital']:,.2f}")
    print(f" Final Portfolio Value:    ${results['final_capital']:,.2f}")
    print(f" Total Net Return:         {results['total_return_pct']:+.2f}%")
    print(f" Buy-and-Hold Return:      {results['buy_and_hold_return_pct']:+.2f}%")
    print("-" * 65)
    print(f" Total Trades:             {results['total_trades']}")
    print(f" Winning Trades:           {results['winning_trades']}")
    print(f" Losing Trades:            {results['losing_trades']}")
    print(f" Win Rate:                 {results['win_rate_pct']:.2f}%")
    print(f" Avg Return per Trade:     {results['avg_return_per_trade_pct']:+.2f}%")
    print(f" TP Hits:                  {results['tp_hits']}")
    print(f" SL Hits:                  {results['sl_hits']}")
    print(f" Profit Factor:            {results['profit_factor']:.2f}")
    print(f" Maximum Drawdown:         {results['max_drawdown_pct']:.2f}%")
    print(f" Sharpe Ratio:             {results['sharpe_ratio']:.2f}")
    print("=" * 65 + "\n")


if __name__ == "__main__":
    main()
