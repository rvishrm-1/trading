"""
Main execution script for the Multi-Timeframe Trading Neural Network.

Workflow:
1. Generate synthetic data OR fetch real market data (e.g. BTC-USD, ETH-USD) OR load from CSV.
2. Compute HTF indicators (MACD, Supertrend) and 15m order flow features (OI, CVD).
3. Split data into training period and testing period.
4. Train PyTorch neural network on historical training set.
5. Predict signals on unseen out-of-sample test set.
6. Execute backtest with optional Risk-Reward Ratio & Stop-Loss risk management.
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
    parser.add_argument("--stop_loss", type=float, default=None, help="Stop Loss fraction e.g. 0.01 for 1%%")
    parser.add_argument("--risk_reward", type=float, default=None, help="Risk Reward ratio e.g. 2.0 for 1:2 R:R (TP = SL * RR)")
    return parser.parse_args()


def main():
    args = parse_args()

    print("=" * 60)
    print(" MULTI-TIMEFRAME TRADING NEURAL NETWORK PIPELINE ")
    print(" HTF Indicators: MACD & Supertrend")
    print(" 15m Indicators: Open Interest (OI) & CVD")
    print("=" * 60)

    # 1. Load or Generate Data
    if args.real:
        print(f"\n[1/5] Fetching REAL market data for {args.symbol} (period={args.period})...")
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

    # 3. Split into Train & Test
    print("\n[3/5] Splitting data into training and testing datasets...")
    train_df, test_df = split_train_test_by_months(df_15m, train_months=10, test_months=10)

    print(f"  Training Set:   {train_df['timestamp'].min()} to {train_df['timestamp'].max()} ({len(train_df)} rows)")
    print(f"  Testing Set:    {test_df['timestamp'].min()} to {test_df['timestamp'].max()} ({len(test_df)} rows)")

    # 4. Train Neural Network Model
    print(f"\n[4/5] Training PyTorch Neural Network for {args.epochs} epochs on train set...")
    model, scaler = train_model(train_df, epochs=args.epochs)
    print("  Model training complete.")

    # 5. Predict & Backtest on Test Set
    print("\n[5/5] Predicting trading signals and running backtest on test set...")
    test_signals = predict_signals(model, scaler, test_df)

    backtester = Backtester(
        initial_capital=args.initial_capital,
        stop_loss_pct=args.stop_loss,
        risk_reward_ratio=args.risk_reward
    )
    results = backtester.run(test_df, test_signals)

    # Print Profitability Report
    data_source_label = f"REAL DATA ({args.symbol})" if args.real else ("CUSTOM CSV" if args.csv_15m else "SYNTHETIC DATA")
    rr_str = f"1:{args.risk_reward:.1f}" if args.risk_reward else "None"
    sl_str = f"{args.stop_loss * 100:.2f}%" if args.stop_loss else "None"

    print("\n" + "=" * 60)
    print(f"           OUT-OF-SAMPLE TEST PROFITABILITY REPORT           ")
    print(f" Source: {data_source_label}")
    print(f" Risk Management: SL={sl_str} | Risk-Reward={rr_str}")
    print("=" * 60)
    print(f" Initial Capital:         ${results['initial_capital']:,.2f}")
    print(f" Final Capital:           ${results['final_capital']:,.2f}")
    print(f" Total Net Return:        {results['total_return_pct']:+.2f}%")
    print(f" Total Trades Executed:   {results['total_trades']}")
    print(f" Win Rate:                {results['win_rate_pct']:.2f}%")
    print(f" Profit Factor:           {results['profit_factor']:.2f}")
    print(f" Maximum Drawdown:        {results['max_drawdown_pct']:.2f}%")
    print(f" Sharpe Ratio:            {results['sharpe_ratio']:.2f}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
