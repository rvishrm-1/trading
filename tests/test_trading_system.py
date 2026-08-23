"""
Unit tests for multi-timeframe trading system components.
"""

import os
import unittest
import numpy as np
import pandas as pd
import torch

from data import generate_synthetic_data, split_train_test_by_months, load_data_from_csv
from indicators import compute_macd, compute_supertrend, compute_htf_indicators, compute_15m_features
from model import MultiTimeframeTradingNN, create_labels, train_model, predict_signals
from backtest import Backtester


class TestDataModule(unittest.TestCase):
    def test_synthetic_data_generation(self):
        df_15m, df_htf = generate_synthetic_data(total_days=30, seed=123)
        self.assertFalse(df_15m.empty)
        self.assertFalse(df_htf.empty)
        self.assertIn('open_interest', df_15m.columns)
        self.assertIn('cvd', df_15m.columns)

    def test_train_test_split(self):
        df_15m, _ = generate_synthetic_data(total_days=600, seed=123)
        train_df, test_df = split_train_test_by_months(df_15m, train_months=10, test_months=10)
        self.assertGreater(len(train_df), 0)
        self.assertGreater(len(test_df), 0)
        self.assertLess(train_df['timestamp'].max(), test_df['timestamp'].min())


class TestIndicatorsModule(unittest.TestCase):
    def test_indicators_computation(self):
        df_15m, df_htf = generate_synthetic_data(total_days=10, seed=42)
        df_htf = compute_htf_indicators(df_htf)
        self.assertIn('macd_hist', df_htf.columns)
        self.assertIn('supertrend_dir', df_htf.columns)
        self.assertIn('htf_bias_score', df_htf.columns)

        df_15m = compute_15m_features(df_15m, df_htf)
        self.assertIn('oi_change_1', df_15m.columns)
        self.assertIn('cvd_momentum', df_15m.columns)
        self.assertIn('htf_bias_score', df_15m.columns)


class TestModelModule(unittest.TestCase):
    def test_model_forward_pass(self):
        batch_size = 16
        input_dim = 11
        model = MultiTimeframeTradingNN(input_dim=input_dim)
        dummy_x = torch.randn(batch_size, input_dim)
        logits = model(dummy_x)
        self.assertEqual(logits.shape, (batch_size, 3))

    def test_training_and_prediction(self):
        df_15m, df_htf = generate_synthetic_data(total_days=30, seed=42)
        df_htf = compute_htf_indicators(df_htf)
        df_15m = compute_15m_features(df_15m, df_htf)
        model, scaler = train_model(df_15m, epochs=1, batch_size=32)
        preds = predict_signals(model, scaler, df_15m.iloc[:50])
        self.assertEqual(len(preds), 50)


class TestBacktestModule(unittest.TestCase):
    def test_backtester_metrics(self):
        df_15m, _ = generate_synthetic_data(total_days=10, seed=42)
        signals = np.zeros(len(df_15m), dtype=int)
        signals[10:20] = 1  # Long signal
        signals[30:40] = 2  # Short signal
        bt = Backtester(initial_capital=10000.0)
        results = bt.run(df_15m, signals)
        self.assertIn('total_return_pct', results)
        self.assertIn('win_rate_pct', results)
        self.assertIn('sharpe_ratio', results)


if __name__ == '__main__':
    unittest.main()
