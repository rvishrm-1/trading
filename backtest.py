"""
Backtesting module for multi-timeframe trading system.

Simulates trade execution based on model signals on 15m candles:
- Signal 0: Neutral / Exit position
- Signal 1: Go Long
- Signal 2: Go Short

Calculates key performance metrics:
- Total Return %
- Win Rate %
- Total Trades
- Sharpe Ratio
- Maximum Drawdown %
- Profit Factor
"""

from typing import Dict, Any
import numpy as np
import pandas as pd


class Backtester:
    def __init__(
        self,
        initial_capital: float = 10000.0,
        fee_rate: float = 0.0006,  # 0.06% taker fee
        slippage: float = 0.0002,  # 0.02% slippage
        position_size: float = 1.0  # Fraction of capital per trade
    ):
        self.initial_capital = initial_capital
        self.fee_rate = fee_rate
        self.slippage = slippage
        self.position_size = position_size

    def run(self, df: pd.DataFrame, signals: np.ndarray) -> Dict[str, Any]:
        """
        Runs backtest given dataframe with OHLCV data and signals array.
        signals: 0 = Flat, 1 = Long, 2 = Short
        """
        df = df.copy().reset_index(drop=True)
        n = len(df)

        capital = self.initial_capital
        equity_curve = [capital]

        current_pos = 0  # 0 = cash, 1 = long, -1 = short
        entry_price = 0.0
        position_units = 0.0

        trades = []  # Record individual trades: pnl, return_pct, holding_period
        trade_entry_idx = 0

        close_prices = df['close'].values
        timestamps = df['timestamp'].values

        for i in range(n):
            sig = signals[i]
            target_pos = 0
            if sig == 1:
                target_pos = 1
            elif sig == 2:
                target_pos = -1

            price = close_prices[i]

            # Check if position needs to be changed
            if target_pos != current_pos:
                # Close existing position if open
                if current_pos != 0:
                    exit_price = price * (1 - self.slippage) if current_pos == 1 else price * (1 + self.slippage)
                    if current_pos == 1:
                        pnl = (exit_price - entry_price) * position_units
                    else:
                        pnl = (entry_price - exit_price) * position_units

                    fee = exit_price * position_units * self.fee_rate
                    net_pnl = pnl - fee
                    capital += net_pnl

                    trade_return = net_pnl / (entry_price * position_units) if (entry_price * position_units) > 0 else 0
                    trades.append({
                        'type': 'Long' if current_pos == 1 else 'Short',
                        'entry_time': timestamps[trade_entry_idx],
                        'exit_time': timestamps[i],
                        'entry_price': entry_price,
                        'exit_price': exit_price,
                        'pnl': net_pnl,
                        'return_pct': trade_return
                    })

                    current_pos = 0
                    position_units = 0.0

                # Open new position if target_pos is not 0
                if target_pos != 0:
                    current_pos = target_pos
                    trade_entry_idx = i
                    entry_price = price * (1 + self.slippage) if current_pos == 1 else price * (1 - self.slippage)

                    allocated_capital = capital * self.position_size
                    position_units = allocated_capital / entry_price
                    entry_fee = entry_price * position_units * self.fee_rate
                    capital -= entry_fee

            # Track equity curve
            if current_pos == 1:
                unrealized_pnl = (price - entry_price) * position_units
                current_equity = capital + unrealized_pnl
            elif current_pos == -1:
                unrealized_pnl = (entry_price - price) * position_units
                current_equity = capital + unrealized_pnl
            else:
                current_equity = capital

            equity_curve.append(current_equity)

        # Force close open position at the end
        if current_pos != 0:
            price = close_prices[-1]
            exit_price = price * (1 - self.slippage) if current_pos == 1 else price * (1 + self.slippage)
            pnl = (exit_price - entry_price) * position_units if current_pos == 1 else (entry_price - exit_price) * position_units
            fee = exit_price * position_units * self.fee_rate
            net_pnl = pnl - fee
            capital += net_pnl
            trades.append({
                'type': 'Long' if current_pos == 1 else 'Short',
                'entry_time': timestamps[trade_entry_idx],
                'exit_time': timestamps[-1],
                'entry_price': entry_price,
                'exit_price': exit_price,
                'pnl': net_pnl,
                'return_pct': net_pnl / (entry_price * position_units)
            })

        equity_curve = np.array(equity_curve)
        total_return_pct = ((capital - self.initial_capital) / self.initial_capital) * 100.0

        # Calculate Win Rate & Profit Factor
        if len(trades) > 0:
            pnls = [t['pnl'] for t in trades]
            winning_trades = [p for p in pnls if p > 0]
            losing_trades = [p for p in pnls if p <= 0]
            win_rate = (len(winning_trades) / len(trades)) * 100.0
            gross_profit = sum(winning_trades)
            gross_loss = abs(sum(losing_trades))
            profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else np.nan
        else:
            win_rate = 0.0
            profit_factor = 0.0

        # Drawdown calculation
        peak = np.maximum.accumulate(equity_curve)
        drawdown = (equity_curve - peak) / peak
        max_drawdown_pct = abs(np.min(drawdown)) * 100.0 if len(drawdown) > 0 else 0.0

        # Sharpe Ratio (annualized for 15m candles: 35040 candles per year)
        equity_returns = pd.Series(equity_curve).pct_change().dropna()
        if len(equity_returns) > 1 and equity_returns.std() > 0:
            sharpe_ratio = (equity_returns.mean() / equity_returns.std()) * np.sqrt(35040)
        else:
            sharpe_ratio = 0.0

        return {
            'initial_capital': self.initial_capital,
            'final_capital': capital,
            'total_return_pct': total_return_pct,
            'total_trades': len(trades),
            'win_rate_pct': win_rate,
            'profit_factor': profit_factor,
            'max_drawdown_pct': max_drawdown_pct,
            'sharpe_ratio': sharpe_ratio,
            'trades': trades,
            'equity_curve': equity_curve
        }
