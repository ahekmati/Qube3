from ib_insync import *
import pandas as pd
import numpy as np
import itertools
from datetime import timedelta

# --- CONFIG ---
symbol = 'QQQ'
exchange = 'ARCA'
currency = 'USD'
period = '2 Y'                     # IB requires a space format for longer durations
candle_size = '4 hours'
ema_fast = 20
ema_slow = 50
take_profit_pct = 0.05
max_hold = 10

z_range = np.arange(-0.5, -2.6, -0.1)     # -0.5 to -2.5 step -0.1
win_range = range(10, 41, 5)              # 10, 15, ... 40

# --- Walk-Forward Windows ---
in_sample_days = 90
out_sample_days = 30

tz = 'US/Eastern'
backtest_start = pd.Timestamp('2024-01-01', tz=tz)
backtest_end = pd.Timestamp('2025-10-01', tz=tz)

# --- IB Connection ---
ib = IB()
ib.connect('127.0.0.1', 4001, clientId=1)

def get_data(symbol, size, period):
    contract = Stock(symbol, exchange, currency)
    bars = ib.reqHistoricalData(
        contract,
        endDateTime='',
        durationStr=period,
        barSizeSetting=size,
        whatToShow='TRADES',
        useRTH=False,
        formatDate=1
    )
    if not bars:
        raise RuntimeError(
            "IB historical data request failed. Check 'period' format (e.g., '2 Y'), connection, and permissions."
        )
    df = util.df(bars)
    df.set_index('date', inplace=True)
    df = df.sort_index()
    df['close'] = df['close'].astype(float)
    # Ensure tz-aware index for compatibility
    if df.index.tz is None:
        df.index = df.index.tz_localize(tz)
    return df

def compute_buy_the_dip(df, z_thresh, z_window):
    df['ema_fast'] = df['close'].ewm(span=ema_fast, adjust=False).mean()
    df['ema_slow'] = df['close'].ewm(span=ema_slow, adjust=False).mean()
    uptrend = df['ema_fast'] > df['ema_slow']
    ema20 = df['ema_fast']
    std20 = df['close'].rolling(z_window).std()
    dip_z = (df['close'] - ema20) / std20
    dip_signal = dip_z < z_thresh
    buy_signals = uptrend & dip_signal
    df['buy_the_dip'] = buy_signals.fillna(False)
    df['signal'] = ''
    entry_dates = df.index[df['buy_the_dip']]
    for date in entry_dates:
        entry_price = df.at[date, 'close']
        df.at[date, 'signal'] = 'BUY'
        idx = df.index.get_loc(date)
        sold = False
        for j in range(idx+1, min(idx+max_hold+1, len(df))):
            tp_price = entry_price * (1 + take_profit_pct)
            closej = df['close'].iloc[j]
            if closej >= tp_price:
                df.at[df.index[j], 'signal'] = 'SELL (TP)'
                sold = True
                break
            elif df['ema_fast'].iloc[j] < df['ema_slow'].iloc[j]:
                df.at[df.index[j], 'signal'] = 'SELL (EMA CROSS)'
                sold = True
                break
        if not sold and (idx+max_hold < len(df)):
            df.at[df.index[idx+max_hold], 'signal'] = 'SELL (TIME)'
    return df

# --- Download and Prepare Data ---
data = get_data(symbol, candle_size, period)
data = data[(data.index >= backtest_start) & (data.index <= backtest_end)]

# --- Walk-Forward Analysis ---
current_start = backtest_start
walk_results = []
while current_start + timedelta(days=in_sample_days + out_sample_days) <= backtest_end:
    in_end = current_start + timedelta(days=in_sample_days)
    out_end = in_end + timedelta(days=out_sample_days)
    in_df = data[(data.index >= current_start) & (data.index < in_end)].copy()
    out_df = data[(data.index >= in_end) & (data.index < out_end)].copy()

    # Grid search on in-sample segment
    best_pnl = -np.inf
    best_params = None
    for z_thresh, z_window in itertools.product(z_range, win_range):
        df_test = compute_buy_the_dip(in_df.copy(), z_thresh, z_window)
        signals = df_test[df_test['signal'].str.startswith('BUY') | df_test['signal'].str.startswith('SELL')][['signal', 'close']]
        buy_prices = signals[signals['signal'].str.startswith('BUY')]['close'].values
        sell_prices = signals[signals['signal'].str.startswith('SELL')]['close'].values
        n_trades = min(len(buy_prices), len(sell_prices))
        profits = sell_prices[:n_trades] - buy_prices[:n_trades]
        total_pnl = profits.sum()
        if total_pnl > best_pnl:
            best_pnl = total_pnl
            best_params = (z_thresh, z_window)

    # Test best parameters on out-of-sample segment
    out_df_test = compute_buy_the_dip(out_df.copy(), *best_params)
    out_signals = out_df_test[out_df_test['signal'].str.startswith('BUY') | out_df_test['signal'].str.startswith('SELL')][['signal', 'close']]
    buy_prices = out_signals[out_signals['signal'].str.startswith('BUY')]['close'].values
    sell_prices = out_signals[out_signals['signal'].str.startswith('SELL')]['close'].values
    n_trades = min(len(buy_prices), len(sell_prices))
    profits = sell_prices[:n_trades] - buy_prices[:n_trades]
    total_pnl = profits.sum()
    mean_pnl = profits.mean() if n_trades else 0
    win_rate = np.mean(profits > 0) if n_trades else 0

    walk_results.append({
        'start': in_end.date(),
        'end': out_end.date(),
        'z': best_params[0],
        'window': best_params[1],
        'Trades': n_trades,
        'PnL': total_pnl,
        'Mean': mean_pnl,
        'WinRate': win_rate
    })

    print(f"Out-of-sample {in_end.date()} to {out_end.date()}: Z={best_params[0]:.2f}, Window={best_params[1]}, "
          f"Trades={n_trades}, PnL={total_pnl:.2f}, Win Rate={win_rate:.2%}")

    current_start += timedelta(days=out_sample_days)

# --- Aggregate Results ---
df_walk = pd.DataFrame(walk_results)
print("\nAggregate Walk-Forward Results:")
print(df_walk)
print("\nTotal Out-of-Sample PnL:", df_walk['PnL'].sum())

ib.disconnect()
