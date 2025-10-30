from ib_insync import *
import pandas as pd
import numpy as np
import itertools

# --- CONFIG ---
symbol = 'QQQ'
exchange = 'ARCA'
currency = 'USD'
candle_size = '4 hours'
period = '365 D'                # fetch enough data for walk-forward
take_profit_pct = 0.05
max_hold = 40
backtest_start = '2024-01-01'
backtest_end = '2025-10-01'

# Grid search ranges
z_range = np.arange(-0.5, -2.1, -0.5)
z_window_range = [10, 20, 30]
ema_fast_range = [10, 20]
ema_slow_range = [50, 60]

# Walk-forward config
train_days = 60
test_days = 20

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
    df = util.df(bars)
    df.set_index('date', inplace=True)
    df = df.sort_index()
    df['close'] = df['close'].astype(float)
    return df

def compute_buy_the_dip(df, z_thresh, z_window, ema_fast, ema_slow):
    df['ema_fast'] = df['close'].ewm(span=ema_fast, adjust=False).mean()
    df['ema_slow'] = df['close'].ewm(span=ema_slow, adjust=False).mean()
    uptrend = df['ema_fast'] > df['ema_slow']

    ema_val = df['ema_fast']
    std_val = df['close'].rolling(z_window).std()
    dip_z = (df['close'] - ema_val) / std_val
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

def calculate_pnl(df):
    signals = df[df['signal'].str.startswith('BUY') | df['signal'].str.startswith('SELL')]
    buy_prices = signals[signals['signal'].str.startswith('BUY')]['close'].values
    sell_prices = signals[signals['signal'].str.startswith('SELL')]['close'].values
    n_trades = min(len(buy_prices), len(sell_prices))
    profits = sell_prices[:n_trades] - buy_prices[:n_trades]
    total_pnl = profits.sum() if n_trades else 0
    win_rate = np.mean(profits > 0) if n_trades else 0
    return total_pnl, win_rate, n_trades

# --- Main Walk-Forward Grid Search ---
data = get_data(symbol, candle_size, period)
data = data[(data.index >= backtest_start) & (data.index <= backtest_end)]
results = []

start_idx = 0
while start_idx + train_days + test_days < len(data):
    train_df = data.iloc[start_idx:start_idx+train_days]
    test_df = data.iloc[start_idx+train_days:start_idx+train_days+test_days]

    # Grid search on training
    best_pnl = -np.inf
    best_params = None
    for z, zw, ef, es in itertools.product(z_range, z_window_range, ema_fast_range, ema_slow_range):
        temp = compute_buy_the_dip(train_df.copy(), z, zw, ef, es)
        pnl, _, _ = calculate_pnl(temp)
        if pnl > best_pnl:
            best_pnl = pnl
            best_params = (z, zw, ef, es)

    # Test best params on out-of-sample
    z, zw, ef, es = best_params
    test_temp = compute_buy_the_dip(test_df.copy(), z, zw, ef, es)
    pnl, win_rate, n_trades = calculate_pnl(test_temp)
    results.append({
        'train_start': train_df.index[0], 
        'train_end': train_df.index[-1],
        'test_start': test_df.index[0], 
        'test_end': test_df.index[-1],
        'z': z, 'z_window': zw, 'ema_fast': ef, 'ema_slow': es,
        'PnL': pnl, 'WinRate': win_rate, 'Trades': n_trades
    })

    start_idx += test_days  # roll forward

# --- Summary ---
df_results = pd.DataFrame(results)
print(df_results)
print("\nTotal Walk-Forward PnL:", df_results['PnL'].sum())
print("Average Win Rate:", df_results['WinRate'].mean())

ib.disconnect()
