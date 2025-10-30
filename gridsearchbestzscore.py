from ib_insync import *
import pandas as pd
import numpy as np
import itertools

# --- CONFIG ---
symbol = 'QQQ'
exchange = 'ARCA'
currency = 'USD'
period = '365 D'                 # Longer period for thorough test
candle_size = '4 hours'
ema_fast = 20
ema_slow = 50
take_profit_pct = 0.02
max_hold = 10
backtest_start = '2024-01-01'
backtest_end = '2025-10-01'

z_range = np.arange(-0.5, -2.6, -0.1)     # -0.5 to -2.5 step -0.1
win_range = range(10, 41, 5)              # 10, 15, ... 40

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

# --- Main Grid Search Loop ---
data = get_data(symbol, candle_size, period)
data = data[(data.index >= backtest_start) & (data.index <= backtest_end)]
results = []
for z_thresh, z_window in itertools.product(z_range, win_range):
    df = data.copy()
    df = compute_buy_the_dip(df, z_thresh, z_window)
    signals = df[df['signal'].str.startswith('BUY') | df['signal'].str.startswith('SELL')][['signal', 'close']]
    buy_prices = signals[signals['signal'].str.startswith('BUY')]['close'].values
    sell_prices = signals[signals['signal'].str.startswith('SELL')]['close'].values
    n_trades = min(len(buy_prices), len(sell_prices))
    profits = sell_prices[:n_trades] - buy_prices[:n_trades]
    total_pnl = profits.sum()
    mean_pnl = profits.mean() if n_trades else 0
    win_rate = np.mean(profits > 0) if n_trades else 0
    results.append({'z': z_thresh, 'window': z_window, 'PnL': total_pnl, 'Mean': mean_pnl, 'WinRate': win_rate, 'Trades': n_trades})

# --- Results ---
df_results = pd.DataFrame(results)
best = df_results.sort_values('PnL', ascending=False).iloc[0]
print("\nBest Parameters:")
print(f"Z threshold: {best['z']:.2f}, Window: {best['window']}, Trades: {best['Trades']}, Total PnL: {best['PnL']:.2f}, Win Rate: {best['WinRate']:.2%}")

ib.disconnect()
