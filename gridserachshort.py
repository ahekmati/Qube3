from ib_insync import *
import pandas as pd
import numpy as np
import itertools

# --- CONFIG ---
symbol = 'QQQ'
exchange = 'ARCA'
currency = 'USD'
period = '365 D'                 
candle_size = '4 hours'
backtest_start = '2024-01-01'
backtest_end = '2025-10-01'

z_range = np.arange(0.5, 2.6, 0.1)            # +0.5 to +2.5 for shorts
win_range = range(10, 41, 5)                  
ema_fast_range = range(8, 29, 4)              
ema_slow_range = range(20, 71, 10)            
tp_range = [0.01, 0.02, 0.03, 0.05, 0.07]     
max_hold_range = [10, 15, 20, 40, 60]         

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
    if not bars or len(bars) == 0:
        raise RuntimeError("No data returned from IB.")
    df = util.df(bars)
    df.set_index('date', inplace=True)
    df = df.sort_index()
    df['close'] = df['close'].astype(float)
    return df

def compute_sell_the_rally(df, z_thresh, z_window, ema_fast, ema_slow, take_profit_pct, max_hold):
    df['ema_fast'] = df['close'].ewm(span=ema_fast, adjust=False).mean()
    df['ema_slow'] = df['close'].ewm(span=ema_slow, adjust=False).mean()
    downtrend = df['ema_fast'] < df['ema_slow']
    fast_ma = df['ema_fast']
    std_rolling = df['close'].rolling(z_window).std()
    dip_z = (df['close'] - fast_ma) / std_rolling
    rally_signal = dip_z > z_thresh
    sell_signals = downtrend & rally_signal
    df['sell_the_rally'] = sell_signals.fillna(False)
    df['signal'] = ''
    entry_dates = df.index[df['sell_the_rally']]
    for date in entry_dates:
        entry_price = df.at[date, 'close']
        df.at[date, 'signal'] = 'SELL'
        idx = df.index.get_loc(date)
        covered = False
        for j in range(idx+1, min(idx+max_hold+1, len(df))):
            tp_price = entry_price * (1 - take_profit_pct)  # TP for shorts (price drops)
            closej = df['close'].iloc[j]
            if closej <= tp_price:        # Price falls enough: take profit
                df.at[df.index[j], 'signal'] = 'COVER (TP)'
                covered = True
                break
            elif df['ema_fast'].iloc[j] > df['ema_slow'].iloc[j]:  # regime fail
                df.at[df.index[j], 'signal'] = 'COVER (EMA CROSS)'
                covered = True
                break
        if not covered and (idx+max_hold < len(df)):
            df.at[df.index[idx+max_hold], 'signal'] = 'COVER (TIME)'
    return df

data = get_data(symbol, candle_size, period)
data = data[(data.index >= backtest_start) & (data.index <= backtest_end)]

results = []
for (z_thresh, z_window, ema_fast, ema_slow, tp, max_hold) in itertools.product(
        z_range, win_range, ema_fast_range, ema_slow_range, tp_range, max_hold_range):
    if ema_fast >= ema_slow:  # Only test fast < slow for downtrend
        continue
    df = data.copy()
    df = compute_sell_the_rally(df, z_thresh, z_window, ema_fast, ema_slow, tp, max_hold)
    signals = df[df['signal'].str.startswith('SELL') | df['signal'].str.startswith('COVER')][['signal', 'close']]
    sell_prices = signals[signals['signal'].str.startswith('SELL')]['close'].values
    cover_prices = signals[signals['signal'].str.startswith('COVER')]['close'].values
    n_trades = min(len(sell_prices), len(cover_prices))
    if n_trades == 0:
        continue
    profits = sell_prices[:n_trades] - cover_prices[:n_trades]  # Short trade profit
    total_pnl = profits.sum()
    mean_pnl = profits.mean()
    win_rate = np.mean(profits > 0)
    results.append({
        'z': z_thresh,
        'window': z_window,
        'ema_fast': ema_fast,
        'ema_slow': ema_slow,
        'take_profit': tp,
        'max_hold': max_hold,
        'PnL': total_pnl,
        'Mean': mean_pnl,
        'WinRate': win_rate,
        'Trades': n_trades
    })

df_results = pd.DataFrame(results)
df_results = df_results.sort_values('PnL', ascending=False)
print("\nTop 10 Short Parameter Sets by Total PnL:")
print(df_results.head(10).to_string(index=False))

if not df_results.empty:
    best = df_results.iloc[0]
    print("\nBest Short Parameters:")
    print(best)

ib.disconnect()
