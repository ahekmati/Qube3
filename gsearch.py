from ib_insync import *
import pandas as pd
import numpy as np
import itertools

# --- CONFIG ---
symbol = 'TQQQ'
exchange = 'ARCA'
currency = 'USD'
period = '365 D'
candle_size = '4 hours'
backtest_start = '2024-01-01'
backtest_end = '2025-10-27'

z_long_range = np.arange(-0.5, -2.6, -0.1)   # Buy: z < threshold
z_short_range = np.arange(0.5, 2.6, 0.1)     # Sell: z > threshold
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

def compute_buy_the_dip(df, z_thresh, z_window, ema_fast, ema_slow, take_profit_pct, max_hold):
    df['ema_fast'] = df['close'].ewm(span=ema_fast, adjust=False).mean()
    df['ema_slow'] = df['close'].ewm(span=ema_slow, adjust=False).mean()
    uptrend = df['ema_fast'] > df['ema_slow']
    fast_ma = df['ema_fast']
    std_rolling = df['close'].rolling(z_window).std()
    dip_z = (df['close'] - fast_ma) / std_rolling
    dip_signal = dip_z < z_thresh
    buy_signals = uptrend & dip_signal
    df['buy_the_dip'] = buy_signals.fillna(False)
    df['signal'] = ''
    trades = []
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
                trades.append({'type':'long', 'entry_time':date, 'entry_price':entry_price,
                               'exit_time':df.index[j], 'exit_price':closej, 'exit_type':'TP'})
                sold = True
                break
            elif df['ema_fast'].iloc[j] < df['ema_slow'].iloc[j]:
                df.at[df.index[j], 'signal'] = 'SELL (EMA CROSS)'
                trades.append({'type':'long','entry_time':date, 'entry_price':entry_price,
                               'exit_time':df.index[j], 'exit_price':closej, 'exit_type':'EMA'})
                sold = True
                break
        if not sold and (idx+max_hold < len(df)):
            df.at[df.index[idx+max_hold], 'signal'] = 'SELL (TIME)'
            trades.append({'type':'long','entry_time':date, 'entry_price':entry_price,
                           'exit_time':df.index[idx+max_hold], 'exit_price':df['close'].iloc[idx+max_hold], 'exit_type':'TIME'})
    return trades

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
    trades = []
    entry_dates = df.index[df['sell_the_rally']]
    for date in entry_dates:
        entry_price = df.at[date, 'close']
        df.at[date, 'signal'] = 'SELL'
        idx = df.index.get_loc(date)
        covered = False
        for j in range(idx+1, min(idx+max_hold+1, len(df))):
            tp_price = entry_price * (1 - take_profit_pct)
            closej = df['close'].iloc[j]
            if closej <= tp_price:
                df.at[df.index[j], 'signal'] = 'COVER (TP)'
                trades.append({'type':'short','entry_time':date, 'entry_price':entry_price,
                               'exit_time':df.index[j], 'exit_price':closej, 'exit_type':'TP'})
                covered = True
                break
            elif df['ema_fast'].iloc[j] > df['ema_slow'].iloc[j]:
                df.at[df.index[j], 'signal'] = 'COVER (EMA CROSS)'
                trades.append({'type':'short','entry_time':date, 'entry_price':entry_price,
                               'exit_time':df.index[j], 'exit_price':closej, 'exit_type':'EMA'})
                covered = True
                break
        if not covered and (idx+max_hold < len(df)):
            df.at[df.index[idx+max_hold], 'signal'] = 'COVER (TIME)'
            trades.append({'type':'short','entry_time':date, 'entry_price':entry_price,
                           'exit_time':df.index[idx+max_hold], 'exit_price':df['close'].iloc[idx+max_hold], 'exit_type':'TIME'})
    return trades

data = get_data(symbol, candle_size, period)
data = data[(data.index >= backtest_start) & (data.index <= backtest_end)]

# --- LONG GRID SEARCH ---
long_results = []
for (z_thresh, z_window, ema_fast, ema_slow, tp, max_hold) in itertools.product(
        z_long_range, win_range, ema_fast_range, ema_slow_range, tp_range, max_hold_range):
    if ema_fast >= ema_slow:  # fast < slow for uptrend
        continue
    trades = compute_buy_the_dip(data.copy(), z_thresh, z_window, ema_fast, ema_slow, tp, max_hold)
    if len(trades) == 0:
        continue
    profits = [t['exit_price']-t['entry_price'] for t in trades]
    total_pnl = np.sum(profits)
    mean_pnl = np.mean(profits)
    win_rate = np.mean([p>0 for p in profits])
    long_results.append({
        'type':'long',
        'z':z_thresh,
        'window':z_window,
        'ema_fast':ema_fast,
        'ema_slow':ema_slow,
        'take_profit':tp,
        'max_hold':max_hold,
        'PnL':total_pnl,
        'Mean':mean_pnl,
        'WinRate':win_rate,
        'Trades':len(trades),
        'trade_details':trades
    })

# --- SHORT GRID SEARCH ---
short_results = []
for (z_thresh, z_window, ema_fast, ema_slow, tp, max_hold) in itertools.product(
        z_short_range, win_range, ema_fast_range, ema_slow_range, tp_range, max_hold_range):
    if ema_fast >= ema_slow:  # fast < slow for downtrend
        continue
    trades = compute_sell_the_rally(data.copy(), z_thresh, z_window, ema_fast, ema_slow, tp, max_hold)
    if len(trades) == 0:
        continue
    profits = [t['entry_price']-t['exit_price'] for t in trades]  # short trade profit
    total_pnl = np.sum(profits)
    mean_pnl = np.mean(profits)
    win_rate = np.mean([p>0 for p in profits])
    short_results.append({
        'type':'short',
        'z':z_thresh,
        'window':z_window,
        'ema_fast':ema_fast,
        'ema_slow':ema_slow,
        'take_profit':tp,
        'max_hold':max_hold,
        'PnL':total_pnl,
        'Mean':mean_pnl,
        'WinRate':win_rate,
        'Trades':len(trades),
        'trade_details':trades
    })

# --- RESULT AGGREGATION ---
all_results = pd.DataFrame(long_results + short_results)
all_results = all_results.sort_values('PnL', ascending=False)

# --- Print Top Results with Trade Dates ---
print("\nTop 5 Parameter Sets by Total PnL:")
for idx, row in all_results.head(5).iterrows():
    print(f"\n{'LONG' if row['type']=='long' else 'SHORT'}: z={row['z']}, window={row['window']}, ema_fast={row['ema_fast']}, ema_slow={row['ema_slow']}, tp={row['take_profit']}, max_hold={row['max_hold']}, PnL={row['PnL']:.2f}, Trades={row['Trades']}, WinRate={row['WinRate']:.2%}")
    for t in row['trade_details']:
        print(f"  Entry: {t['entry_time']} @ {t['entry_price']:.2f} | Exit ({t['exit_type']}): {t['exit_time']} @ {t['exit_price']:.2f} | Profit: {(t['exit_price']-t['entry_price']) if row['type']=='long' else (t['entry_price']-t['exit_price']):.2f}")

ib.disconnect()
