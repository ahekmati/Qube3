from ib_insync import *
import pandas as pd
import numpy as np

# --- CONFIG ---
tickers = ['QQQ', 'SPY']
exchange = 'ARCA'
currency = 'USD'
period = '60 D'
candle_sizes = ['4 hours', '8 hours']
ema_fast = 20
ema_slow = 50
z_window = 20
z_thresh = -0.5
take_profit_pct = .12
max_hold = 50
backtest_start = '2025-03-01'
backtest_end = '2025-10-29'

# --- Connect to IB ---
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

def compute_buy_the_dip_time_tp_only(df):
    df['ema_fast'] = df['close'].ewm(span=ema_fast, adjust=False).mean()
    df['ema_slow'] = df['close'].ewm(span=ema_slow, adjust=False).mean()
    std20 = df['close'].rolling(z_window).std()
    dip_z = (df['close'] - df['ema_fast']) / std20
    dip_signal = dip_z < z_thresh
    buy_signals = (df['close'] < df['ema_slow']) & dip_signal
    df['buy_the_dip'] = buy_signals.fillna(False)
    df['signal'] = ''
    in_position = False
    i = 0
    while i < len(df):
        idx = df.index[i]
        if not in_position and df.at[idx, 'buy_the_dip']:
            entry_price = df.at[idx, 'close']
            df.at[idx, 'signal'] = 'BUY'
            in_position = True
            exit_found = False
            for j in range(i+1, min(i+max_hold+1, len(df))):
                tp_price = entry_price * (1 + take_profit_pct)
                closej = df['close'].iloc[j]
                idxj = df.index[j]
                if closej >= tp_price:
                    df.at[idxj, 'signal'] = 'SELL (TP)'
                    i = j + 1
                    in_position = False
                    exit_found = True
                    break
            if not exit_found and (i+max_hold < len(df)):
                idxj = df.index[i+max_hold]
                df.at[idxj, 'signal'] = 'SELL (TIME)'
                i = i + max_hold + 1
                in_position = False
            elif not exit_found:
                i += 1
        elif in_position:
            i += 1  # skip bars during position
        else:
            i += 1
    return df

# --- Main loop: run for each ticker, each timeframe ---
for symbol in tickers:
    for size in candle_sizes:
        print(f"\n===== {symbol}: {size.upper()} bars =====")
        df = get_data(symbol, size, period)
        df = df[(df.index >= backtest_start) & (df.index <= backtest_end)]
        df = compute_buy_the_dip_time_tp_only(df)
        signals = df[df['signal'].str.startswith('BUY') | df['signal'].str.startswith('SELL')][['signal', 'close', 'ema_fast', 'ema_slow']]
        for idx, row in signals.iterrows():
            print(f"{idx.strftime('%Y-%m-%d %H:%M')}: {row['signal']} @ {row['close']:.2f} (EMA{ema_fast}={row['ema_fast']:.2f}, EMA{ema_slow}={row['ema_slow']:.2f})")
        # PnL calculation
        buy_prices = signals[signals['signal'].str.startswith('BUY')]['close'].values
        sell_prices = signals[signals['signal'].str.startswith('SELL')]['close'].values
        n_trades = min(len(buy_prices), len(sell_prices))
        profits = sell_prices[:n_trades] - buy_prices[:n_trades]
        print(f"\nTrades: {n_trades} | Total PnL: {profits.sum():.2f} | Mean per trade: {profits.mean() if n_trades else 0:.2f}")

ib.disconnect()
