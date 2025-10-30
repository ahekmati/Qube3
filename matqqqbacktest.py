from ib_insync import *
import pandas as pd
import numpy as np

# --- CONFIG ---
symbol = 'QQQ'
exchange = 'ARCA'
currency = 'USD'
period = '60 D'                  # IB format (e.g., '60 D')
candle_sizes = ['4 hours', '8 hours']
ema_fast = 20                    # Fast EMA period
ema_slow = 50                    # Slow EMA period
z_window = 20
z_thresh = -0.5                    # Dip threshold (z-score)
take_profit_pct = 0.05           # 2% take profit
max_hold = 10                    # max bars to hold if no exit
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

def compute_buy_the_dip(df):
    # Smoothed regime filter: 20 EMA above 50 EMA
    df['ema_fast'] = df['close'].ewm(span=ema_fast, adjust=False).mean()
    df['ema_slow'] = df['close'].ewm(span=ema_slow, adjust=False).mean()
    uptrend = df['ema_fast'] > df['ema_slow']

    # Dip detection: z-score below fast EMA
    ema20 = df['ema_fast']
    std20 = df['close'].rolling(z_window).std()
    dip_z = (df['close'] - ema20) / std20
    dip_signal = dip_z < z_thresh

    # Aggregated buy signal (no RSI)
    buy_signals = uptrend & dip_signal
    df['buy_the_dip'] = buy_signals.fillna(False)

    # Signal logic
    df['signal'] = ''
    entry_dates = df.index[df['buy_the_dip']]
    for date in entry_dates:
        entry_price = df.at[date, 'close']
        df.at[date, 'signal'] = 'BUY'
        idx = df.index.get_loc(date)
        sold = False
        # Search forward for exit (TP, uptrend resume, max hold)
        for j in range(idx+1, min(idx+max_hold+1, len(df))):
            tp_price = entry_price * (1 + take_profit_pct)
            closej = df['close'].iloc[j]
            # Take profit
            if closej >= tp_price:
                df.at[df.index[j], 'signal'] = 'SELL (TP)'
                sold = True
                break
            # Exit on EMA cross (trend failed)
            elif df['ema_fast'].iloc[j] < df['ema_slow'].iloc[j]:
                df.at[df.index[j], 'signal'] = 'SELL (EMA CROSS)'
                sold = True
                break
        if not sold and (idx+max_hold < len(df)):
            df.at[df.index[idx+max_hold], 'signal'] = 'SELL (TIME)'
    return df

# --- Main loop with backtest & print ---
for size in candle_sizes:
    print(f"\n===== {size.upper()} bars =====")
    df = get_data(symbol, size, period)
    df = df[(df.index >= backtest_start) & (df.index <= backtest_end)]
    df = compute_buy_the_dip(df)
    signals = df[df['signal'].str.startswith('BUY') | df['signal'].str.startswith('SELL')][['signal', 'close', 'ema_fast', 'ema_slow']]
    for idx, row in signals.iterrows():
        print(f"{idx.strftime('%Y-%m-%d %H:%M')}: {row['signal']} @ {row['close']:.2f} (EMA20={row['ema_fast']:.2f}, EMA50={row['ema_slow']:.2f})")
    # PnL calculation
    buy_prices = signals[signals['signal'].str.startswith('BUY')]['close'].values
    sell_prices = signals[signals['signal'].str.startswith('SELL')]['close'].values
    n_trades = min(len(buy_prices), len(sell_prices))
    profits = sell_prices[:n_trades] - buy_prices[:n_trades]
    print(f"\nTrades: {n_trades} | Total PnL: {profits.sum():.2f} | Mean per trade: {profits.mean() if n_trades else 0:.2f}")

ib.disconnect()
