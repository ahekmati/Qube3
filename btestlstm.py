import yfinance as yf
import pandas as pd
import numpy as np

symbol = 'TQQQ'
timeframes = {
    '1d': {'interval': '1d', 'period': '1y'},
    '4h': {'interval': '4h', 'period': '60d'},
    '1h': {'interval': '1h', 'period': '60d'},
}
ema_fast = 20
ema_slow = 50
z_window = 20
z_thresh_long = -0.5
take_profit_pct = .12
stop_loss_pct = .04
max_hold = 50

def get_data_yf(symbol, tf, config):
    ticker = yf.Ticker(symbol)
    df = ticker.history(interval=config['interval'], period=config['period'])
    if df.empty or 'Close' not in df:
        return pd.DataFrame()
    df = df.rename_axis('date').reset_index()
    df['close'] = df['Close']
    df = df[['date', 'close']].set_index('date')
    if tf == '8h':
        # Start with 1h data to allow resampling to 8h later
        ticker = yf.Ticker(symbol)
        base = ticker.history(interval='1h', period='60d')
        if base.empty or 'Close' not in base:
            return pd.DataFrame()
        base = base.rename_axis('date').reset_index()
        base['close'] = base['Close']
        base = base[['date', 'close']].set_index('date')
        df = base.resample('8H').last().dropna()
    return df

def compute_signals(df):
    df['ema_fast'] = df['close'].ewm(span=ema_fast, adjust=False).mean()
    df['ema_slow'] = df['close'].ewm(span=ema_slow, adjust=False).mean()
    std20 = df['close'].rolling(z_window).std()
    dip_z = (df['close'] - df['ema_fast']) / std20
    buy_signals = (df['close'] < df['ema_slow']) & (dip_z < z_thresh_long)
    df['buy_signal'] = buy_signals.fillna(False)
    return df

def backtest(df):
    trades = []
    position = 0
    entry = None
    entry_idx = None
    for i, (idx, row) in enumerate(df.iterrows()):
        if not position and row['buy_signal']:
            entry = row['close']
            entry_idx = i
            position = 1
        elif position:
            tp = entry * (1 + take_profit_pct)
            sl = entry * (1 - stop_loss_pct)
            if row['close'] >= tp:
                trades.append({'entry': entry, 'exit': tp, 'pnl': tp-entry, 'win': 1, 'bars': i-entry_idx})
                position = 0
            elif row['close'] <= sl:
                trades.append({'entry': entry, 'exit': sl, 'pnl': sl-entry, 'win': 0, 'bars': i-entry_idx})
                position = 0
            elif i - entry_idx >= max_hold:
                trades.append({'entry': entry, 'exit': row['close'], 'pnl': row['close']-entry, 'win': int(row['close']>entry), 'bars': i-entry_idx})
                position = 0
    return trades

results = []
for tf in ['1d', '8h', '4h', '1h']:
    print(f"\nBacktesting {symbol} [{tf}] timeframe ...")
    config = timeframes.get(tf, timeframes['1h'])
    data = get_data_yf(symbol, tf, config)
    if data.empty:
        print("No data for", tf)
        continue
    data = compute_signals(data)
    trades = backtest(data)
    if not trades:
        print("No trades found.")
        continue
    trade_df = pd.DataFrame(trades)
    total_pnl = trade_df['pnl'].sum()
    ntrades = len(trade_df)
    winrate = (trade_df['win'].sum() / ntrades)*100
    max_dd = (trade_df['exit'].cummax() - trade_df['exit']).max()
    avg_pnl = trade_df['pnl'].mean()
    print(f"Total PnL: {total_pnl:.2f}, #Trades: {ntrades}, Winrate: {winrate:.1f}%, MaxDrawdown: {max_dd:.2f}, AvgPnL: {avg_pnl:.2f}")
    results.append({'timeframe': tf, 'trades': ntrades, 'total_pnl': total_pnl, 'winrate': winrate, 'max_dd': max_dd, 'avg_pnl': avg_pnl})

if results:
    summary = pd.DataFrame(results)
    print("\n==== Timeframe Comparison ====")
    print(summary.sort_values('total_pnl', ascending=False).to_string(index=False))
else:
    print("\nNo data available for any timeframe. Adjust symbol, period, or intervals.")
