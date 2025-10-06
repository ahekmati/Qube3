from binance.client import Client
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# ---- Binance Auth -----
api_key = 'tbk4AJ9Lk6uNSFIUJuEOgd8e2UXB2r1IfId0OQi9GlG1hPCM1nfDNRdjWsej9psB'
api_secret = 'eHVdI6rlsIn33MCwXH33jX8G0Xp27a8cMESVZm5oslpSCnB4tw5E6ympRpbFjxeg'
client = Client(api_key, api_secret)

# ---- MA Functions ----
def sma(series, window):
    return pd.Series(series).rolling(window=window).mean()

def ema(series, window):
    return pd.Series(series).ewm(span=window, adjust=False).mean()

def wma(series, window):
    weights = np.arange(1, window + 1)
    return pd.Series(series).rolling(window).apply(
        lambda x: np.dot(x, weights)/weights.sum(), raw=True)

def smma(series, window):
    s = pd.Series(series)
    if len(s) < window: return s.copy()
    result = s.copy()
    result.iloc[:window] = s.iloc[:window].mean()
    for i in range(window, len(result)):
        result.iloc[i] = (result.iloc[i-1]*(window-1) + s.iloc[i])/window
    return result

MA_FUNCTIONS = {'SMA': sma, 'EMA': ema, 'WMA': wma, 'SMMA': smma}

# ---- Backtest Logic -----
def backtest_ma_strategy(df, fast, slow, ma_type):
    fast_ma = MA_FUNCTIONS[ma_type](df['close'], fast)
    slow_ma = MA_FUNCTIONS[ma_type](df['close'], slow)
    df = df.copy()
    df['fast_ma'] = fast_ma
    df['slow_ma'] = slow_ma

    df['position'] = 0
    # Crossover logic
    up_cross = (df['fast_ma'].shift(1) <= df['slow_ma'].shift(1)) & (df['fast_ma'] > df['slow_ma'])
    down_cross = (df['fast_ma'].shift(1) >= df['slow_ma'].shift(1)) & (df['fast_ma'] < df['slow_ma'])
    df.loc[up_cross, 'position'] = 1
    df.loc[down_cross, 'position'] = 0
    df['position'] = df['position'].replace(0, np.nan).ffill().fillna(0)

    df['returns'] = df['close'].pct_change() * df['position']
    total_return = (df['returns'] + 1).prod() - 1
    sharpe = df['returns'].mean() / df['returns'].std() * np.sqrt(252) if df['returns'].std() != 0 else 0
    trades = df['position'].diff().abs().sum() / 2
    up_cross_count = up_cross.sum()
    return total_return, sharpe, trades, up_cross_count

# ---- Binance Data Fetch -----
def fetch_historical_data_binance(symbol, interval, days):
    klines = client.get_historical_klines(symbol, interval, f"{days} day ago UTC")
    df = pd.DataFrame(klines, columns=[
        'timestamp','open','high','low','close','volume','close_time','quote_asset_volume',
        'num_trades','taker_buy_base','taker_buy_quote','ignore'
    ])
    df['close'] = df['close'].astype(float)
    df['open_time_dt'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.set_index('open_time_dt', inplace=True)
    return df

# ---- Best Combo Search ------
def find_best_ma_combo_binance(symbol, candidate_types, candidate_pairs, duration_days=720):
    timeframes = {'1d':'daily', '8h':'8H', '6h':'6H', '4h':'4H', '1h':'hourly'}
    results = {}
    for interval, label in timeframes.items():
        df = fetch_historical_data_binance(symbol, interval, duration_days)
        if df.empty or len(df) < 50:
            continue
        best_score = -np.inf
        best_combo = None
        for ma_type in candidate_types:
            for fast, slow in candidate_pairs:
                if fast >= slow: continue
                total_ret, sharpe, trades, up_cross_count = backtest_ma_strategy(df, fast, slow, ma_type)
                score = sharpe + 0.01 * trades
                if score > best_score:
                    best_score = score
                    best_combo = (ma_type, fast, slow, sharpe, total_ret, trades, up_cross_count)
        if best_combo:
            results[label] = {
                'type': best_combo[0],
                'fast': best_combo[1],
                'slow': best_combo[2],
                'sharpe': best_combo[3],
                'total_return': best_combo[4],
                'trades': best_combo[5],
                'up_crosses': best_combo[6]
            }
    return results

# ---------- Main -------------
def main():
    symbol = "ETHUSDT"
    candidate_types = ['SMA', 'EMA', 'WMA', 'SMMA']
    candidate_pairs = [(9,18), (13,31), (21,50), (26,150), (10,20), (5,14)]
    print(f"\nAnalyzing {symbol} ...")
    combos = find_best_ma_combo_binance(symbol, candidate_types, candidate_pairs)
    if combos:
        print(f"{symbol} BEST COMBOS:")
        for tf, info in combos.items():
            print(
                f"  {tf.upper()}: {info['type']} {info['fast']}/{info['slow']} | "
                f"Sharpe: {info['sharpe']:.3f} | Return: {info['total_return']:.2%} | "
                f"Trades: {info['trades']} | Up Crosses: {info['up_crosses']}"
            )
    else:
        print(f"{symbol}: No sufficient data or signals.")

if __name__ == "__main__":
    main()
