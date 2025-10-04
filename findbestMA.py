import os
import numpy as np
import pandas as pd
from ib_insync import *
import itertools
import pickle
import pprint

DATA_CACHE_FOLDER = "ma_cache"
os.makedirs(DATA_CACHE_FOLDER, exist_ok=True)

# --- Ticker List ---
TICKERS = [   "SSO", "SPXS", "TQQQ", "SQQQ", "SVXY",
    "SOXL", "IBIT", "TSLL", "NVDL", "SPY", "QQQ"
    
]

# --- MA Logic ---
def smma(series, period):
    """Smoothed moving average calculation."""
    s = pd.Series(series)
    if len(s) < period:
        return s.copy()
    result = s.copy()
    result.iloc[:period] = s.iloc[:period].mean()
    for i in range(period, len(result)):
        result.iloc[i] = (result.iloc[i-1]*(period-1) + s.iloc[i])/period
    return result

def backtest_ma(df, fast_period, slow_period):
    df = df.copy()
    df['fast'] = smma(df['close'], fast_period)
    df['slow'] = smma(df['close'], slow_period)
    signals = (df['fast'] > df['slow']).astype(int)
    signals = signals.diff().fillna(0)
    capital = 1.0
    position = 0
    entry = 0
    for i in range(1, len(signals)):
        if signals.iloc[i] == 1 and position == 0:
            position = 1
            entry = df['close'].iloc[i]
        elif signals.iloc[i] == -1 and position == 1:
            capital *= df['close'].iloc[i] / entry
            position = 0
    if position == 1:
        capital *= df['close'].iloc[-1] / entry
    return capital

def cached_fetch_ohlcv(ib, symbol, durationStr, barSizeSetting):
    cache_fn = os.path.join(DATA_CACHE_FOLDER, f"{symbol}_{barSizeSetting.replace(' ','_')}_{durationStr.replace(' ','_')}.pkl")
    if os.path.exists(cache_fn):
        return pd.read_pickle(cache_fn)
    contract = Stock(symbol, 'SMART', 'USD')
    bars = ib.reqHistoricalData(contract, endDateTime='', durationStr=durationStr, barSizeSetting=barSizeSetting,
                                whatToShow='TRADES', useRTH=True)
    df = util.df(bars)
    if not df.empty:
        df.to_pickle(cache_fn)
    return df

def analyze_timeframe(ib, symbol, durationStr, barSizeSetting, tf_name):
    df = cached_fetch_ohlcv(ib, symbol, durationStr, barSizeSetting)
    if df.empty or len(df) < 100:
        print(f"No data for {symbol} ({tf_name}).")
        return None
    best_result = -np.inf
    best_combo = None
    results = []
    for fast_period in range(5, 21, 2):
        for slow_period in range(18, 61, 5):
            if fast_period >= slow_period: continue
            total_return = backtest_ma(df, fast_period, slow_period)
            results.append((fast_period, slow_period, total_return))
            if total_return > best_result:
                best_result = total_return
                best_combo = (fast_period, slow_period)
    print(f"Best SMMA combo for {symbol} ({tf_name}): Fast={best_combo[0]}, Slow={best_combo[1]}, Capital={best_result:.2f}x")
    top5 = sorted(results, key=lambda x: -x[2])[:5]
    print("Top 5 combos:")
    for r in top5:
        print(f"  Fast {r[0]}, Slow {r[1]}: Return={r[2]:.2f}x")
    return best_combo, best_result

def main():
    ib = IB()
    ib.connect('127.0.0.1', 4001, clientId=1101)
    timeframes = [
        ('4 Y', '1 day', 'Daily'),
        ('2 Y', '8 hours', '8-Hour'),
        ('1 Y', '4 hours', '4-Hour'),
        ('6 M', '2 hours', '2-Hour'),
        ('3 M', '1 hour', '1-Hour'),
       
    ]
    for symbol in TICKERS:
        print(f"\n====== {symbol} ======")
        for durationStr, barSize, tf_name in timeframes:
            analyze_timeframe(ib, symbol, durationStr, barSize, tf_name)
    ib.disconnect()

    

if __name__ == "__main__":
    main()
