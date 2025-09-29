import numpy as np
import pandas as pd
from ta.trend import SMAIndicator, EMAIndicator, WMAIndicator
from ib_insync import *
import itertools

def smma(series, period):
    return series.ewm(alpha=1/period, adjust=False).mean()

def get_ma(series, period, ma_type):
    if ma_type == 'SMA':
        return SMAIndicator(series, window=period).sma_indicator()
    elif ma_type == 'EMA':
        return EMAIndicator(series, window=period).ema_indicator()
    elif ma_type == 'WMA':
        return WMAIndicator(series, window=period).wma()
    elif ma_type == 'SMMA':
        return smma(series, period)
    else:
        raise ValueError('Unknown MA type')

def backtest_ma(df, fast_type, slow_type, fast_period, slow_period):
    df = df.copy()
    df['fast'] = get_ma(df['close'], fast_period, fast_type)
    df['slow'] = get_ma(df['close'], slow_period, slow_type)
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

def run_for_timeframe(ib, contract, durationStr, barSizeSetting, timeframe_name):
    bars = ib.reqHistoricalData(contract, endDateTime='', durationStr=durationStr,
                                barSizeSetting=barSizeSetting, whatToShow='TRADES', useRTH=True)
    df = util.df(bars)
    if df.empty:
        print(f"No data found for {contract.symbol} in timeframe: {timeframe_name}.")
        return None
    ma_types = ['SMA', 'EMA', 'SMMA', 'WMA']
    best_result = -np.inf
    best_combo = None
    results = []

    for fast_type, slow_type in itertools.product(ma_types, repeat=2):
        for fast_period in range(5, 21, 2):
            for slow_period in range(18, 61, 5):
                if fast_period >= slow_period:
                    continue
                total_return = backtest_ma(df, fast_type, slow_type, fast_period, slow_period)
                results.append((fast_type, fast_period, slow_type, slow_period, total_return))
                if total_return > best_result:
                    best_result = total_return
                    best_combo = (fast_type, fast_period, slow_type, slow_period)

    print(f"\nBest MA combo for {contract.symbol} ({timeframe_name}): {best_combo} with final capital: {best_result:.2f}x")
    print(f"\nTop 5 combinations for {contract.symbol} ({timeframe_name}):")
    sorted_results = sorted(results, key=lambda x: -x[4])[:5]
    for row in sorted_results:
        print(f"Fast {row[0]}({row[1]}), Slow {row[2]}({row[3]}): {row[4]:.2f}x")

def main():
    ticker = input("Enter the stock ticker you want to find the best MA for: ").upper()
    ib = IB()
    ib.connect('127.0.0.1', '4001', clientId=111)
    contract = Stock(ticker, 'SMART', 'USD')

    timeframes = [
        ('4 Y', '1 day', 'Daily'),
        ('2 Y', '8 hours', '8-Hour'),
        ('1 Y', '4 hours', '4-Hour'),
        ('180 D', '1 hour', '1-Hour')
    ]

    for durationStr, barSizeSetting, tf_name in timeframes:
        run_for_timeframe(ib, contract, durationStr, barSizeSetting, tf_name)

    ib.disconnect()

if __name__ == "__main__":
    main()

