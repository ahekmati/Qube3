import numpy as np
import pandas as pd
from ib_insync import *

def smma(series, window):
    s = pd.Series(series)
    out = s.copy()
    out.iloc[:window] = s.iloc[:window].mean()
    for i in range(window, len(out)):
        out.iloc[i] = (out.iloc[i-1]*(window-1)+s.iloc[i])/window
    return out

def fetch_bars(ib, ticker, years, exchange):
    contract = Stock(ticker, exchange, 'USD')
    bars = ib.reqHistoricalData(contract, '', f'{years} Y', '1 day', 'TRADES', useRTH=False)
    print(f"\nTrying exchange: {exchange}")
    print("API bars returned:", len(bars), "records")
    if bars and hasattr(bars[0], 'date'):
        for sample in bars[:3]:
            print("Sample bar:", sample)
    return bars

def main():
    ticker = input("Enter ticker symbol: ").upper().strip()
    years = input("Enter number of years to look back: ").strip()
    years = int(years) if years.isdigit() else 1
    fast = input("Enter fast SMMA period: ").strip()
    slow = input("Enter slow SMMA period: ").strip()
    fast = int(fast) if fast.isdigit() else 9
    slow = int(slow) if slow.isdigit() else 36

    ib = IB()
    ib.connect('127.0.0.1', 4001, clientId=1001)
    bars = fetch_bars(ib, ticker, years, 'ARCA')
    if not bars:
        bars = fetch_bars(ib, ticker, years, 'NASDAQ')
    if not bars:
        bars = fetch_bars(ib, ticker, years, 'SMART')
    ib.disconnect()

    if not bars:
        print(f"No valid historical data returned for {ticker} on any major exchange. Try another ticker, or check API permissions and market data subscriptions.")
        return
    df = util.df(bars)
    df['date'] = pd.to_datetime(df['date'])
    df = df.set_index('date')
    print(df.head(), "\nDataFrame index type:", type(df.index[0]) if len(df) > 0 else "EMPTY")
    if df.empty or not isinstance(df.index[0], pd.Timestamp):
        print(f"No valid historical data after DataFrame conversion for {ticker}.")
        return

    df['smma_fast'] = smma(df['close'], fast)
    df['smma_slow'] = smma(df['close'], slow)

    prev_bull_high = None
    prev_bull_date = None
    prev_bear_low = None
    prev_bear_date = None
    looking_for_bull_cross = True
    looking_for_bear_cross = True

    for i in range(1, len(df)):
        # Bullish cross
        if df['smma_fast'].iloc[i] > df['smma_slow'].iloc[i] and df['smma_fast'].iloc[i-1] <= df['smma_slow'].iloc[i-1]:
            # On next bull cross, print prior bull high
            if prev_bull_high is not None:
                print(f"BULL cross at {df.index[i].date()} -> Previous bull cross HIGH: {prev_bull_high:.2f} on {prev_bull_date.date()}")
            prev_bull_high = df['high'].iloc[i]
            prev_bull_date = df.index[i]

        # Bearish cross
        if df['smma_fast'].iloc[i] < df['smma_slow'].iloc[i] and df['smma_fast'].iloc[i-1] >= df['smma_slow'].iloc[i-1]:
            # On next bear cross, print prior bear low
            if prev_bear_low is not None:
                print(f"BEAR cross at {df.index[i].date()} -> Previous bear cross LOW: {prev_bear_low:.2f} on {prev_bear_date.date()}")
            prev_bear_low = df['low'].iloc[i]
            prev_bear_date = df.index[i]

if __name__ == "__main__":
    main()

# --------------------------------------------------------------------
# SCRIPT DESCRIPTION:
#
# This script connects to Interactive Brokers (IB) and downloads historical 
# daily price data for a user-specified stock ticker. It then computes two 
# smoothed moving averages (SMMA) on the closing prices, using user-defined 
# fast and slow periods.
# 
# As the script analyzes the historical price series, it detects crossover 
# events:
#   - When the fast SMMA crosses above the slow SMMA (bullish cross), it 
#     prints the high from the previous bullish cross.
#   - When the fast SMMA crosses below the slow SMMA (bearish cross), it 
#     prints the low from the previous bearish cross.
#
# Each time a crossover is detected, the previous regime's high or low is 
# output in the terminal, making it easy to identify levels to use for 
# entry, exit, or option strike selection in trend-following regimes.
#
# This approach is intended for traders and analysts who want to
# automate regime detection and quickly visualize cycle highs and 
# lows for further strategy development or historical testing.
# --------------------------------------------------------------------
