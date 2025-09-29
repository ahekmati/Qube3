import numpy as np
import pandas as pd
import json
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
    return ib.reqHistoricalData(contract, '', f'{years} Y', '1 day', 'TRADES', useRTH=False)

def analyze_crosses(df, fast, slow, ticker):
    df['smma_fast'] = smma(df['close'], fast)
    df['smma_slow'] = smma(df['close'], slow)
    prev_bull_high, prev_bull_date = None, None
    prev_bear_low, prev_bear_date = None, None
    results = []
    for i in range(1, len(df)):
        # Bullish cross signal
        if df['smma_fast'].iloc[i] > df['smma_slow'].iloc[i] and df['smma_fast'].iloc[i-1] <= df['smma_slow'].iloc[i-1]:
            if prev_bull_high is not None:
                results.append(f"{ticker} BULL cross at {df.index[i].date()} -> Previous bull cross HIGH: {prev_bull_high:.2f} on {prev_bull_date.date()}")
            prev_bull_high = df['high'].iloc[i]
            prev_bull_date = df.index[i]
        # Bearish cross signal
        if df['smma_fast'].iloc[i] < df['smma_slow'].iloc[i] and df['smma_fast'].iloc[i-1] >= df['smma_slow'].iloc[i-1]:
            if prev_bear_low is not None:
                results.append(f"{ticker} BEAR cross at {df.index[i].date()} -> Previous bear cross LOW: {prev_bear_low:.2f} on {prev_bear_date.date()}")
            prev_bear_low = df['low'].iloc[i]
            prev_bear_date = df.index[i]
    return results

def main():
    # Load tickers from JSON file
    with open('tickers.json', 'r') as f:
        config = json.load(f)
    years = config.get('years', 1)
    fast = config.get('fast', 9)
    slow = config.get('slow', 36)
    tickers = config['tickers']

    ib = IB()
    ib.connect('127.0.0.1', 4001, clientId=1001)
    all_results = []
    for ticker in tickers:
        bars = fetch_bars(ib, ticker, years, 'ARCA') or \
               fetch_bars(ib, ticker, years, 'NASDAQ') or \
               fetch_bars(ib, ticker, years, 'SMART')
        if not bars:
            print(f"No valid data for {ticker}")
            continue
        df = util.df(bars)
        df['date'] = pd.to_datetime(df['date'])
        df = df.set_index('date')
        results = analyze_crosses(df, fast, slow, ticker)
        all_results.extend(results)
    ib.disconnect()
    # Print all results across tickers
    for line in all_results:
        print(line)

if __name__ == "__main__":
    main()

"""
===============================================================================
SCRIPT DESCRIPTION
===============================================================================
This script is an automated regime tracker and crossover signal notifier for 
multiple stock or ETF tickers. 

What it does:
- Reads a JSON file ('tickers.json') containing the list of tickers to monitor, 
  and optionally default parameters for lookback years and SMMA periods.
- Prompts Interactive Brokers (IB) for historical daily price data for each ticker.
- For each ticker, computes two smoothed moving averages (SMMA) on closing prices:
    - One 'fast' and one 'slow', using window sizes from JSON or defaults.
- Scans each ticker's history to identify regime crossovers:
    - Bulls: Fast SMMA crosses above slow SMMA.
        * At every bull cross, prints previous bull cross's high and its date.
    - Bears: Fast SMMA crosses below slow SMMA.
        * At every bear cross, prints previous bear cross's low and its date.
- Aggregates and prints all bull/bear crossover signals across all tickers.

How this helps:
- Allows options traders and technical analysts to systematically scan 
  multiple symbols for trend regime changes and previous extremes, 
  which are useful for choosing option strike prices or timing entries.
===============================================================================
"""
