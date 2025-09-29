import numpy as np
import pandas as pd
import json
from datetime import datetime, timedelta
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

def analyze_crosses(df, fast, slow, ticker, today):
    df['smma_fast'] = smma(df['close'], fast)
    df['smma_slow'] = smma(df['close'], slow)
    prev_bull_high, prev_bull_date = None, None
    prev_bear_low, prev_bear_date = None, None
    cross_results = []
    new_cross_alerts = []
    for i in range(1, len(df)):
        # Bullish cross signal
        if df['smma_fast'].iloc[i] > df['smma_slow'].iloc[i] and df['smma_fast'].iloc[i-1] <= df['smma_slow'].iloc[i-1]:
            cross_date = df.index[i].date()
            if prev_bull_high is not None:
                msg = f"{ticker} BULL cross at {cross_date} -> Previous bull cross HIGH: {prev_bull_high:.2f} on {prev_bull_date.date()}"
                cross_results.append((cross_date, msg))
                # Alert if cross within last 10 days
                if (today - cross_date).days <= 30:
                    new_cross_alerts.append((cross_date, "ALERT: NEW CROSS -> " + msg))
            prev_bull_high = df['high'].iloc[i]
            prev_bull_date = df.index[i]
        # Bearish cross signal
        if df['smma_fast'].iloc[i] < df['smma_slow'].iloc[i] and df['smma_fast'].iloc[i-1] >= df['smma_slow'].iloc[i-1]:
            cross_date = df.index[i].date()
            if prev_bear_low is not None:
                msg = f"{ticker} BEAR cross at {cross_date} -> Previous bear cross LOW: {prev_bear_low:.2f} on {prev_bear_date.date()}"
                cross_results.append((cross_date, msg))
                if (today - cross_date).days <= 10:
                    new_cross_alerts.append((cross_date, "ALERT: NEW CROSS -> " + msg))
            prev_bear_low = df['low'].iloc[i]
            prev_bear_date = df.index[i]
    return cross_results, new_cross_alerts

def main():
    with open('tickers.json', 'r') as f:
        config = json.load(f)
    years = config.get('years', 1)
    fast = config.get('fast', 9)
    slow = config.get('slow', 18)
    tickers = config['tickers']

    today = datetime.now().date()
    ib = IB()
    ib.connect('127.0.0.1', 4001, clientId=1001)
    all_cross_results = []
    all_new_cross_alerts = []
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
        cross_results, new_cross_alerts = analyze_crosses(df, fast, slow, ticker, today)
        all_cross_results.extend(cross_results)
        all_new_cross_alerts.extend(new_cross_alerts)
    ib.disconnect()

    # Sort results by date (oldest to newest)
    all_cross_results.sort(key=lambda x: x[0])
    all_new_cross_alerts.sort(key=lambda x: x[0])

    print("\n=== All Crosses (Chronological) ===")
    if all_cross_results:
        for _, line in all_cross_results:
            print(line)
    else:
        print("No cross events found.")

    print("\n=== New Cross Alerts (Last 30 Days, Chronological) ===")
    if all_new_cross_alerts:
        for _, alert in all_new_cross_alerts:
            print(alert)
    else:
        print("No new crosses in last 30 days.")

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
    - Bears: Fast SMMA crosses below slow SMMA.
- Prints all historical cross events for all tickers in chronological order.
- Also alerts if any new cross occurred in the last 10 days, showing an "ALERT" for each, also sorted chronologically.
- If no new crosses occurred recently, displays a summary message.

This helps traders quickly identify actionable regime shifts and supports strike price decisions for option trades.
===============================================================================
"""
