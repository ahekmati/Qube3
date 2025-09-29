import numpy as np
import pandas as pd
import json
from datetime import datetime
from ib_insync import *

def smma(series, window):
    s = pd.Series(series)
    out = s.copy()
    out.iloc[:window] = s.iloc[:window].mean()
    for i in range(window, len(out)):
        out.iloc[i] = (out.iloc[i-1]*(window-1)+s.iloc[i])/window
    return out

def fetch_bars(ib, ticker, duration_str, bar_size, exchange):
    contract = Stock(ticker, exchange, 'USD')
    bars = ib.reqHistoricalData(
        contract,
        endDateTime='',
        durationStr=duration_str,
        barSizeSetting=bar_size,
        whatToShow='TRADES',
        useRTH=False
    )
    return bars

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
            cross_date = df.index[i]
            if prev_bull_high is not None:
                msg = f"{ticker} BULL cross at {cross_date.date()} -> Previous bull cross HIGH: {prev_bull_high:.2f} on {prev_bull_date.date()}"
                cross_results.append((cross_date, msg))
                if (today - cross_date.date()).days <= 30:
                    new_cross_alerts.append((cross_date, "ALERT: NEW CROSS -> " + msg))
            prev_bull_high = df['high'].iloc[i]
            prev_bull_date = df.index[i]
        # Bearish cross signal
        if df['smma_fast'].iloc[i] < df['smma_slow'].iloc[i] and df['smma_fast'].iloc[i-1] >= df['smma_slow'].iloc[i-1]:
            cross_date = df.index[i]
            if prev_bear_low is not None:
                msg = f"{ticker} BEAR cross at {cross_date.date()} -> Previous bear cross LOW: {prev_bear_low:.2f} on {prev_bear_date.date()}"
                cross_results.append((cross_date, msg))
                if (today - cross_date.date()).days <= 30:
                    new_cross_alerts.append((cross_date, "ALERT: NEW CROSS -> " + msg))
            prev_bear_low = df['low'].iloc[i]
            prev_bear_date = df.index[i]
    return cross_results, new_cross_alerts

def main():
    with open('tickers.json', 'r') as f:
        config = json.load(f)
    tickers = config['tickers']
    years = config.get('years', 1)
    duration_str = f"{years} Y"
    today = datetime.now().date()
    ib = IB()
    ib.connect('127.0.0.1', 4001, clientId=1001)
    exchanges = ['ARCA', 'NASDAQ', 'SMART']

    tickers_with_alerts = set()

    for ticker in tickers:
        print(f"\n==== {ticker} ====")
        alert_found = False

        # Daily timeframe: 9/18 SMMA
        bars_daily = None
        for exch in exchanges:
            bars_daily = fetch_bars(ib, ticker, duration_str, "1 day", exch)
            if bars_daily:
                break
        if bars_daily:
            df_daily = util.df(bars_daily)
            df_daily['date'] = pd.to_datetime(df_daily['date'])
            df_daily = df_daily.set_index('date')
            cross_results_daily, new_cross_alerts_daily = analyze_crosses(df_daily, 9, 18, ticker, today)
            print("\nDaily 9/18 SMMA Crosses:")
            if cross_results_daily:
                for _, line in cross_results_daily:
                    print(line)
            else:
                print("No daily cross events found.")
            print("New cross alerts (Last 30 Days):")
            if new_cross_alerts_daily:
                alert_found = True
                for _, alert in new_cross_alerts_daily:
                    print(alert)
            else:
                print("No new daily crosses in last 30 days.")
        else:
            print("No valid daily data.")

        # 4-hour timeframe: 26/150 SMMA
        bars_4h = None
        for exch in exchanges:
            bars_4h = fetch_bars(ib, ticker, duration_str, "4 hours", exch)
            if bars_4h:
                break
        if bars_4h:
            df_4h = util.df(bars_4h)
            df_4h['date'] = pd.to_datetime(df_4h['date'])
            df_4h = df_4h.set_index('date')
            cross_results_4h, new_cross_alerts_4h = analyze_crosses(df_4h, 26, 150, ticker, today)
            print("\n4-Hour 26/150 SMMA Crosses:")
            if cross_results_4h:
                for _, line in cross_results_4h:
                    print(line)
            else:
                print("No 4-hour cross events found.")
            print("New cross alerts (Last 30 Days):")
            if new_cross_alerts_4h:
                alert_found = True
                for _, alert in new_cross_alerts_4h:
                    print(alert)
            else:
                print("No new 4-hour crosses in last 30 days.")
        else:
            print("No valid 4-hour data.")

        if alert_found:
            tickers_with_alerts.add(ticker)

    ib.disconnect()

    print("\n==== SUMMARY OF TICKERS WITH NEW ALERTS (LAST 30 DAYS) ====")
    if tickers_with_alerts:
        print(", ".join(sorted(tickers_with_alerts)))
    else:
        print("No alerts found for any ticker in the last 30 days.")

if __name__ == "__main__":
    main()



## checks tickers on the 4 hour and daily for the 9-18 and 26/150 smma and gives the previous high points