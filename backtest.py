import numpy as np
import pandas as pd
from ib_insync import *

# Computes Smoothed Moving Average (SMMA) for a pandas Series
def smma(series, window):
    s = pd.Series(series)
    out = s.copy()
    # Initialize first 'window' values with the mean
    out.iloc[:window] = s.iloc[:window].mean()
    for i in range(window, len(out)):
        # Recursive SMMA calculation
        out.iloc[i] = (out.iloc[i-1]*(window-1)+s.iloc[i])/window
    return out

# Fetch historical daily bars for a ticker from a specific exchange
def fetch_bars(ib, ticker, years, exchange):
    contract = Stock(ticker, exchange, 'USD')
    # Request historical data from IB
    bars = ib.reqHistoricalData(contract, '', f'{years} Y', '1 day', 'TRADES', useRTH=False)
    print(f"\nTrying exchange: {exchange}")
    print("API bars returned:", len(bars), "records")
    if bars and hasattr(bars[0], 'date'):
        for sample in bars[:3]:
            print("Sample bar:", sample)
    return bars

def main():
    # Prompt for user input: ticker, lookback period, and SMMA window sizes
    ticker = input("Enter ticker symbol: ").upper().strip()
    years = input("Enter number of years to look back: ").strip()
    years = int(years) if years.isdigit() else 1
    fast = input("Enter fast SMMA period: ").strip()
    slow = input("Enter slow SMMA period: ").strip()
    fast = int(fast) if fast.isdigit() else 9
    slow = int(slow) if slow.isdigit() else 36

    # Connect to Interactive Brokers TWS/Gateway
    ib = IB()
    ib.connect('127.0.0.1', 4001, clientId=1001)
    # Try ARCA first, then NASDAQ, then SMART
    bars = fetch_bars(ib, ticker, years, 'ARCA')
    if not bars:
        bars = fetch_bars(ib, ticker, years, 'NASDAQ')
    if not bars:
        bars = fetch_bars(ib, ticker, years, 'SMART')
    ib.disconnect()

    # Early exit if no data found
    if not bars:
        print(f"No valid historical data returned for {ticker} on any major exchange. Try another ticker, or check API permissions and market data subscriptions.")
        return

    # Convert IB bar data to pandas DataFrame
    df = util.df(bars)
    df['date'] = pd.to_datetime(df['date'])
    df = df.set_index('date')
    print(df.head(), "\nDataFrame index type:", type(df.index[0]) if len(df) > 0 else "EMPTY")

    # Validate valid DataFrame with date index
    if df.empty or not isinstance(df.index[0], pd.Timestamp):
        print(f"No valid historical data after DataFrame conversion for {ticker}.")
        return

    # Compute fast and slow SMMA on closing prices
    df['smma_fast'] = smma(df['close'], fast)
    df['smma_slow'] = smma(df['close'], slow)

    # Identify and print bullish regimes (fast SMMA > slow SMMA)
    print(f"\nBullish Regimes (SMMA {fast} > SMMA {slow}):")
    print("Start Date | End Date   | Highest Close | High Date")
    in_bull = False
    for i in range(len(df)):
        if df['smma_fast'].iloc[i] > df['smma_slow'].iloc[i]:
            if not in_bull:
                in_bull = True
                bull_start = df.index[i]
                bull_high = df['close'].iloc[i]
                bull_high_date = df.index[i]
            elif df['close'].iloc[i] > bull_high:
                bull_high = df['close'].iloc[i]
                bull_high_date = df.index[i]
        else:
            if in_bull:
                bull_end = df.index[i]
                print(f"{bull_start.date()} | {bull_end.date()} | {bull_high:.2f}     | {bull_high_date.date()}")
                in_bull = False

    # Identify and print bearish regimes (fast SMMA < slow SMMA)
    print(f"\nBearish Regimes (SMMA {fast} < SMMA {slow}):")
    print("Start Date | End Date   | Lowest Close | Low Date")
    in_bear = False
    for i in range(len(df)):
        if df['smma_fast'].iloc[i] < df['smma_slow'].iloc[i]:
            if not in_bear:
                in_bear = True
                bear_start = df.index[i]
                bear_low = df['close'].iloc[i]
                bear_low_date = df.index[i]
            elif df['close'].iloc[i] < bear_low:
                bear_low = df['close'].iloc[i]
                bear_low_date = df.index[i]
        else:
            if in_bear:
                bear_end = df.index[i]
                print(f"{bear_start.date()} | {bear_end.date()} | {bear_low:.2f}     | {bear_low_date.date()}")
                in_bear = False

if __name__ == "__main__":
    main()

'''
==============================
SCRIPT SUMMARY AND DESCRIPTION
==============================
This script performs the following actions:

- Prompts the user for a ticker symbol, number of years to look back, and fast/slow moving average windows (SMMA).
- Connects to Interactive Brokers (IB) and fetches historical daily price data for the ticker.
- Calculates two smoothed moving averages: SMMA(fast) and SMMA(slow) on the closing price series.
- Identifies all bullish regimes (where SMMA(fast) > SMMA(slow)), for each prints:
    * The regime's start and end date
    * The highest close within the regime and the date it occurred.
- Identifies all bearish regimes (where SMMA(fast) < SMMA(slow)), for each prints:
    * The regime's start and end date
    * The lowest close within the regime and the date it occurred.
- Outputs each regime's summary in the terminal. Useful for visualizing cycle extremes and confirming trend structure.
- Script is meant for technical traders using Interactive Brokers and daily data.

You can use this output for regime-based trading strategy design or for option strike selection based on identified regime highs/lows.
'''
