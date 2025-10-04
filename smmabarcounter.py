from ib_insync import IB, Stock, util
import numpy as np
import pandas as pd

def calc_smma(prices, period):
    smma = [np.mean(prices[:period])]
    for i in range(period, len(prices)):
        prev = smma[-1]
        curr = ((prev * (period - 1)) + prices[i]) / period
        smma.append(curr)
    return np.array([np.nan] * (period - 1) + smma)

def prompt_params():
    ticker = input('Ticker: ').strip()
    timeframe = input('Timeframe (eg "1d", "1h", "5 mins"): ').strip()
    fast = int(input('SMMA Fast period: '))
    slow = int(input('SMMA Slow period: '))
    lookback_years = float(input('Lookback (years): '))
    return ticker, timeframe, fast, slow, lookback_years

def fetch_history(ib, ticker, timeframe, lookback_years):
    contract = Stock(ticker, 'SMART', 'USD')
    endDateTime = ''
    durationStr = f'{int(lookback_years)} Y' if lookback_years >= 1 else f'{int(lookback_years*365)} D'
    bars = ib.reqHistoricalData(
        contract,
        endDateTime,
        durationStr,
        timeframe,
        'TRADES',
        useRTH=True,
        formatDate=1
    )
    if not bars:
        return None
    df = util.df(bars)
    if df is None or df.empty:
        return None
    df['date'] = pd.to_datetime(df['date'])
    return df

def analyze_crossovers(df, fast, slow):
    closes = df['close'].values
    smma_fast = calc_smma(closes, fast)
    smma_slow = calc_smma(closes, slow)
    dates = df['date'].values

    cross_idxs = []
    cross_dates = []
    time_to_close_below = []
    below_slow_dates = []

    # For high-after-cross study
    cross_to_high_candles = []
    cross_to_high_days = []
    cross_high_dates = []
    cross_high_closes = []

    for i in range(1, len(closes)):
        if smma_fast[i-1] <= smma_slow[i-1] and smma_fast[i] > smma_slow[i]:
            cross_idxs.append(i)
            cross_dates.append(dates[i])
            idx_close_below = None
            for j in range(i+1, len(closes)):
                if closes[j] < smma_slow[j]:
                    time_to_close_below.append(j-i)
                    below_slow_dates.append(dates[j])
                    idx_close_below = j
                    break
            # Find high close between cross and next close below
            search_end = idx_close_below if idx_close_below else len(closes)
            if search_end > i:
                segment = closes[i:search_end]
                max_idx = np.argmax(segment)
                idx_high = i + max_idx
                cross_to_high_candles.append(idx_high - i)
                cross_to_high_days.append((df['date'].iloc[idx_high] - df['date'].iloc[i]).days)
                cross_high_dates.append(df['date'].iloc[idx_high])
                cross_high_closes.append(df['close'].iloc[idx_high])

    avg_candles1 = np.mean(time_to_close_below) if time_to_close_below else None

    print("\nSMMA fast crosses above slow at:")
    for d in cross_dates:
        print(pd.to_datetime(d).strftime('%Y-%m-%d %H:%M:%S'))
    print("\nFirst close below SMMA slow after crossover at:")
    for d in below_slow_dates:
        print(pd.to_datetime(d).strftime('%Y-%m-%d %H:%M:%S'))

    # Print high-after-cross study results
    print("\nHighest close after crossover (until next close below SMMA slow):")
    for d_cross, d_high, high_c, n_candles, n_days in zip(
        cross_dates, cross_high_dates, cross_high_closes, cross_to_high_candles, cross_to_high_days):
        print(f"Cross {pd.to_datetime(d_cross).strftime('%Y-%m-%d %H:%M:%S')}: "
              f"High {high_c:.2f} @ {pd.to_datetime(d_high).strftime('%Y-%m-%d %H:%M:%S')} "
              f"({n_candles} candles, {n_days} days)")
    if cross_to_high_candles:
        print(f"\nAverage candles to high after cross: {np.mean(cross_to_high_candles):.2f}")
        print(f"Average days to high after cross: {np.mean(cross_to_high_days):.2f}")

    # Second type: below then above then below again
    periods2 = []
    below_event_dates = []
    above_to_high_candles = []
    above_to_high_days = []
    above_high_dates = []
    above_high_closes = []

    i = 1
    while i < len(closes):
        if closes[i-1] >= smma_slow[i-1] and closes[i] < smma_slow[i]:
            for j in range(i+1, len(closes)):
                if closes[j] > smma_slow[j]:
                    idx_above = j
                    for k in range(j+1, len(closes)):
                        if closes[k] < smma_slow[k]:
                            periods2.append(k-i)
                            below_event_dates.append((dates[i], dates[j], dates[k]))
                            # High after crossing above, before next close below
                            segment = closes[j:k]
                            if len(segment) > 0:
                                max_idx = np.argmax(segment)
                                idx_high = j + max_idx
                                above_to_high_candles.append(idx_high - j)
                                above_to_high_days.append((df['date'].iloc[idx_high] - df['date'].iloc[j]).days)
                                above_high_dates.append(df['date'].iloc[idx_high])
                                above_high_closes.append(df['close'].iloc[idx_high])
                            i = k
                            break
                    break
        i += 1

    avg_candles2 = np.mean(periods2) if periods2 else None

    print("\nPrice crosses below, then above, then below SMMA slow at:")
    for d1, d2, d3 in below_event_dates:
        print(f"Below on {pd.to_datetime(d1).strftime('%Y-%m-%d %H:%M:%S')}, "
              f"above on {pd.to_datetime(d2).strftime('%Y-%m-%d %H:%M:%S')}, "
              f"below again on {pd.to_datetime(d3).strftime('%Y-%m-%d %H:%M:%S')}")

    print("\nHighest close after crossing above SMMA slow (until next close below):")
    for d_above, d_high, high_c, n_candles, n_days in zip(
        [x[1] for x in below_event_dates], above_high_dates, above_high_closes, above_to_high_candles, above_to_high_days):
        print(f"Above {pd.to_datetime(d_above).strftime('%Y-%m-%d %H:%M:%S')}: "
              f"High {high_c:.2f} @ {pd.to_datetime(d_high).strftime('%Y-%m-%d %H:%M:%S')} "
              f"({n_candles} candles, {n_days} days)")
    if above_to_high_candles:
        print(f"\nAverage candles to high after above-cross: {np.mean(above_to_high_candles):.2f}")
        print(f"Average days to high after above-cross: {np.mean(above_to_high_days):.2f}")

    return avg_candles1, avg_candles2, cross_dates, below_slow_dates, below_event_dates

def main():
    ib = IB()
    ib.connect('127.0.0.1', 4001, clientId=1)
    ticker, timeframe, fast, slow, lookback_years = prompt_params()
    df = fetch_history(ib, ticker, timeframe, lookback_years)
    if df is None or df.empty:
        print("No historical data returned. Check the ticker, timeframe, and lookback period.")
        ib.disconnect()
        return
    avg1, avg2, cross_dates, below_slow_dates, below_event_dates = analyze_crossovers(df, fast, slow)
    print(f"\n(1) Avg candles from bullish crossover to close below SMMA slow: {avg1}")
    print(f"(2) Avg candles from cross below SMMA slow back above and then below again: {avg2}")
    ib.disconnect()

if __name__ == '__main__':
    main()
