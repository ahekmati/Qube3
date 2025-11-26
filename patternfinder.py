from ib_insync import IB, ContFuture
import pandas as pd
import ta
from datetime import datetime

# IBKR connection details
IB_HOST = '127.0.0.1'
IB_PORT = 4001
IB_CLIENT_ID = 13

def fetch_nq_data(bar_size, duration):
    ib = IB()
    ib.connect(IB_HOST, IB_PORT, clientId=IB_CLIENT_ID)
    
    # Use the "continuous" NQ contract to always get the current front month
    contract = ContFuture('QQQ', exchange='SMART')
    
    bars = ib.reqHistoricalData(
        contract,
        endDateTime='',
        durationStr=duration,
        barSizeSetting=bar_size,
        whatToShow='TRADES',
        useRTH=True,
        formatDate=1
    )
    ib.disconnect()
    if not bars:
        print(f"Error: No data returned for {bar_size}")
        return pd.DataFrame()
    data = pd.DataFrame([[b.date, b.open, b.high, b.low, b.close, b.volume] for b in bars],
                        columns=['date', 'open', 'high', 'low', 'close', 'volume'])
    data['date'] = pd.to_datetime(data['date'])
    return data

def detect_bullish_reversal(df):
    df['ema9'] = ta.trend.ema_indicator(df['close'], window=9)
    df['ema12'] = ta.trend.ema_indicator(df['close'], window=12)
    signals = []
    for i in range(2, len(df)):
        bullish_candle = df['close'][i] > df['open'][i]
        crossover = df['ema9'][i-1] < df['ema12'][i-1] and df['ema9'][i] > df['ema12'][i]
        prev_bearish = df['close'][i-1] < df['open'][i-1]
        if crossover and bullish_candle and prev_bearish:
            signals.append((df['date'][i], df['close'][i]))
    return signals

if __name__ == "__main__":
    bar_sizes = ['1 hour', '4 hours', '1 day']
    duration = '2 Y'
    for bar_size in bar_sizes:
        print(f"\n--- {bar_size.upper()} (NQ FUT CONTINUOUS) ---")
        df = fetch_nq_data(bar_size=bar_size, duration=duration)
        if df.empty:
            continue
        signals = detect_bullish_reversal(df)
        for ts, price in signals:
            print(f"Pattern found: {ts} at price {price}")
