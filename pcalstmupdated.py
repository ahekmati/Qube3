from ib_insync import *
import pandas as pd
import numpy as np

IB_HOST = '127.0.0.1'
IB_PORT = 4001
IB_CLIENT_ID = 9

symbols = ['QQQ', 'SPY']
exchange = 'SMART'
currency = 'USD'

# From your summary:
EXPECTED = {
    ('QQQ', '1D'): {'avg_dip': -2.927018, 'avg_reb': 2.468717},
    ('QQQ', '4H'): {'avg_dip': -2.152439, 'avg_reb': 1.299490},
    ('SPY', '1D'): {'avg_dip': -2.489006, 'avg_reb': 1.993848},
    ('SPY', '4H'): {'avg_dip': -1.952473, 'avg_reb': 1.108578},
}

ib = IB()
ib.connect(IB_HOST, IB_PORT, IB_CLIENT_ID)

def to_df(bars):
    df = util.df(bars)
    if df.empty:
        return df
    df.rename(columns=str.lower, inplace=True)
    df.set_index('date', inplace=True)
    df.index = pd.to_datetime(df.index)
    df = df.sort_index()
    return df

def fetch_ib_history(symbol, bar_size, duration):
    contract = Stock(symbol, exchange, currency)
    bars = ib.reqHistoricalData(
        contract,
        endDateTime='',
        durationStr=duration,
        barSizeSetting=bar_size,
        whatToShow='TRADES',
        useRTH=False,
        formatDate=1
    )
    if not bars:
        return pd.DataFrame()
    return to_df(bars)

def detect_latest_dip(df, min_dip_pct=-1.0):
    """
    Detects if the latest completed bar is a turn-up local low:
    - local low: close[i] < close[i-1] and close[i] < close[i+1]
    - turns up: close[i+1] > close[i]
    - there exists a prior local high before i
    - dip depth from that high <= min_dip_pct (negative threshold, e.g., -1.0)
    Returns dict with info or None.
    """
    if len(df) < 5:
        return None
    closes = df['close'].values
    dates = df.index.to_list()

    # Work on all but the last bar as "completed" if you want conservative; here we treat last as completed.
    i = len(closes) - 2  # second to last, so i+1 exists
    # Move backwards to find the most recent local low that turns up
    while i > 1:
        # local low & turn up
        if closes[i] < closes[i-1] and closes[i] < closes[i+1] and closes[i+1] > closes[i]:
            low_idx = i
            # search backwards for last local high before low
            last_high_idx = None
            j = low_idx - 1
            while j > 0:
                if closes[j] > closes[j-1] and closes[j] > closes[j+1]:
                    last_high_idx = j
                    break
                j -= 1
            if last_high_idx is None:
                return None
            high_price = closes[last_high_idx]
            low_price = closes[low_idx]
            dip_pct = (low_price - high_price) / high_price * 100.0
            if dip_pct <= min_dip_pct:
                return {
                    'high_date': dates[last_high_idx],
                    'high_price': high_price,
                    'low_date': dates[low_idx],
                    'low_price': low_price,
                    'dip_pct': dip_pct
                }
            else:
                return None
        i -= 1
    return None

def check_symbol_timeframe(symbol, bar_size, duration, tf_label):
    df = fetch_ib_history(symbol, bar_size, duration)
    if df.empty or 'close' not in df:
        print(f"No data for {symbol} {tf_label}")
        return
    info = detect_latest_dip(df, min_dip_pct=-1.0)  # at least -1%
    if info is None:
        print(f"{symbol} {tf_label}: No new qualifying dip detected.")
        return
    exp = EXPECTED.get((symbol, tf_label))
    exp_reb = exp['avg_reb'] if exp else None
    print(f"\n=== DIP ALERT: {symbol} {tf_label} ===")
    print(f"Prior local high: {info['high_date']} @ {info['high_price']:.2f}")
    print(f"Turn-up low:      {info['low_date']} @ {info['low_price']:.2f}")
    print(f"Dip depth:        {info['dip_pct']:.2f}%")
    if exp_reb is not None:
        tgt = info['low_price'] * (1 + exp_reb/100.0)
        print(f"Expected rebound: ~{exp_reb:.2f}%  -> target price ≈ {tgt:.2f}")
    else:
        print("No historical rebound stats available for this symbol/TF.")

for sym in symbols:
    print(f"\n======== {sym} CHECK ========")
    # Daily: 5Y
    check_symbol_timeframe(sym, '1 day', '5 Y', '1D')
    # 4H: 2Y
    check_symbol_timeframe(sym, '4 hours', '2 Y', '4H')

ib.disconnect()
