from ib_insync import *
import pandas as pd
import numpy as np

# Connect to Interactive Brokers API
ib = IB()
ib.connect('127.0.0.1', 4001, clientId=1)  # adjust port/clientId as needed

def get_momentum_score_ibkr(symbol, lookback=20):
    contract = Stock(symbol, 'SMART', 'USD')
    bars = ib.reqHistoricalData(
        contract,
        endDateTime='',
        durationStr=f'{lookback + 2} D',
        barSizeSetting='1 day',
        whatToShow='TRADES',
        useRTH=True,
        formatDate=1
    )
    if len(bars) < lookback + 1:
        return None, None, None

    df = util.df(bars)
    if df.empty:
        return None, None, None

    # Relative Volume (RVOL)
    prev_vol = df['volume'][-lookback-1:-1].mean()
    today_vol = df['volume'].iloc[-1]
    rvol = today_vol / prev_vol if prev_vol > 0 else 0

    # Momentum: % Change from N days ago
    momentum = (df['close'].iloc[-1] - df['close'].iloc[-lookback]) / df['close'].iloc[-lookback]

    # Composite Score
    score = momentum * 0.6 + rvol * 0.4

    return score, rvol, momentum

def rank_tickers_ibkr(ticker_list, lookback=20):
    results = []
    for symbol in ticker_list:
        try:
            score, rvol, momentum = get_momentum_score_ibkr(symbol, lookback)
            if score is not None:
                results.append({
                    'Ticker': symbol,
                    'Score': score,
                    'RVOL': rvol,
                    'Momentum': momentum
                })
        except Exception as e:
            print(f"Error for {symbol}: {e}")

    df = pd.DataFrame(results)
    top5 = df.sort_values('Score', ascending=False).head(5)
    return top5

# Example Usage
my_tickers = ['AAPL', 'TSLA', 'NVDA', 'MSFT', 'AMZN', 'META', 'NFLX', 'QQQ', 'SPY', 'AMD']
top5_df = rank_tickers_ibkr(my_tickers, lookback=20)

print("Top 5 Momentum Stocks Based on Ranking Logic:")
for idx, row in top5_df.iterrows():
    print(f"{row['Ticker']}: Score={row['Score']:.3f}, RVOL={row['RVOL']:.3f}, Momentum={row['Momentum']:.3f}")

# Disconnect (optional)
ib.disconnect()
