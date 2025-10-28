import yfinance as yf
import numpy as np
import pandas as pd

# Download 2 years of daily data for Nasdaq (QQQ ETF proxy) and S&P 500 (SPY ETF proxy)
df_nq = yf.download('QQQ', period='2y', interval='1d')
df_es = yf.download('SPY', period='2y', interval='1d')

# Calculate daily log returns
df = pd.DataFrame()
df['NQ_return'] = np.log(df_nq['Close']).diff()
df['ES_return'] = np.log(df_es['Close']).diff()
df = df.dropna()

# Compute basis projections
def spread_market_components(nq, es):
    market = 0.5 * (nq + es)
    spread = 0.5 * (nq - es)
    return market, spread

def mean_reversion_signal(spread, lookback=20, threshold=2):
    spread_mean = pd.Series(spread).rolling(window=lookback).mean()
    spread_std = pd.Series(spread).rolling(window=lookback).std()
    z_score = (spread - spread_mean) / spread_std
    signal = np.where(z_score > threshold, -1, np.where(z_score < -threshold, 1, 0))
    return z_score, signal

market, spread = spread_market_components(df['NQ_return'], df['ES_return'])
z_score, signal = mean_reversion_signal(spread)

df['market'] = market
df['spread'] = spread
df['z_score'] = z_score
df['signal'] = signal

# Show last few signals
print(df.tail(150))
