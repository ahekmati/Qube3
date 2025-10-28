import yfinance as yf
import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM

def analyze_regimes(df, n_states=3, win_vol=5):
    df = df.rename(columns={'Close':'close'})
    df['log_ret'] = np.log(df['close'] / df['close'].shift(1))
    df['volatility'] = df['log_ret'].rolling(window=win_vol).std()
    df = df.dropna()

    hmm_features = df[['log_ret', 'volatility']].values
    hmm_model = GaussianHMM(n_components=n_states, covariance_type='full', n_iter=2000, random_state=2025)
    hmm_model.fit(hmm_features)
    hidden_states = hmm_model.predict(hmm_features)
    df['regime'] = hidden_states

    # Regime characterization: map to bullish, bearish, hold by mean return
    regime_stats = df.groupby('regime')['log_ret'].mean()
    regime_map = {}
    max_state = regime_stats.idxmax()
    min_state = regime_stats.idxmin()
    for state in regime_stats.index:
        if state == max_state:
            regime_map[state] = 'bullish'
        elif state == min_state:
            regime_map[state] = 'bearish'
        else:
            regime_map[state] = 'hold'
    
    df['regime_label'] = df['regime'].map(regime_map)
    # Identify regime change dates
    df['regime_shift'] = df['regime'] != df['regime'].shift(1)
    change_points = df[df['regime_shift']][['regime_label']]

    last5 = df['regime_label'].iloc[-5:].tolist()
    current_label = df['regime_label'].iloc[-1]
    if current_label == 'bullish':
        action = "BUY"
    elif current_label == 'bearish':
        action = "SHORT"
    else:
        action = "HOLD"

    return current_label, last5, action, change_points

# Daily candles
df_day = yf.download('QQQ', period='2y', interval='1d')
day_label, day_last5, day_action, day_changes = analyze_regimes(df_day)

# 4 Hour candles
df_4h = yf.download('QQQ', period='180d', interval='4h')
fourh_label, fourh_last5, fourh_action, fourh_changes = analyze_regimes(df_4h)

print("\n=== QQQ Daily Regime Analysis ===")
print(f"Current regime: {day_label}")
print(f"Last 5 regimes: {day_last5}")
print(f"Action: {day_action}")
print("\nRegime change dates (daily):")
print(day_changes.tail(15))

print("\n=== QQQ 4-hour Regime Analysis ===")
print(f"Current regime: {fourh_label}")
print(f"Last 5 regimes: {fourh_last5}")
print(f"Action: {fourh_action}")
print("\nRegime change dates (4h):")
print(fourh_changes.tail(15))
