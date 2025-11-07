import numpy as np
import pandas as pd
from ib_insync import IB, Stock
import ruptures as rpt
from hmmlearn.hmm import GaussianHMM

def regime_label(mean_return, threshold=0.001):
    if mean_return > threshold:
        return "bullish"
    elif mean_return < -threshold:
        return "bearish"
    else:
        return "consolidation"

ib = IB()
ib.connect('127.0.0.1', 4001, clientId=1)

tickers = ['QQQ', 'SPY','TQQQ','SSO']  # Example tickers
timeframes = [
  
    ('daily', '1 day', '1D', '1 Y')
]

for ticker in tickers:
    contract = Stock(ticker, 'SMART', 'USD')
    for timeframe_name, ib_bar_size, pandas_freq, durationStr in timeframes:
        print(f"\n====== {ticker} | {timeframe_name} ======")
        bars = ib.reqHistoricalData(
            contract, endDateTime='',
            durationStr=durationStr,
            barSizeSetting=ib_bar_size,
            whatToShow='TRADES', useRTH=True, formatDate=1)
        if not bars or len(bars) < 15:
            print("No data returned or too little data for analysis.")
            continue
        df = pd.DataFrame(bars)
        df['returns'] = df['close'].pct_change()
        df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=['returns']).reset_index(drop=True)
        signal = df['returns'].values

        if len(signal) < 10 or len(np.unique(signal)) < 3:
            print("Not enough valid or unique data for HMM/CPD.")
            continue

        # CPD detection
        try:
            model = "l2"
            algo = rpt.Pelt(model=model, min_size=5, jump=1).fit(signal)
            result = algo.predict(pen=5)
            df['cpd_regime'] = 0
            prev = 0
            for idx, cp in enumerate(result):
                df.loc[df.index[prev:cp], 'cpd_regime'] = idx
                prev = cp
        except Exception as e:
            print("CPD error:", e)
            result = []

        # HMM detection
        try:
            hmm_model = GaussianHMM(n_components=min(3,len(np.unique(signal))), covariance_type='full', n_iter=100)
            hmm_model.fit(signal.reshape(-1, 1))
            hidden_states = hmm_model.predict(signal.reshape(-1, 1))
            df['hmm_regime'] = hidden_states
        except Exception as e:
            print("HMM error:", e)
            hidden_states = np.array([])

        # CPD regime changes - last 5
        print("\nLast 5 CPD regime changes:")
        if isinstance(result, list) and len(result) > 1:
            indices = result[-6:-1] if len(result) > 6 else result[:-1]
            for i, cp_idx in enumerate(indices):
                dt_val = df.loc[cp_idx, 'date'] if 'date' in df.columns else df.index[cp_idx]
                close_val = df.loc[cp_idx, 'close']
                prev_cp = indices[i-1] if i > 0 else 0
                mean_ret = df.loc[prev_cp:cp_idx, 'returns'].mean() if prev_cp < cp_idx else np.nan
                regime = regime_label(mean_ret) if not np.isnan(mean_ret) else "unknown"
                print(f"Date: {dt_val} | Price: {close_val:.2f} | Regime: {regime}")
        else:
            print("No regime changes detected by CPD.")

        # HMM regime changes - last 5
        print("\nLast 5 HMM regime changes:")
        if hidden_states.size > 0:
            hmm_transitions = np.where(np.diff(hidden_states) != 0)[0]
            indices = hmm_transitions[-5:] if len(hmm_transitions) > 5 else hmm_transitions
            for idx, i in enumerate(indices):
                dt_val = df.loc[i+1, 'date'] if 'date' in df.columns else df.index[i+1]
                close_val = df.loc[i+1, 'close']
                new_regime = hidden_states[i+1]
                # estimate next transition or end of data
                if idx+1 < len(indices):
                    next_idx = indices[idx+1]
                else:
                    next_idx = len(df)-1
                mean_ret = df.loc[i+1:next_idx, 'returns'].mean() if i+1 < next_idx else np.nan
                regime = regime_label(mean_ret) if not np.isnan(mean_ret) else "unknown"
                print(f"Date: {dt_val} | Price: {close_val:.2f} | HMM regime: {new_regime} | Regime: {regime}")
        else:
            print("No regime changes detected by HMM.")

ib.disconnect()
