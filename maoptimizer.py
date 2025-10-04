import pandas as pd
import numpy as np
from ib_insync import IB, Stock, util

# -------- Moving Average Functions --------
def sma(series, window):
    return pd.Series(series).rolling(window=window).mean()

def ema(series, window):
    return pd.Series(series).ewm(span=window, adjust=False).mean()

def wma(series, window):
    weights = np.arange(1, window + 1)
    return pd.Series(series).rolling(window).apply(
        lambda x: np.dot(x, weights)/weights.sum(), raw=True)

def smma(series, window):
    s = pd.Series(series)
    if len(s) < window: return s.copy()
    result = s.copy()
    result.iloc[:window] = s.iloc[:window].mean()
    for i in range(window, len(result)):
        result.iloc[i] = (result.iloc[i-1]*(window-1) + s.iloc[i])/window
    return result

MA_FUNCTIONS = {'SMA': sma, 'EMA': ema, 'WMA': wma, 'SMMA': smma}

# ----------- Signal Backtest -------------
def backtest_ma_strategy(df, fast, slow, ma_type):
    # Calculate MAs
    fast_ma = MA_FUNCTIONS[ma_type](df['close'], fast)
    slow_ma = MA_FUNCTIONS[ma_type](df['close'], slow)
    df = df.copy()
    df['fast_ma'] = fast_ma
    df['slow_ma'] = slow_ma

    df['position'] = 0
    # Crossover logic
    df.loc[
        (df['fast_ma'].shift(1) <= df['slow_ma'].shift(1)) & (df['fast_ma'] > df['slow_ma']), 'position'
    ] = 1
    df.loc[
        (df['fast_ma'].shift(1) >= df['slow_ma'].shift(1)) & (df['fast_ma'] < df['slow_ma']), 'position'
    ] = 0
    df['position'] = df['position'].replace(to_replace=0, method='ffill').fillna(0)
    df['returns'] = df['close'].pct_change() * df['position']
    # Return sharpe and total return
    total_return = (df['returns'] + 1).prod() - 1
    sharpe = df['returns'].mean() / df['returns'].std() * np.sqrt(252) if df['returns'].std() != 0 else 0
    trades = df['position'].diff().abs().sum() / 2
    return total_return, sharpe, trades

# ----------- IB Fetch Helper --------------
def fetch_historical_data(ib, symbol, duration_days, bar_size):
    contract = Stock(symbol, 'SMART', 'USD')
    bars = ib.reqHistoricalData(
        contract, '', f'{duration_days} D', bar_size, 'TRADES', useRTH=True
    )
    if not bars or len(bars) == 0:
        return pd.DataFrame()
    df = util.df(bars)
    return df

# --------- Best Combo Finder -------------
def find_best_ma_combo(ib, symbol, candidate_types, candidate_pairs, duration_days=180):
    results = {}
    for label, bar_size in [('daily', '1 day'), ('4h', '4 hours')]:
        df = fetch_historical_data(ib, symbol, duration_days, bar_size)
        if df.empty or len(df) < 50:
            continue
        best_score = -np.inf
        best_combo = None
        for ma_type in candidate_types:
            for fast, slow in candidate_pairs:
                if fast >= slow: continue
                total_ret, sharpe, trades = backtest_ma_strategy(df, fast, slow, ma_type)
                # Sharpe with bonus for more trades, adjust scoring as needed
                score = sharpe + 0.01 * trades
                if score > best_score:
                    best_score = score
                    best_combo = (ma_type, fast, slow, sharpe, total_ret, trades)
        if best_combo:
            results[label] = {
                'type': best_combo[0],
                'fast': best_combo[1],
                'slow': best_combo[2],
                'sharpe': best_combo[3],
                'total_return': best_combo[4],
                'trades': best_combo[5]
            }
    return results

# ----------- Main Loop --------------------
def main():
    ib = IB()
    ib.connect('127.0.0.1', 4001, clientId=11)
    tickers = [ "AAPL", "AEVA", "AMD", "AMLX", "AMPX", "AMZN", "APP", "APLD", "APPS", "ARKK", "AVGO", "AXON", "BAP", "BBVA", "BE",
    "BILI", "BSBR", "CANG", "CDE", "CEG", "CELC", "CIB", "CMCL", "COMM", "CPNG", "CPS", "CRWD", "CVS", "CZR", "DASH",
    "DB", "EC", "EEM", "FAST", "FUBO", "GDS", "GE", "GEV", "GDXJ", "GH", "GLW", "GOOG", "GOOGL", "HWM", "IBIT", "IDXX",
    "ILMN", "IMRX", "ING", "IONQ", "ITUB", "JBL", "JAZZ", "KLAC", "KTOS", "LASR", "LCTX", "LEU", "LRCX", "LYG", "MASS",
    "MDB", "MELI", "META", "METC", "MLYS", "MPWR", "MSFT", "MU", "NBIS", "NEM", "NFLX", "NRG", "NUGT", "NU", "NVDA",
    "NVS", "OPEN", "OPRX", "ORCL", "ORLY", "PAGS", "PBR", "PDD", "PGEN", "PLTR", "PRGO", "QTUM", "QGEN", "QQQ", "RIO",
    "RPRX", "SATS", "SCCO", "SCPH", "SHEL", "SMCI", "SMFG", "SOXL", "SPY", "SSL", "SSO", "STX", "SVXY", "STM",
    "TDUP", "TEL", "TEVA", "TME", "TPR", "TSLA", "TSLL", "TTWO", "TV", "UBER", "UUUU", "VALE", "VIPS", "VST", "WBD",
    "WDC", "WYNN", "YPF", "ZS"]  # <-- Replace with your tickers!
    candidate_types = ['SMA', 'EMA', 'WMA', 'SMMA']
    candidate_pairs = [(9,18), (13,31), (21,50), (26,150), (10,20), (5,14)]
    all_results = {}
    for tkr in tickers:
        print(f"\nAnalyzing {tkr} ...")
        combos = find_best_ma_combo(ib, tkr, candidate_types, candidate_pairs)
        if combos:
            print(f"{tkr} BEST COMBOS:")
            for tf, info in combos.items():
                print(f"  {tf.upper()}: {info['type']} {info['fast']}/{info['slow']} | Sharpe: {info['sharpe']:.3f} | Return: {info['total_return']:.2%} | Trades: {info['trades']}")
        else:
            print(f"{tkr}: No sufficient data or signals.")
        all_results[tkr] = combos
    ib.disconnect()
    return all_results

if __name__ == "__main__":
    results = main()
    print("\n--- SUMMARY ---")
    for tkr, res in results.items():
        print(f"{tkr}: {res}")
