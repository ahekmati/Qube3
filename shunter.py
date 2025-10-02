import numpy as np
import pandas as pd
from ib_insync import *
from datetime import datetime

def smma(series, window):
    s = pd.Series(series)
    out = s.copy()
    out.iloc[:window] = s.iloc[:window].mean()
    for i in range(window, len(out)):
        out.iloc[i] = (out.iloc[i-1]*(window-1)+s.iloc[i])/window
    return out

def tsi(close, r=25, s=13):
    pc = close.diff()
    double_smoothed_pc = smma(smma(pc, r), s)
    double_smoothed_abs_pc = smma(smma(pc.abs(), r), s)
    return 100 * double_smoothed_pc / double_smoothed_abs_pc

def tsi_bull_cross(tsi_series):
    return (tsi_series > 0) & (tsi_series.shift(1) <= 0)

def relative_volume(volume, avgN=10):
    avg_vol = volume.rolling(avgN).mean()
    return volume / avg_vol

def run_analysis(ib, symbol, exchange='SMART'):
    contract = Stock(symbol, exchange, 'USD')
    bars = ib.reqHistoricalData(contract, '', '15 D', '1 day', 'TRADES', useRTH=False)
    if not bars or len(bars) < 20:
        return None, None

    df = util.df(bars)
    df['smma_fast'] = smma(df['close'], 9)
    df['smma_slow'] = smma(df['close'], 18)
    df['TSI'] = tsi(df['close'])
    df['rvol'] = relative_volume(df['volume'])
    dollar_vol10 = (df['close'][-10:] * df['volume'][-10:]).mean()
    if dollar_vol10 < 1e7:
        return None, None

    crosses_with_tsi = []
    crosses_without_tsi = []
    for i in range(1, len(df)):
        cond_smma = (df['smma_fast'].iloc[i] > df['smma_slow'].iloc[i]) and (df['smma_fast'].iloc[i-1] <= df['smma_slow'].iloc[i-1])
        cond_rvol = df['rvol'].iloc[i] > 2
        cond_price = df['close'].iloc[i] > 10
        cond_tsi = tsi_bull_cross(df['TSI']).iloc[i]
        dollar_vol = df['close'].iloc[i] * df['volume'].iloc[i] > 1e7
        recent_date = df.index[i]
        if (datetime.now() - recent_date).days <= 7:
            if cond_smma and cond_rvol and cond_price and dollar_vol:
                if cond_tsi:
                    crosses_with_tsi.append(i)
                crosses_without_tsi.append(i)

    result_with_tsi = None
    if crosses_with_tsi:
        xidx = crosses_with_tsi[-1]
        result_with_tsi = {
            "symbol": symbol,
            "date": str(df.index[xidx].date()),
            "price": round(df['close'].iloc[xidx], 2),
            "volume": int(df['volume'].iloc[xidx]),
            "rvol": round(df['rvol'].iloc[xidx], 2),
            "TSI": round(df['TSI'].iloc[xidx], 2),
            "exchange": exchange
        }
    result_without_tsi = None
    # Only add to no-TSI if not already found in TSI to avoid duplicate printing
    if crosses_without_tsi:
        xidx = crosses_without_tsi[-1]
        if not result_with_tsi or xidx != crosses_with_tsi[-1]:
            result_without_tsi = {
                "symbol": symbol,
                "date": str(df.index[xidx].date()),
                "price": round(df['close'].iloc[xidx], 2),
                "volume": int(df['volume'].iloc[xidx]),
                "rvol": round(df['rvol'].iloc[xidx], 2),
                "TSI": round(df['TSI'].iloc[xidx], 2),
                "exchange": exchange
            }
    return result_with_tsi, result_without_tsi

def main():
    ib = IB()
    ib.connect('127.0.0.1', 4001, clientId=1001)

    watchlist = [   "AAPL", "MSFT", "NVDA", "TSLA", "AMZN", "META", "GOOGL", "GOOG", "AVGO",
        "ORCL", "WDC", "NEM", "STX", "GEV", "GE", "CVS", "NRG", "HWM", "RCL",
        "PLTR", "JPM", "LLY", "V", "NFLX", "XOM", "MA", "COST", "WMT", "PG",
        "JNJ", "HD", "ABBV", "BAC", "IREN", "MSTR", "INTC", "QBTS", "BMNR", "IONQ",
        "CRWV", "UNP", "TXN", "CRWD", "SBUX", "TSM", "SNOW", "PYPL", "AMD", "SHOP",
        # Your new symbols
        "TSLL", "TQQQ", "NUGT", "EEM", "SOXL", "IBIT", "ARKK", "QTUM", "SVXY",
        "STX", "PLTR", "WDC", "NEM", "MU", "GEV", "ORCL", "WBD", "NRG", "GE", "APH",
        "LRCX", "TPR", "HWM", "GLW", "CVS", "KLAC", "UBER", "IDXX", "DASH", "JBL",
        "MPWR", "VST", "TEL", "WYNN",
        "APP", "ZS", "FAST", "AVGO", "CEG", "MDB", "MELI", "ORLY", "TTWO", "NFLX",
        "NVDA", "AMD", "PDD", "META", "CRWD", "AXON", "GOOGL", "GOOG",
        "TDUP", "OPEN", "LEU", "IMRX", "AMPX", "CELC", "AEVA", "BE", "OPRX", "CMCL",
        "AMLX", "FUBO", "PGEN", "MASS", "APPS", "MLYS", "SATS", "COMM", "KTOS",
        "LCTX", "CDE", "LASR", "METC", "UUUU", "CPS", "SSO", "QQQ", "SPY", "APP"
        # Insert your full deduplicated ticker list here as in previous response.
    ]

    # Deduplicate and prepare pairs
    watchlist = list(dict.fromkeys(watchlist))

    results_with_tsi = []
    results_without_tsi = []
    for symbol in watchlist:
        print(f"Analyzing {symbol}...")
        res_with, res_without = run_analysis(ib, symbol, "SMART")
        if res_with:
            results_with_tsi.append(res_with)
        if res_without:
            results_without_tsi.append(res_without)

    ib.disconnect()

    print("\n=== Stocks with Recent Bullish Cross AND TSI Breakout ===")
    if results_with_tsi:
        for s in results_with_tsi:
            print(f"{s['symbol']} ({s['exchange']}): Crossed on {s['date']}, Price ${s['price']}, Vol {s['volume']}, rVol {s['rvol']}, TSI {s['TSI']}")
    else:
        print("No matches found with TSI filter.")

    print("\n=== Stocks with Recent Bullish Cross WITHOUT TSI Filter ===")
    if results_without_tsi:
        for s in results_without_tsi:
            print(f"{s['symbol']} ({s['exchange']}): Crossed on {s['date']}, Price ${s['price']}, Vol {s['volume']}, rVol {s['rvol']}, TSI {s['TSI']}")
    else:
        print("No matches found without TSI filter.")

if __name__ == "__main__":
    main()

# =============================================================================
# Script Summary:
#
# This script scans a large watchlist of stock tickers for bullish technical signals,
# using price and volume data from Interactive Brokers (IBKR).
# For each symbol, it:
#   - Fetches up to 15 days of daily market data.
#   - Calculates fast and slow SMMA (Smoothed Moving Average) signals on closing prices.
#   - Computes TSI (True Strength Index) and relative volume (rVol) indicators.
#   - Identifies recent bullish cross events where fast SMMA crosses above slow SMMA,
#     filtering for strong volume and price, and optionally for TSI "breakout" confirmation.
#   - Aggregates and prints all matches in two lists: those confirmed by TSI, and those 
#     detected without TSI filter.
#   - Produces clear summary tables for actionable opportunities with relevant stats.
#
# The workflow helps identify stocks experiencing momentum shifts validated by price,
# volume, and trend indicators, and guides traders for further action.
# =============================================================================
