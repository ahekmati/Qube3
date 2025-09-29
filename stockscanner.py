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

def run_scanner(ib, scan_code):
    scanner_sub = ScannerSubscription()
    scanner_sub.instrument = 'STK'
    scanner_sub.locationCode = 'STK.US.MAJOR'
    scanner_sub.scanCode = scan_code
    scanner_sub.abovePrice = 10
    scanner_sub.aboveMarketCap = 500

    scan_results = ib.reqScannerData(scanner_sub)
    print(f"\nScanning {len(scan_results)} tickers with code '{scan_code}'...")
    result_stocks = []

    for res in scan_results:
        contract = res.contractDetails.contract
        symbol = contract.symbol
        exchange = contract.exchange or 'SMART'
        print(f"Checking {symbol}...")
        bars = ib.reqHistoricalData(contract, '', '15 D', '1 day', 'TRADES', useRTH=False)
        if not bars or len(bars) < 20:
            continue

        df = util.df(bars)
        df['smma_fast'] = smma(df['close'], 9)
        df['smma_slow'] = smma(df['close'], 18)
        df['TSI'] = tsi(df['close'])
        df['rvol'] = relative_volume(df['volume'])

        # Require average daily dollar volume (last 10 days) > $10M
        dollar_vol10 = (df['close'][-10:] * df['volume'][-10:]).mean()
        if dollar_vol10 < 1e7:
            continue

        # Find bullish cross in last 7 days with momentum
        crosses = []
        for i in range(1, len(df)):
            cond_smma = (df['smma_fast'].iloc[i] > df['smma_slow'].iloc[i]) and (df['smma_fast'].iloc[i-1] <= df['smma_slow'].iloc[i-1])
            cond_rvol = df['rvol'].iloc[i] > 2
            cond_price = df['close'].iloc[i] > 10
            cond_tsi = tsi_bull_cross(df['TSI']).iloc[i]
            dollar_vol = df['close'].iloc[i] * df['volume'].iloc[i] > 1e7

            recent_date = df.index[i]
            if (datetime.now() - recent_date).days <= 7:
                if cond_smma and cond_rvol and cond_price and cond_tsi and dollar_vol:
                    crosses.append(i)

        if crosses:
            xidx = crosses[-1]
            result_stocks.append({
                "symbol": symbol,
                "date": str(df.index[xidx].date()),
                "price": round(df['close'].iloc[xidx], 2),
                "volume": int(df['volume'].iloc[xidx]),
                "rvol": round(df['rvol'].iloc[xidx], 2),
                "TSI": round(df['TSI'].iloc[xidx], 2),
                "exchange": exchange,
                "scan_code": scan_code
            })
    return result_stocks

def main():
    ib = IB()
    ib.connect('127.0.0.1', 4001, clientId=1001)

    all_results = []
    for code in ['TOP_PERC_GAIN', 'HOT_BY_VOLUME']:
        results = run_scanner(ib, code)
        all_results.extend(results)

    ib.disconnect()

    print("\n=== Stocks with Recent Bullish Cross and Breakout Setup ===")
    if all_results:
        for s in all_results:
            print(f"{s['symbol']} ({s['exchange']}, {s['scan_code']}): Crossed on {s['date']}, Price ${s['price']}, Vol {s['volume']}, rVol {s['rvol']}, TSI {s['TSI']}")
    else:
        print("No matches found based on scanner filters.")

if __name__ == "__main__":
    main()
