import time
import pandas as pd
import numpy as np
from threading import Event
from ibapi.client import EClient
from ibapi.wrapper import EWrapper
from ibapi.scanner import ScannerSubscription
from ibapi.contract import Contract
from ibapi.common import *
from ibapi.ticktype import *
from ibapi.utils import iswrapper

def smma(series, window):
    s = pd.Series(series)
    if len(s) < window:
        return s.copy()
    result = s.copy()
    result.iloc[:window] = s.iloc[:window].mean()
    for i in range(window, len(result)):
        result.iloc[i] = (result.iloc[i-1]*(window-1) + s.iloc[i])/window
    return result

def crossed_up(slow, fast):
    return (fast.iloc[-2] <= slow.iloc[-2]) and (fast.iloc[-1] > slow.iloc[-1])

class IBScanner(EWrapper, EClient):
    def __init__(self):
        EClient.__init__(self, self)
        self.scanner_data = []
        self.scanner_done = Event()
        self.hist_data = {}
        self.hist_done = Event()

    def nextValidId(self, orderId: int):
        # Start the scanning process
        print("Connected. Requesting scanner...")
        scan_sub = ScannerSubscription()
        scan_sub.instrument = "STK"
        scan_sub.locationCode = "STK.US.MAJOR"
        scan_sub.scanCode = "HIGH_MO"
        scan_sub.abovePrice = 3
        scan_sub.marketCapAbove = 500_000_000
        self.reqScannerSubscription(4001, scan_sub, [], [])

    def scannerData(self, reqId, rank, contractDetails, distance, benchmark, projection, legsStr):
        symbol = contractDetails.contract.symbol
        exchange = contractDetails.contract.exchange
        if rank < 30:  # limit the count for speed
            print(f"Scanner found: {symbol}")
            self.scanner_data.append(symbol)

    def scannerDataEnd(self, reqId):
        print("Scanner complete.")
        self.scanner_done.set()

    def historicalData(self, reqId, bar):
        # bar has .date, .close, .high, .low
        self.hist_data.setdefault(reqId, []).append(bar)

    def historicalDataEnd(self, reqId: int, start: str, end: str):
        self.hist_done.set()

def main():
    app = IBScanner()
    app.connect("127.0.0.1", 4001, 1)
    time.sleep(1)  # let connection establish

    # Wait for scanner results
    app.scanner_done.wait(timeout=10)  # up to 10s for scanner
    if not app.scanner_data:
        time.sleep(3)
    app.scanner_done.wait(timeout=15)

    found_symbols = app.scanner_data[:10]  # scan will likely return fast- so take the first N
    print("Symbols to check for crossover:", found_symbols)

    crossover_candidates = []
    req_id = 5000
    for sym in found_symbols:
        contract = Contract()
        contract.symbol = sym
        contract.secType = "STK"
        contract.exchange = "SMART"
        contract.currency = "USD"

        app.reqHistoricalData(
            req_id, contract, "", "80 D", "1 day", "TRADES", 1, 1, False, []
        )
        app.hist_done.wait(timeout=15)
        bars = app.hist_data.get(req_id, [])
        if not bars or len(bars) < 45:
            req_id += 1
            continue

        df = pd.DataFrame(
            {
                "date": [b.date for b in bars],
                "close": [b.close for b in bars],
                "high": [b.high for b in bars],
                "low": [b.low for b in bars],
            }
        )
        df["smma12"] = smma(df["close"], 12)
        df["smma40"] = smma(df["close"], 40)
        if crossed_up(df["smma40"], df["smma12"]):
            print(f"{sym} crossed up 12/40 SMMA today at close {df['close'].iloc[-1]:.2f}")
            crossover_candidates.append(sym)
        app.hist_done.clear()
        req_id += 1

    print("Final crossover candidates:", crossover_candidates)
    app.disconnect()

if __name__ == "__main__":
    main()
