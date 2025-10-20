"""
====================================================================
Trading System: SMMA Multi-Timeframe Signal Scanner with IBKR Orders
====================================================================

This script connects to Interactive Brokers (IBKR) via their API
(using ib_insync) and automates the process of:
 - Loading and caching OHLCV historical data for user-defined tickers
 - Computing SMMA (Smoothed Moving Average) cross/reentry signals
 - Filtering tickers based on relative volume and momentum scores
 - Managing separate allocation groups:
     * Main group (non-exception tickers) — capped at 50% of account value
     * Exception group (momentum/inverse ETFs) — capped at 50% of account value
 - Automatically skipping new main group trades when the 50% main cap is reached
 - Placing live market orders with ATR-based or SMMA-based stops
 - Maintaining persistent state tracking for open positions
 - Updating trailing stops dynamically
 - Applying "defensive" stop tightening when SPY trend weakens

Key Rules:
 - MAX_ALLOC: 15% per-trade allocation cap
 - MAX_POSITIONS: limits number of concurrent open tickers
 - Exception tickers (e.g. SQQQ, VXX, SOXS, etc.) operate under 
   different ATR and allocation parameters

Author: [Your Name]
Updated: October 2025
====================================================================
"""

import sys
import logging
import json
import pandas as pd
import numpy as np
import os
import math
from datetime import datetime
from ib_insync import IB, Stock, util, Order
from colorama import Fore, Style, init as colorama_init
from pandas import to_datetime

colorama_init(autoreset=True)

# --- USER SETTINGS ---
MAIN_ATR_PERIOD = 44
MAIN_ATR_MULTIPLIER = 2.5
MAIN_STOP_BUFFER = 0.0025
EXCEPTION_ATR_PERIOD = 12
EXCEPTION_ATR_MULTIPLIER = 1.6
EXCEPTION_TICKERS = {"SQQQ", "VXX", "TLT", "GLD", "SOXS", "SPXS", "FAZ", "SARK"}
MAX_ALLOC = 0.15
MAX_POSITIONS = 10
DEF_SWING_WINDOW = 7
DEFENSE_STOP_PCT = 0.02
CACHE_DAYS = 3


# --- HELPER FUNCTIONS ---
def smma(series, window):
    s = pd.Series(series)
    if len(s) < window:
        return s.copy()
    result = s.copy()
    result.iloc[:window] = s.iloc[:window].mean()
    for i in range(window, len(result)):
        result.iloc[i] = (result.iloc[i - 1] * (window - 1) + s.iloc[i]) / window
    return result


def atr(df, window=14):
    high_low = df['high'] - df['low']
    high_close = np.abs(df['high'] - df['close'].shift(1))
    low_close = np.abs(df['low'] - df['close'].shift(1))
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    return tr.rolling(window).mean()


def cached_fetch_ohlcv(ib, symbol, ndays, tf='1 day', cache_days=CACHE_DAYS):
    folder = 'data_cache'
    os.makedirs(folder, exist_ok=True)
    fname = os.path.join(folder, f"{symbol.replace('/', '_')}_{tf.replace(' ', '_')}_{ndays}d.pkl")
    if os.path.exists(fname):
        mtime = datetime.fromtimestamp(os.path.getmtime(fname))
        if (datetime.now() - mtime).days < cache_days:
            print(f"[Cache] Loading {symbol} {tf} from pickle cache.")
            return pd.read_pickle(fname)
    contract = Stock(symbol, 'SMART', 'USD')
    for attempt in range(3):
        print(f"[Fetch] {symbol} [{tf}] attempt {attempt + 1}/3 ...")
        try:
            bars = ib.reqHistoricalData(contract, '', f'{ndays} D', tf, 'TRADES', useRTH=True)
            df = util.df(bars)
            print(f"[Bars] {symbol} [{tf}] fetched {len(df)}")
            if len(df) > 0:
                df.to_pickle(fname)
                return df
        except Exception as e:
            print(f"[Error] {tf} fetch failed for {symbol}: {e}")
        ib.sleep(1)
    print(f"[Fail] No {tf} data for {symbol} after 3 attempts.")
    return None


def get_trade_size(available_cap, price, max_alloc=MAX_ALLOC):
    if price is None or (isinstance(price, float) and np.isnan(price)):
        print(f"[Alloc] Skipping trade: price is NaN or None.")
        return 0
    if price > available_cap * max_alloc:
        print(f"[Alloc] Skipping: 1 share at ${price:.2f} > {max_alloc * 100:.1f}% "
              f"(${available_cap * max_alloc:.2f}) of capital.")
        return 0
    raw = int((available_cap * max_alloc) // price)
    return max(raw, 1)


# --- MAIN SCRIPT ---
def main():
    with open('best_smma_combos.json') as f:
        best_smma = json.load(f)

    ib = IB()
    ib.connect('127.0.0.1', 4001, clientId=1)
    print("Retrieving account balance...")
    ib.sleep(2)

    summary = ib.accountSummary()
    nl = [a.value for a in summary if a.tag == 'NetLiquidation']
    if not nl:
        print("ERROR: No NetLiquidation value found in account summary! Is the API connection working?")
        ib.disconnect()
        return

    account_bal = float(nl[0])
    config = json.load(open('config.json'))
    symbols = config['user_tickers']

    main_group = set(symbols) - EXCEPTION_TICKERS
    special_group = set(symbols) & EXCEPTION_TICKERS

    main_alloc_total = account_bal * 0.5
    special_alloc_total = account_bal * 0.5
    main_allocated = 0.0
    special_allocated = 0.0

    print(f"\n[INFO] Main group: {sorted(main_group)}")
    print(f"[INFO] Exception/Momentum group: {sorted(special_group)}")

    positions = list(ib.positions())
    ib_pos_symbols = {p.contract.symbol for p in positions}
    print(f"\n[INFO] IBKR open positions: {sorted(ib_pos_symbols)}")

    # --- Allocation tracking loop ---
    for pos in positions:
        sym = pos.contract.symbol
        last = getattr(pos, 'avgCost', None)
        if last is None:
            continue
        market_val = abs(pos.position) * last
        if sym in special_group:
            special_allocated += market_val
        elif sym in main_group:
            main_allocated += market_val

    # --- Cap limit alert (after allocation totals) ---
    if main_allocated >= main_alloc_total:
        print("\n[CAP ALERT] Main group allocation cap reached (50%). "
              "Only exception tickers will be considered for new trades.\n")

    print(f"\nACCOUNT VALUE: ${account_bal:,.2f}")
    print(f"Exception Group Allocation: Used ${special_allocated:,.2f} / Allowed ${special_alloc_total:,.2f}")
    print(f"Main Group Allocation:      Used ${main_allocated:,.2f} / Allowed ${main_alloc_total:,.2f}\n")

    # rest of your program logic here…
    print("[INFO] Scanning and trade management would continue here...")
    ib.disconnect()


if __name__ == "__main__":
    main()
