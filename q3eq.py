# =============================================================================
# QUBE3 Systematic Equities Trading Script
#
# - Scans a custom ticker list for SMMA cross and momentum signals on multiple timeframes.
# - Uses a composite score (momentum, RVOL, age) to rank top trades per symbol.
# - Rapid pickle caching of OHLCV data for efficiency.
# - Auto-sizes trades, places entries, and manages dynamic trailing stops.
# - Stops are tightened for all positions if the SPY index is bearish.
# - Tracks and syncs state with IBKR account, cleans up orphan stops, and logs all trades.
#
# 
# =============================================================================

import sys
import logging
import json
import pandas as pd
import numpy as np
import os
from datetime import datetime
from ib_insync import IB, Stock, util, Order
from colorama import Fore, Style, init as colorama_init
from pandas import to_datetime

#sys.stdout = open("script_output.log", "w")
#sys.stderr = sys.stdout

colorama_init(autoreset=True)

# --- USER SETTINGS ---
ATR_PERIOD = 10
ATR_MULTIPLIER = 1.2
SUPERTREND_TRAIL_PERIOD = 22
SUPERTREND_TRAIL_MULTIPLIER = 2.0
EXCEPTION_TICKERS = {"SQQQ","VXX","TLT","GLD","SOXS","SPXS","FAZ","SARK"}
MAX_ALLOC = 0.15
MAX_POSITIONS = 10
DEF_SWING_WINDOW = 4
DEFENSE_STOP_PCT = 0.02
CACHE_DAYS = 2

def smma(series, window):
    s = pd.Series(series)
    if len(s) < window: return s.copy()
    result = s.copy()
    result.iloc[:window] = s.iloc[:window].mean()
    for i in range(window, len(result)):
        result.iloc[i] = (result.iloc[i-1]*(window-1) + s.iloc[i])/window
    return result

def atr(df, window=14):
    high_low = df['high'] - df['low']
    high_close = np.abs(df['high'] - df['close'].shift(1))
    low_close = np.abs(df['low'] - df['close'].shift(1))
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    return tr.rolling(window).mean()

def supertrend(df, period=SUPERTREND_TRAIL_PERIOD, multiplier=SUPERTREND_TRAIL_MULTIPLIER):
    df = df.copy()
    atr_ = atr(df, period)
    hl2 = (df['high'] + df['low']) / 2
    final_upperband = hl2 + (multiplier * atr_)
    final_lowerband = hl2 - (multiplier * atr_)
    supertrend_vals = []
    direction = []
    for i in range(len(df)):
        if i == 0:
            supertrend_vals.append(final_lowerband.iloc[i])
            direction.append("green")
            continue
        prev_st = supertrend_vals[-1]
        if df['close'].iloc[i] > prev_st:
            st = max(final_lowerband.iloc[i], prev_st)
            direction.append("green")
        else:
            st = min(final_upperband.iloc[i], prev_st)
            direction.append("red")
        supertrend_vals.append(st)
    return pd.Series(supertrend_vals, index=df.index), pd.Series(direction, index=df.index)

def cached_fetch_ohlcv(ib, symbol, ndays, tf='1 day', cache_days=CACHE_DAYS):
    folder = 'data_cache'
    os.makedirs(folder, exist_ok=True)
    fname = os.path.join(folder, f"{symbol.replace('/','_')}_{tf.replace(' ','_')}_{ndays}d.pkl")
    if os.path.exists(fname):
        mtime = datetime.fromtimestamp(os.path.getmtime(fname))
        if (datetime.now() - mtime).days < cache_days:
            print(f"[Cache] Loading {symbol} {tf} from pickle cache.")
            return pd.read_pickle(fname)
    contract = Stock(symbol, 'SMART', 'USD')
    for attempt in range(3):
        print(f"[Fetch] {symbol} [{tf}] attempt {attempt+1}/3 ...")
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

def recent_smma_signals(df, fast, slow, days_back=14, label="DAILY"):
    if df is None or len(df) < slow + days_back: return []
    df['smma_fast'] = smma(df['close'], fast)
    df['smma_slow'] = smma(df['close'], slow)
    signals = []
    lookback = min(days_back, len(df)-2)
    for i in range(len(df)-lookback, len(df)):
        prev, curr = i-1, i
        if prev < 0: continue
        bull = df['smma_fast'].iloc[prev] <= df['smma_slow'].iloc[prev] and df['smma_fast'].iloc[curr] > df['smma_slow'].iloc[curr]
        reentry = (
            df['smma_fast'].iloc[curr] > df['smma_slow'].iloc[curr] and
            df['close'].iloc[prev] < df['smma_slow'].iloc[prev] and
            df['close'].iloc[curr] > df['smma_slow'].iloc[curr]
        )
        if bull or reentry:
            signals.append({
                "signal_type": "cross" if bull else "reentry",
                "date": df.index[curr],
                "close": float(df["close"].iloc[curr]),
                "label": label
            })
    return signals

def get_momentum_rvol_score(df, lookback=20):
    if df is None or len(df) < lookback + 2:
        return None, None, None
    momentum = (df['close'].iloc[-1] - df['close'].iloc[-lookback]) / df['close'].iloc[-lookback]
    avg_vol = df['volume'].iloc[-lookback-1:-1].mean()
    today_vol = df['volume'].iloc[-1]
    rvol = today_vol / avg_vol if avg_vol > 0 else 0
    score = momentum * 0.6 + rvol * 0.4
    return score, rvol, momentum

def compute_signal_age(signal_date, today=None):
    if today is None:
        today = pd.Timestamp.now()
    if not isinstance(signal_date, pd.Timestamp):
        try:
            signal_date = pd.to_datetime(signal_date)
        except Exception:
            return 0
    if not isinstance(signal_date, pd.Timestamp):
        return 0
    return (today - signal_date).days

def get_trade_size(available_cap, price, max_alloc=MAX_ALLOC):
    if price > available_cap * max_alloc:
        print(f"[Alloc] Skipping: 1 share at ${price:.2f} > {max_alloc*100:.1f}% (${available_cap * max_alloc:.2f}) of capital.")
        return 0
    raw = int((available_cap * max_alloc) // price)
    return max(raw, 1)

def wait_for_fill(ib, trade):
    print("[Order] Waiting for buy order fill...")
    for _ in range(60):
        ib.sleep(0.5)
        if trade.orderStatus.status == "Filled":
            fill_price = getattr(trade.orderStatus, "avgFillPrice", None)
            filled_qty = getattr(trade, "filled", None)
            print(f"[Order] Filled at {fill_price} for {filled_qty} shares.")
            return True
        elif trade.orderStatus.status in ('Cancelled', 'Inactive'):
            print(f"[Order] Order status: {trade.orderStatus.status} - Canceled or Inactive.")
            return False
    print("[Order] Timeout/no fill detected.")
    return False


def place_entry_order(ib, symbol, shares, entry_px, stop_px):
    contract = Stock(symbol, 'SMART', 'USD')
    ib.qualifyContracts(contract)
    print(f"[Order] ENTRY for {symbol}: Buy {shares} @ Market, Stop {stop_px:.2f}")
    parent = Order(action="BUY",orderType="MKT",totalQuantity=shares,tif="DAY", transmit=True)
    trade = ib.placeOrder(contract, parent)
    filled = wait_for_fill(ib,trade)
    if not filled:
        print(f"[Order] Buy for {symbol} not filled, skipping stop.")
        return False, None
    stop = Order(action="SELL",orderType="STP",auxPrice=round(stop_px,2),totalQuantity=shares,tif="GTC")
    ib.placeOrder(contract, stop)
    print(f"[Order] Stop submitted at {stop_px:.2f}")
    return True, trade

class State:
    def __init__(self, fname):
        self.fname = fname
        if os.path.exists(fname):
            self.data = json.load(open(fname))
        else:
            self.data = {}
        self.data.setdefault("positions", {})
        self.data.setdefault("reentries", {})
        self.data.setdefault("max_reentries", 3)
    def save(self):
        json.dump(self.data, open(self.fname, "w"), indent=2)

def sync_positions(ib, state):
    ib_pos = {p.contract.symbol for p in ib.positions()}
    remove_syms = [sym for sym in state.data['positions'] if sym not in ib_pos]
    for sym in remove_syms:
        print(f"[SYNC] Removing {sym} from state: not found in IBKR positions.")
        del state.data['positions'][sym]

def cleanup_orphan_stops(ib):
    ib_pos = {p.contract.symbol for p in ib.positions()}
    for trade in ib.trades():
        sym = trade.contract.symbol
        if trade.order.action == 'SELL' and trade.order.orderType == 'STP' and sym not in ib_pos:
            print(f"[CLEANUP] Canceling orphan stop on {sym}")
            ib.cancelOrder(trade.order)

def log_trade(symbol, shares, entry_price, stop_price, fill_time):
    with open("trade_log.csv", "a") as f:
        f.write(f"{fill_time},{symbol},{shares},{entry_price},{stop_price}\n")

def apply_index_defense(ib, state, spy_bearish):
    if not spy_bearish:
        return
    print("[DEFENSE] SPY is bearish: tightening stops on all non-exception open positions.")
    ib_pos = {p.contract.symbol for p in ib.positions()}
    for symbol, pos in state.data["positions"].items():
        if symbol not in ib_pos or symbol in EXCEPTION_TICKERS:
            continue
        try:
            df = cached_fetch_ohlcv(ib, symbol, DEF_SWING_WINDOW + 2, '1 day')
            if df is None or len(df) < DEF_SWING_WINDOW+2: continue
            swing_low = df["low"].iloc[-DEF_SWING_WINDOW:].min()
            stop_pct = df["close"].iloc[-1] * (1 - DEFENSE_STOP_PCT)
            defensive_stop = max(swing_low, stop_pct)
            if defensive_stop > pos.get('active_stop', -np.inf):
                print(f"[DEFENSE] {symbol}: Raising stop to defensive {defensive_stop:.2f} (swing_low={swing_low:.2f}, pct={stop_pct:.2f})")
                contract = Stock(symbol, 'SMART', 'USD')
                shares = pos['shares']
                for trade in ib.trades():
                    if (trade.contract.symbol == symbol 
                        and trade.order.action == 'SELL' 
                        and trade.order.orderType == 'STP' 
                        and trade.orderStatus.status in ('Submitted','PreSubmitted')):
                        ib.cancelOrder(trade.order)
                stop = Order(action="SELL", orderType="STP", auxPrice=round(defensive_stop,2), totalQuantity=shares, tif="GTC")
                ib.placeOrder(contract, stop)
                pos['active_stop'] = defensive_stop
        except Exception as e:
            print(f"[DEFENSE] Error for {symbol}: {e}")

def update_trailing_stops(ib, state):
    print("[Stops] Updating trailing stops for open positions...")
    ib_pos = {p.contract.symbol for p in ib.positions()}
    for symbol, pos in state.data["positions"].items():
        if symbol not in ib_pos:
            print(f"[Stops] Not updating stop for {symbol}; position closed in IBKR.")
            continue
        print(f"[Stops] Checking {symbol} ...")
        try:
            df = cached_fetch_ohlcv(ib, symbol, 30, '1 day')
            shares = pos['shares']
            contract = Stock(symbol, 'SMART', 'USD')
            for trade in ib.trades():
                if (trade.contract.symbol == symbol 
                    and trade.order.action == 'SELL' 
                    and trade.order.orderType == 'STP' 
                    and trade.orderStatus.status in ('Submitted','PreSubmitted')):
                    ib.cancelOrder(trade.order)
            if symbol in EXCEPTION_TICKERS:
                atr_latest = float(atr(df, ATR_PERIOD).iloc[-1])
                high_since_entry = pos.get("high_since_entry", pos["entry_price"])
                new_high = max(high_since_entry, df["close"].iloc[-1])
                stop_price = new_high - ATR_MULTIPLIER * atr_latest
                if stop_price > pos.get('active_stop', -np.inf):
                    print(f"[Stops] ATR stop moved up for {symbol}: {stop_price:.2f}")
                    stop = Order(action="SELL", orderType="STP", auxPrice=round(stop_price,2), totalQuantity=shares, tif="GTC")
                    ib.placeOrder(contract, stop)
                    pos['active_stop'] = stop_price
                    pos['high_since_entry'] = new_high
            else:
                st, direction = supertrend(df)
                st_val = float(st.iloc[-1])
                if direction.iloc[-1] == "green" and st_val < df['close'].iloc[-1]:
                    if st_val > pos.get('active_stop', -np.inf):
                        print(f"[Stops] Supertrend stop moved up for {symbol}: {st_val:.2f}")
                        stop = Order(action="SELL", orderType="STP", auxPrice=round(st_val,2), totalQuantity=shares, tif="GTC")
                        ib.placeOrder(contract, stop)
                        pos['active_stop'] = st_val
        except Exception as e:
            print(f"[Stops] Error trailing stop for {symbol}: {e}")

def get_underlying_price(ib, underlying):
    t = ib.reqMktData(underlying, '', False, False)
    ib.sleep(2)
    px = t.last if t.last else t.close
    msg = f"Current live price: ${px:.2f}" if px else "No price"
    print(Fore.GREEN + msg + Style.RESET_ALL)
    return float(px) if px else None

def composite_score(momentum, rvol, age):
    return (0.5 * momentum) + (0.3 * rvol) - (0.2 * age)

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
    try:
        account_bal = float(nl[0])
    except Exception as e:
        print(f"ERROR: Could not convert NetLiquidation to float ({nl[0]}): {e}")
        ib.disconnect()
        return

    config = json.load(open('config.json'))
    symbols = config['user_tickers']

    main_group = set(symbols) - {"TQQQ", "SSO", "SOXL", "ARKK"}
    special_group = set(symbols) & {"TQQQ", "SSO", "SOXL", "ARKK"}
    main_alloc_total = account_bal * 0.5
    special_alloc_total = account_bal * 0.5
    main_allocated = 0.0
    special_allocated = 0.0

    print(f"\n[INFO] Main group: {sorted(main_group)}")
    print(f"[INFO] Special group: {sorted(special_group)}")

    # Track current open positions and their value
    positions = list(ib.positions())
    ib_pos_symbols = {p.contract.symbol for p in positions}
    print(f"\n[INFO] IBKR open positions: {sorted(ib_pos_symbols)}")
    for pos in positions:
        sym = pos.contract.symbol
        last = get_underlying_price(ib, pos.contract)
        print(f"[DEBUG] Position: {sym} | Qty: {pos.position} | Price: {last}")
        if last is None:
            continue
        market_val = abs(pos.position) * last
        if sym in special_group:
            special_allocated += market_val
        elif sym in main_group:
            main_allocated += market_val

    print(f"\nACCOUNT VALUE: ${account_bal:,.2f}")
    print(f"Special Ticker Group Allocation: Used ${special_allocated:,.2f} / Allowed ${special_alloc_total:,.2f} | Available ${special_alloc_total-special_allocated:,.2f}")
    print(f"Main Ticker Group Allocation:    Used ${main_allocated:,.2f} / Allowed ${main_alloc_total:,.2f} | Available ${main_alloc_total-main_allocated:,.2f}\n")

    state = State('state.json')
    sync_positions(ib, state)
    cleanup_orphan_stops(ib)
    print("[Scan] Fetching SPY daily bars...")
    spy_df = cached_fetch_ohlcv(ib, 'SPY', 120, '1 day')
    spy_df["smma9"] = smma(spy_df['close'], 9)
    spy_df["smma18"] = smma(spy_df['close'], 18)
    spy_long_allowed = spy_df["smma9"].iloc[-1] > spy_df["smma18"].iloc[-1]
    spy_bearish = not spy_long_allowed
    print(f"[Scan] SPY filter: long_allowed={spy_long_allowed}")

    qualified = []
    print("\n[INFO] --- Scanning tickers for signals ---")
    for symbol in symbols:
        print(f"[CHECK] Ticker: {symbol}")
        if symbol in state.data['positions']:
            print(f"[SKIP] {symbol}: in state file as open. Skipping.")
            continue
        if symbol in ib_pos_symbols:
            print(f"[SKIP] {symbol}: active in IBKR account. Skipping.")
            continue
        if not spy_long_allowed and symbol not in EXCEPTION_TICKERS:
            print(f"[SKIP] {symbol}: Not an exception ticker, blocked by SPY filter.")
            continue
        best = best_smma.get(symbol, {})
        found_signal = False
        # --- DAILY
        df_daily = cached_fetch_ohlcv(ib, symbol, 120, '1 day')
        if df_daily is not None and "Daily" in best:
            p = best["Daily"]
            dsigs = recent_smma_signals(df_daily, p["fast"], p["slow"], 7, label="DAILY")
            print(f"[DEBUG] {symbol}: {len(dsigs)} daily signals found.")
            if dsigs: found_signal = True
            for s in dsigs:
                score, rvol, momentum = get_momentum_rvol_score(df_daily)
                age = compute_signal_age(s["date"])
                if score is not None:
                    comp_score = composite_score(momentum, rvol, age)
                    qualified.append({
                        "symbol":symbol, "score":comp_score, "rvol":rvol,
                        "momentum":momentum, "price":s["close"], "signal_type":s["signal_type"],
                        "timeframe":s["label"], "signal_date":str(s["date"]), "age":age, "df":df_daily
                    })
        else:
            print(f"[DEBUG] {symbol}: Daily data/combo missing.")

        # --- 8-HOUR
        df_8h = cached_fetch_ohlcv(ib, symbol, 240, '8 hours')
        if df_8h is not None and "8-Hour" in best:
            p = best["8-Hour"]
            esigs = recent_smma_signals(df_8h, p["fast"], p["slow"], 14, label="8H")
            print(f"[DEBUG] {symbol}: {len(esigs)} 8h signals found.")
            if esigs: found_signal = True
            for s in esigs:
                score, rvol, momentum = get_momentum_rvol_score(df_8h)
                age = compute_signal_age(s["date"])
                if score is not None:
                    comp_score = composite_score(momentum, rvol, age)
                    qualified.append({
                        "symbol":symbol, "score":comp_score, "rvol":rvol,
                        "momentum":momentum, "price":s["close"], "signal_type":s["signal_type"],
                        "timeframe":s["label"], "signal_date":str(s["date"]), "age":age, "df":df_8h
                    })
        else:
            print(f"[DEBUG] {symbol}: 8h data/combo missing.")

        # --- 4-HOUR
        df_4h = cached_fetch_ohlcv(ib, symbol, 360, '4 hours')
        if df_4h is not None and "4-Hour" in best:
            p = best["4-Hour"]
            fsigs = recent_smma_signals(df_4h, p["fast"], p["slow"], 14, label="4H")
            print(f"[DEBUG] {symbol}: {len(fsigs)} 4h signals found.")
            if fsigs: found_signal = True
            for s in fsigs:
                score, rvol, momentum = get_momentum_rvol_score(df_4h)
                age = compute_signal_age(s["date"])
                if score is not None:
                    comp_score = composite_score(momentum, rvol, age)
                    qualified.append({
                        "symbol":symbol, "score":comp_score, "rvol":rvol,
                        "momentum":momentum, "price":s["close"], "signal_type":s["signal_type"],
                        "timeframe":s["label"], "signal_date":str(s["date"]), "age":age, "df":df_4h
                    })
        else:
            print(f"[DEBUG] {symbol}: 4h data/combo missing.")
        if not found_signal:
            print(f"[INFO] {symbol}: No SMMA cross/reentry signals found on any timeframe.")

    qualified_sorted = sorted(qualified, key=lambda x: x['score'], reverse=True)
    print("\n[Debug] All filter-qualified tickers before 10% price filter:")
    print("Rank | Symbol | Score      | Signal Px |")
    print("-----|--------|------------|-----------|")
    for z, q in enumerate(qualified_sorted, 1):
        print(f"{z:>4} | {q['symbol']:<6} | {q['score']:>10.4f} | {q['price']:>9.2f}")

    top_unique = []
    seen_symbols = set()

    for q in qualified_sorted:
        sym = q["symbol"]
        live_px = get_underlying_price(ib, Stock(sym, 'SMART', 'USD'))
        if live_px is not None and live_px > q['price'] * 1.10:
            print(Fore.RED + f"[SKIP] {sym}: Current price (${live_px:.2f}) >10% above signal price (${q['price']:.2f}), not recommending." + Style.RESET_ALL)
            continue
        if sym not in seen_symbols:
            top_unique.append({**q, "live_px": live_px})
            seen_symbols.add(sym)
            if len(top_unique) == MAX_POSITIONS:
                break

    print(f"Qualified/Actionable Candidates: {len(top_unique)} / Requested: {MAX_POSITIONS}")
    if not top_unique:
        print("[INFO] No actionable candidates found after all filters/scans.\n")
    if len(top_unique) < MAX_POSITIONS:
        print("[INFO] Fewer signals than requested max positions. Review filters/combinations for opportunities.\n")

    print("\n[Result] Top signals (SMMA cross/reentry for all timeframes, unique tickers):")
    print("Idx | Symbol  | Timeframe | Type    | Date                | Price    | RVOL    | MOM")
    print("----|---------|-----------|---------|---------------------|----------|---------|---------")
    for i, q in enumerate(top_unique):
        try:
            nice_date = to_datetime(q['signal_date']).strftime('%Y-%m-%d %H:%M')
        except Exception:
            nice_date = str(q['signal_date'])
        print(f"{i+1:>3} | {q['symbol']:<7} | {q['timeframe']:<9} | {q['signal_type']:<7} | "
              f"{nice_date:<19} | {q['price']:<8.2f} | {q['rvol']:<6.2f} | {q['momentum']:<6.3f}")

    for q in top_unique:
        symbol = q['symbol']
        if symbol in state.data['positions'] or symbol in ib_pos_symbols:
            print(f"[Order] {symbol}: active, skipping order placement.")
            continue
        if symbol in special_group:
            avail_cap = special_alloc_total - special_allocated
            group = "special"
        else:
            avail_cap = main_alloc_total - main_allocated
            group = "main"
        size = get_trade_size(avail_cap, q['price'], MAX_ALLOC)
        if size > 0:
            alloc_amount = size * q["price"]
            if group == "special":
                special_allocated += alloc_amount
            else:
                main_allocated += alloc_amount
        if size == 0:
            print(f"[Order] {symbol}: trade skipped, exceeds 15% allocation rule or group cap.")
            continue
        df = q["df"]
        alloc_pct = size * q["price"] / account_bal * 100
        prompt = f"\n[Prompt] {symbol}: Buy {size} at ${q['price']:.2f} ({alloc_pct:.1f}% capital, stop after fill)?. Confirm (y/n): "
        user_input = input(prompt).strip().lower()
        if user_input != 'y':
            print(f"[Order] {symbol}: user declined trade.")
            continue
        print(f"[Order] Confirmed for {symbol}: placing order for {size} shares at ${q['price']:.2f}.")
        if symbol in EXCEPTION_TICKERS:
            atr_val = float(atr(df, ATR_PERIOD).iloc[-1])
            high_since_entry = q['price']
            stop_px = high_since_entry - ATR_MULTIPLIER * atr_val
            print(f"[Order] ATR stop calculated at {stop_px:.2f}")
        else:
            st, direction = supertrend(df)
            stop_px = float(st.iloc[-1])
            print(f"[Order] Supertrend stop calculated at {stop_px:.2f}")
        success, trade = place_entry_order(ib, symbol, size, q['price'], stop_px)
        if not success:
            print(f"[Order] {symbol}: order/stop failed or not filled.")
            continue
        fill_time = trade.log[-1].time.strftime('%Y-%m-%d %H:%M:%S') if trade.log else "Unknown"
        log_trade(symbol, size, q['price'], stop_px, fill_time)
        state.data['positions'][symbol] = {
            "entry_price": q['price'],
            "shares": size,
            "active_stop": stop_px,
            "stop_type": "atr" if symbol in EXCEPTION_TICKERS else "supertrend",
            "high_since_entry": q['price'] if symbol in EXCEPTION_TICKERS else None,
            "signal_timeframe": q['timeframe'],
            "signal_type": q['signal_type'],
            "signal_date": q['signal_date'],
            "closed": False
        }

    print("[INFO] Applying index defense & updating trailing stops...")
    apply_index_defense(ib, state, spy_bearish)
    update_trailing_stops(ib, state)
    state.save()
    ib.disconnect()
    print("[INFO] Disconnected from IB.")

if __name__ == "__main__":
    main()


