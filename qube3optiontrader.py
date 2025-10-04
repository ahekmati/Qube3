# =============================================================================
# PROFESSIONAL OPTIONS TRADING SCRIPT FOR IBKR WITH SWING HIGH STRIKE TARGETS
# =============================================================================

import os
import time
import pickle
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import math
from ib_insync import *

ATR_PERIOD = 10
MAX_ALLOC = 0.10
TARGET_RR = 3
STOP_RR = 1
TRAILING_PCT = 0.20
MIN_OTM = 0.90

OPTION_EXCHANGES = ['SMART', 'CBOE', 'BOX', 'ARCA']
MAX_STRIKES_PER_EXPIRY = 5
CACHE_DIR = "ibkr_cache"
CACHE_EXPIRE_HOURS = 20

PRIMARY_EXCHANGE_MAP = {
    'AAPL': 'NASDAQ', 'MSFT': 'NASDAQ', 'GOOG': 'NASDAQ', 'GOOGL': 'NASDAQ', 'SPY': 'ARCA', 'TSLA': 'NASDAQ',
}
DEFAULT_PRIMARY = 'NASDAQ'

def should_skip_non_stock_symbol(symbol):
    return len(symbol) == 5 and (symbol.endswith('W') or symbol.endswith('T'))

def smma(series, window):
    s = pd.Series(series)
    if len(s) < window: return s.copy()
    return s.ewm(alpha=1/window, adjust=False).mean()

def make_stock_contract(ticker, exchange, currency='USD'):
    contract = Stock(ticker, exchange, currency)
    if exchange == 'SMART':
        contract.primaryExchange = PRIMARY_EXCHANGE_MAP.get(ticker, DEFAULT_PRIMARY)
    return contract

def get_cache_path(cache_type, ticker, freq):
    if not os.path.exists(CACHE_DIR):
        os.makedirs(CACHE_DIR)
    return os.path.join(CACHE_DIR, f"{cache_type}_{ticker}_{freq}.pkl")

def load_cache(cache_type, ticker, freq):
    pkl = get_cache_path(cache_type, ticker, freq)
    if os.path.exists(pkl) and (time.time() - os.path.getmtime(pkl) < CACHE_EXPIRE_HOURS * 3600):
        try:
            with open(pkl, 'rb') as f:
                return pickle.load(f)
        except Exception as e:
            print(f"[Cache] Failed to read {pkl}: {e}")
    return None

def save_cache(cache_type, ticker, freq, data):
    try:
        pkl = get_cache_path(cache_type, ticker, freq)
        with open(pkl, 'wb') as f:
            pickle.dump(data, f)
    except Exception as e:
        print(f"[Cache] Failed to write {pkl}: {e}")

def fetch_bars(ib, ticker, duration_str, bar_size, exchange):
    if should_skip_non_stock_symbol(ticker):
        print(f"[Skip] {ticker}: Pattern matches warrant/right (likely non-stock), skipping.")
        return None
    freq = bar_size.replace(' ', '')
    cached = load_cache('bars', ticker, freq)
    if cached is not None:
        print(f"[Cache] Using cached bars for {ticker} freq={freq}")
        return cached
    contract = make_stock_contract(ticker, exchange, 'USD')
    contracts = ib.qualifyContracts(contract)
    if not contracts:
        print(f"[Skip] Contract qualification failed for {ticker} ({exchange}), skipping.")
        return None
    contract = contracts[0]
    print(f"[Fetch] {ticker} ({exchange}): {bar_size} bars, {duration_str} window ...")
    try:
        bars = ib.reqHistoricalData(contract, endDateTime='', durationStr=duration_str,
            barSizeSetting=bar_size, whatToShow='TRADES', useRTH=False)
        if bars:
            print(f"[Success] Fetched {len(bars)} bars for {ticker} [{exchange}] [{bar_size}]")
            save_cache('bars', ticker, freq, bars)
        return bars
    except Exception as e:
        print(f"[Error] {ticker} fetch_bars: {e}")
        return None

def get_underlying_price(ib, underlying):
    contracts = ib.qualifyContracts(underlying)
    if not contracts:
        print(f"[Skip] Underlying qualification failed for {underlying.symbol} ({underlying.exchange}).")
        return None
    underlying = contracts[0]
    print(f"[Market] Getting market price for {underlying.symbol} ...")
    t = ib.reqMktData(underlying, '', False, False)
    ib.sleep(2)
    px = t.last if t.last else t.close
    print(f"[Market] {underlying.symbol} price: {px}")
    return float(px) if px else None

def load_option_chain_cache(ticker):
    return load_cache('chain', ticker, 'opt')

def save_option_chain_cache(ticker, chains):
    save_cache('chain', ticker, 'opt', chains)

def select_option_contract_multi_exch(ib, underlying, expiry_after_days, swing_high, price_limit):
    tries = 0
    chains = load_option_chain_cache(underlying.symbol)
    if chains is None:
        try:
            chains = ib.reqSecDefOptParams(underlying.symbol, '', underlying.secType, underlying.conId)
            save_option_chain_cache(underlying.symbol, chains)
        except Exception as e:
            print(f"[Options] Failed to fetch option chains: {e}")
            return None, None, None, None, None, None, None
    for exch in OPTION_EXCHANGES:
        tries += 1
        chain = None
        for c in chains:
            if exch in c.exchange:
                chain = c
                break
        if chain is None:
            for c in chains:
                if c.tradingClass and underlying.symbol in c.tradingClass and c.strikes and c.expirations:
                    chain = c
                    break
        if chain is None:
            for c in chains:
                if c.strikes and c.expirations:
                    chain = c
                    break
        if not chain or not chain.expirations or not chain.strikes:
            print(f"[Options] No valid chain/expirations/strikes for {underlying.symbol} on {exch}")
            continue
        today = datetime.now().date()
        expiry_dates = sorted([datetime.strptime(d, "%Y%m%d").date() for d in chain.expirations])
        expiry_dates_sorted = sorted(expiry_dates, key=lambda d: abs((d - today).days - expiry_after_days))
        try_expiries = expiry_dates_sorted[:2] if len(expiry_dates_sorted) > 1 else expiry_dates_sorted
        strikes = sorted(chain.strikes)
        underlying_px = get_underlying_price(ib, underlying)
        if not underlying_px:
            print("[Options] No underlying price.")
            continue
        print(f"[Options] All strikes: {strikes}")

        min_strike = underlying_px * 0.85
        strike_candidates = [s for s in strikes if s >= min_strike]
        use_swing = swing_high <= underlying_px * 1.10
        candidates = []

        if use_swing:
            idx = np.searchsorted(strike_candidates, swing_high)
            idxs = [idx-2, idx-1, idx, idx+1]
            idxs = [i for i in idxs if 0 <= i < len(strike_candidates)]
            candidates = [strike_candidates[i] for i in idxs]
            print(f"[Contract] Using swing high anchor ({swing_high:.2f}), candidates: {candidates}")
        else:
            print(f"[Contract] Swing high {swing_high:.2f} is not within 10% of spot {underlying_px:.2f}; skipping to fallback.")

        if not candidates:
            idx_cur = np.searchsorted(strike_candidates, underlying_px)
            fallback_idxs = [idx_cur+1, idx_cur+2]
            fallback_idxs = [i for i in fallback_idxs if 0 <= i < len(strike_candidates)]
            candidates = [strike_candidates[i] for i in fallback_idxs]
            print(f"[Contract] Fallback to 2 strikes above spot {underlying_px:.2f}: {candidates}")

        for expiry in try_expiries:
            print(f"[Options] Trying expiry: {expiry}")
            for strike in candidates:
                print(f"[Options] Attempting strike: {strike} on expiry: {expiry} at {chain.exchange}")
                contract = Option(underlying.symbol, expiry.strftime("%Y%m%d"), strike, 'C', chain.exchange)
                contracts = ib.qualifyContracts(contract)
                if not contracts:
                    print(f"[Contract Error] Qualification failed for {contract.symbol} {contract.lastTradeDateOrContractMonth} {strike} {chain.exchange}, skipping.")
                    continue
                contract = contracts[0]
                try:
                    ticker = ib.reqMktData(contract, '', False, False)
                    ib.sleep(2)
                    price = (ticker.bid + ticker.ask) / 2 if ticker.bid and ticker.ask else None
                    if price is None or (isinstance(price, float) and math.isnan(price)):
                        continue
                    cost = price * 100
                    print(f"[Options] Midpoint: {price:.2f}, contract cost: {cost:.2f}")
                    if cost > price_limit:
                        continue
                    iv = getattr(ticker.modelGreeks, 'impliedVol', None)
                    iv = float(iv) if iv else 0.4
                    bid, ask = ticker.bid, ticker.ask
                    spread = (ask - bid) / price if (bid and ask and price) else 0.10
                    print(f"[Options] IV: {iv:.2f}")
                    print(f"[Select] Using expiry {expiry} strike {strike} at {chain.exchange}.")
                    return contract, price, cost, iv, spread, strike, underlying_px
                except Exception as e:
                    continue
            print(f"[Options] No valid contracts for expiry {expiry} at {chain.exchange}.")
        if tries >= 3:
            print(f"[Options] Skipping {underlying.symbol} after {tries} exchange attempts.")
            break
    print(f"[Options] No valid options contract found for {underlying.symbol} after checking multiple exchanges.")
    return None, None, None, None, None, None, None

def analyze_crosses_and_swing_highs(df, fast, slow):
    df['smma_fast'] = smma(df['close'], fast)
    df['smma_slow'] = smma(df['close'], slow)
    prev_bull_idx = None
    cross_events = []
    for i in range(1, len(df)):
        if df['smma_fast'].iloc[i] > df['smma_slow'].iloc[i] and df['smma_fast'].iloc[i-1] <= df['smma_slow'].iloc[i-1]:
            if prev_bull_idx is not None:
                swing_high = df['high'].iloc[prev_bull_idx+1:i].max()
            else:
                swing_high = df['high'].iloc[:i].max()
            cross_events.append({
                'idx': i,
                'date': df.index[i],
                'swing_high_prev': swing_high
            })
            prev_bull_idx = i
    print(f"[Analysis] Found {len(cross_events)} crosses (fast:{fast}, slow:{slow})")
    if len(cross_events) > 0:
        print(f"[DEBUG] First crosses: {cross_events[:2]}")
    return cross_events

def calc_risk_targets(entry, iv, rr_target=TARGET_RR, rr_stop=STOP_RR):
    base_risk = entry * 0.3
    risk = base_risk * (1 + (iv-0.25)*1.5)
    profit = entry + (risk * rr_target)
    stop = entry - (risk * rr_stop)
    print(f"[Risk] Entry:{entry:.2f} IV:{iv:.2f} target:{profit:.2f} stop:{stop:.2f}")
    return round(profit,2), round(stop,2), round(risk,2)

def place_entry_with_dynamic_exits(ib, contract, entry_price, size, profit_tgt, stop_loss, trailing_pct=TRAILING_PCT):
    print(f"[Order] Placing BUY market order: {contract.symbol} {contract.lastTradeDateOrContractMonth} strike {contract.strike}, qty {size}")
    buy = MarketOrder('BUY', size)
    trade = ib.placeOrder(contract, buy)
    while not trade.isDone(): ib.waitOnUpdate()
    print("[Order] Buy filled.")
    oca_group = f'OCO_{contract.symbol}_{contract.strike}_{datetime.now().strftime("%H%M%S")}'
    if size < 2:
        tgt = LimitOrder('SELL', size, profit_tgt, ocaGroup=oca_group, ocaType=1)
        stp = StopOrder('SELL', size, stop_loss, ocaGroup=oca_group, ocaType=1)
        print(f"[OCO] Sending (1-contract): limit target={profit_tgt}, stop={stop_loss}")
        ib.placeOrder(contract, tgt)
        ib.placeOrder(contract, stp)
    else:
        half = size // 2
        tgt = LimitOrder('SELL', half, profit_tgt, ocaGroup=oca_group, ocaType=1)
        stp = StopOrder('SELL', half, stop_loss, ocaGroup=oca_group, ocaType=1)
        ib.placeOrder(contract, tgt)
        ib.placeOrder(contract, stp)
        print(f"[OCO] Scaling: half at target ({profit_tgt}) / stop ({stop_loss}), will trail rest")
        print(f"[ALERT] After partial profit, manually trail remaining contracts at {int(trailing_pct*100)}% below option high.")

def score_signal(signal):
    rr_score = min(signal['rr'], 4) / 4 * 10
    iv_opt = 0.4
    iv_score = max(0, 10 - abs(signal['iv'] - iv_opt) * 25)
    spread_score = max(0, 10 - signal['spread'] * 30)
    otm_score = max(0, 10 - (signal['otm_pct']-1)*40)
    strength_score = signal.get('rel_strength', 5)
    return (rr_score + iv_score + spread_score + otm_score + strength_score) / 5

def main():
    print("[System] Connecting to Interactive Brokers TWS/Gateway ...")
    IBKR_HOST, IBKR_PORT, IBKR_CID = '127.0.0.1', 4001, 101
    ib = IB()
    ib.connect(IBKR_HOST, IBKR_PORT, IBKR_CID)
    print("[Account] Fetching account summary ...")
    summary = ib.accountSummary()
    netliq_obj = next((x for x in summary if x.tag=='NetLiquidation'), None)
    if not netliq_obj:
        print("[Error] Could not retrieve account value. Exiting.")
        ib.disconnect()
        return
    account_bal = float(netliq_obj.value)
    print(f"[Account] Net Liquidation: {account_bal:.2f}")
    max_alloc_val = account_bal * MAX_ALLOC

    tickers = ["AAPL", "NVDA", "TSLA", "AMZN", "MSFT", "TQQQ", "QQQ", "SPY", "AMD", "GOOG", "GOOGL"]
    exchanges = ['ARCA', 'NASDAQ', 'SMART']
    lookback = '180 D'
    all_signals = []

    for ticker in tickers:
        print(f"\n[==== Scanning {ticker} ====")
        # DAILY 9/18
        bars_daily = None
        for exch in exchanges:
            bars_daily = fetch_bars(ib, ticker, lookback, "1 day", exch)
            if bars_daily: break
        if bars_daily is not None:
            df_daily = util.df(bars_daily)
            print(f"[DEBUG] {ticker} daily bars head:\n{df_daily.head()}")
            print(f"[DEBUG] {ticker} daily bars columns: {df_daily.columns}")
            df_daily['date'] = pd.to_datetime(df_daily['date'])
            df_daily = df_daily.set_index('date')
            crosses = analyze_crosses_and_swing_highs(df_daily, 9, 18)
            print(f"[DEBUG] {ticker}: Found {len(crosses)} daily crosses.")
            for ix, event in enumerate(crosses):
                when = event['date']
                if (datetime.now().date() - when.date()) > pd.Timedelta(days=60):
                    continue
                print(f"[DEBUG] {ticker}: Survived date filter for event at {when}")
                swing_high = event['swing_high_prev']
                print(f"\n[Signal] {ticker} DAILY 9/18 crossover at {when.date()}, prior swing high {swing_high:.2f}")
                underlying = make_stock_contract(ticker, 'SMART', 'USD')
                res = select_option_contract_multi_exch(ib, underlying, 45, swing_high, max_alloc_val)
                if not all(res):
                    print(f"[DEBUG] {ticker} {when}: No valid option contract after signal detection")
                    continue
                contract, entry_px, cost, iv, spread, strike, underlying_px = res
                size = int(max_alloc_val // (entry_px * 100))
                if size < 1 or size > 10:
                    print(f"[DEBUG] {ticker} {when}: Option position size filter failed (size={size})")
                    continue
                profit_tgt, stop_loss, _ = calc_risk_targets(entry_px, iv)
                print(f"[DEBUG] Appending signal for {ticker} on {when}")
                all_signals.append({
                    'symbol': ticker,
                    'contract': f"{contract.symbol} {contract.lastTradeDateOrContractMonth} {contract.strike}C",
                    'rr': profit_tgt / abs(stop_loss) if stop_loss != 0 else 0,
                    'iv': iv,
                    'spread': spread,
                    'otm_pct': strike / underlying_px if underlying_px else float('nan'),
                    'rel_strength': 5,
                    'premium': entry_px,
                    'qty': size,
                    'timeframe': 'DAILY 9/18',
                    'signal_date': when.strftime('%Y-%m-%d'),
                    'score': 0
                })

        # 4H 26/150
        bars_4h = None
        for exch in exchanges:
            bars_4h = fetch_bars(ib, ticker, lookback, "4 hours", exch)
            if bars_4h: break
        if bars_4h is not None:
            df_4h = util.df(bars_4h)
            print(f"[DEBUG] {ticker} 4H bars head:\n{df_4h.head()}")
            print(f"[DEBUG] {ticker} 4H bars columns: {df_4h.columns}")
            df_4h['date'] = pd.to_datetime(df_4h['date'])
            df_4h = df_4h.set_index('date')
            crosses = analyze_crosses_and_swing_highs(df_4h, 26, 150)
            print(f"[DEBUG] {ticker}: Found {len(crosses)} 4H crosses.")
            for ix, event in enumerate(crosses):
                when = event['date']
                if (datetime.now() - when.tz_localize(None)) > pd.Timedelta(days=60):
                    continue
                print(f"[DEBUG] {ticker}: Survived date filter for event at {when}")
                swing_high = event['swing_high_prev']
                print(f"\n[Signal] {ticker} 4H 26/150 crossover at {when.date()}, prior swing high {swing_high:.2f}")
                underlying = make_stock_contract(ticker, 'SMART', 'USD')
                res = select_option_contract_multi_exch(ib, underlying, 30, swing_high, max_alloc_val)
                if not all(res):
                    print(f"[DEBUG] {ticker} {when}: No valid option contract after signal detection")
                    continue
                contract, entry_px, cost, iv, spread, strike, underlying_px = res
                size = int(max_alloc_val // (entry_px * 100))
                if size < 1 or size > 10:
                    print(f"[DEBUG] {ticker} {when}: Option position size filter failed (size={size})")
                    continue
                profit_tgt, stop_loss, _ = calc_risk_targets(entry_px, iv)
                print(f"[DEBUG] Appending signal for {ticker} on {when}")
                all_signals.append({
                    'symbol': ticker,
                    'contract': f"{contract.symbol} {contract.lastTradeDateOrContractMonth} {contract.strike}C",
                    'rr': profit_tgt / abs(stop_loss) if stop_loss != 0 else 0,
                    'iv': iv,
                    'spread': spread,
                    'otm_pct': strike / underlying_px if underlying_px else float('nan'),
                    'rel_strength': 5,
                    'premium': entry_px,
                    'qty': size,
                    'timeframe': '4H 26/150',
                    'signal_date': when.strftime('%Y-%m-%d'),
                    'score': 0
                })

    print(f"[DEBUG] Total signals detected and ready to rank: {len(all_signals)}")
    for sig in all_signals:
        sig['score'] = score_signal(sig)
    top_signals = sorted(all_signals, key=lambda x: x['score'], reverse=True)[:3]
    GREEN = "\033[92m"
    END = "\033[0m"

    print(GREEN + "\n=== TOP 3 RANKED SIGNALS ===" + END)
    for i, sig in enumerate(top_signals, 1):
        print(GREEN + f"{i}. {sig['symbol']} | Score: {sig['score']:.2f} | RR: {sig['rr']:.2f} | IV: {sig['iv']:.2f} | Spread: {sig['spread']:.3f} | OTM%: {sig['otm_pct']:.2f}")
        print(f"    Contract: {sig['contract']}, Qty: {sig['qty']}, Signal Date: {sig['signal_date']}, Timeframe: {sig['timeframe']}" + END)

    ib.disconnect()
    print("\n[Result] Script finished scanning all tickers.")

if __name__ == "__main__":
    main()
