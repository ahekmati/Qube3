# =============================================================================
# PROFESSIONAL OPTIONS TRADING SCRIPT FOR IBKR WITH SWING HIGH STRIKE TARGETS
# -- Robust: Multi-strike and multi-expiry fallback logic
# =============================================================================

import numpy as np
import pandas as pd
import math
from datetime import datetime
from ib_insync import *

ATR_PERIOD = 10
MAX_ALLOC = 0.10
TARGET_RR = 2
STOP_RR = 1
TRAILING_PCT = 0.20
MIN_OTM = 0.90

def smma(series, window):
    s = pd.Series(series)
    if len(s) < window: return s.copy()
    return s.ewm(alpha=1/window, adjust=False).mean()

def fetch_bars(ib, ticker, duration_str, bar_size, exchange):
    contract = Stock(ticker, exchange, 'USD')
    print(f"[Fetch] {ticker} ({exchange}): {bar_size} bars, {duration_str} window ...")
    try:
        bars = ib.reqHistoricalData(
            contract, endDateTime='', durationStr=duration_str,
            barSizeSetting=bar_size, whatToShow='TRADES', useRTH=False
        )
        if bars:
            print(f"[Success] Fetched {len(bars)} bars for {ticker} [{exchange}] [{bar_size}]")
        return bars
    except Exception as e:
        print(f"[Error] {ticker} fetch_bars: {e}")
        return None

def get_underlying_price(ib, underlying):
    print(f"[Market] Getting market price for {underlying.symbol} ...")
    t = ib.reqMktData(underlying, '', False, False)
    ib.sleep(2)
    px = t.last if t.last else t.close
    print(f"[Market] {underlying.symbol} price: {px}")
    return float(px) if px else None

def find_filtered_strikes(strikes, swing_high, current_px):
    min_strike = max(current_px * MIN_OTM, swing_high)
    filtered = [s for s in strikes if s >= min_strike]
    print(f"[Contract] Filtering strikes >= {min_strike:.2f}, n={len(filtered)}")
    return sorted(filtered)

def select_option_contract(ib, underlying, expiry_after_days, swing_high, price_limit):
    print(f"[Options] Getting chains for {underlying.symbol} ...")
    chains = ib.reqSecDefOptParams(underlying.symbol, '', underlying.secType, underlying.conId)
    chain = next((c for c in chains if c.tradingClass == underlying.symbol or c.exchange in ['SMART', 'CBOE']), None)
    if not chain or not chain.expirations:
        print(f"[Options] No option chain/expirations for {underlying.symbol}")
        return None, None, None, None
    today = datetime.now().date()
    expiry_dates = sorted([datetime.strptime(d, "%Y%m%d").date() for d in chain.expirations])
    # Find both the nearest and 2nd-nearest expiry
    expiry_dates_sorted = sorted(expiry_dates, key=lambda d: abs((d - today).days - expiry_after_days))
    try_expiries = expiry_dates_sorted[:2] if len(expiry_dates_sorted) > 1 else expiry_dates_sorted
    strikes = sorted(chain.strikes)
    underlying_px = get_underlying_price(ib, underlying)
    if not underlying_px:
        print("[Options] No underlying price.")
        return None, None, None, None
    filtered_strikes = find_filtered_strikes(strikes, swing_high, underlying_px)
    for expiry in try_expiries:
        print(f"[Options] Trying expiry: {expiry}")
        for strike in filtered_strikes:
            print(f"[Options] Attempting strike: {strike} on expiry: {expiry}")
            contract = Option(underlying.symbol, expiry.strftime("%Y%m%d"), strike, 'C', 'SMART')
            try:
                ib.qualifyContracts(contract)
                ticker = ib.reqMktData(contract, '', False, False)
                ib.sleep(2)
                price = (ticker.bid + ticker.ask) / 2 if ticker.bid and ticker.ask else None
                if price is None or (isinstance(price, float) and math.isnan(price)):
                    print(f"[Options] {contract}: No valid midpoint.")
                    continue
                cost = price * 100
                print(f"[Options] Midpoint: {price:.2f}, contract cost: {cost:.2f}")
                if cost > price_limit:
                    print(f"[Skip] Contract cost ${cost:.2f} > 10% allocation (${price_limit:.2f})")
                    continue
                iv = getattr(ticker.modelGreeks, 'impliedVol', None)
                iv = float(iv) if iv else 0.4
                print(f"[Options] IV: {iv:.2f}")
                print(f"[Select] Using expiry {expiry} strike {strike}.")
                return contract, price, cost, iv
            except Exception as e:
                print(f"[Options] {contract}: could not qualify/price ({e}).")
                continue
        print(f"[Options] No valid contracts for expiry {expiry}.")
    print(f"[Options] No valid strikes in allowed expiries for {underlying.symbol}.")
    return None, None, None, None

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
    oca_group = f'OCO_{contract.symbol}_{contract.strike}_{datetime.now().strftime('%H%M%S')}'
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

    tickers = [
           "IBIT"
    ]
    exchanges = ['ARCA', 'NASDAQ', 'SMART']
    lookback = '180 D'
    filled_trades = []

    for ticker in tickers:
        print(f"\n[==== Scanning {ticker} ====")
        # DAILY 9/18
        bars_daily = None
        for exch in exchanges:
            bars_daily = fetch_bars(ib, ticker, lookback, "1 day", exch)
            if bars_daily: break
        if bars_daily:
            print(f"[Daily Analysis] {ticker}: Analyzing daily bars ...")
            df_daily = util.df(bars_daily)
            df_daily['date'] = pd.to_datetime(df_daily['date'])
            df_daily = df_daily.set_index('date')
            crosses = analyze_crosses_and_swing_highs(df_daily, 9, 18)
            for ix, event in enumerate(crosses):
                when = event['date']
                if (datetime.now().date() - when.date()) > pd.Timedelta(days=10):
                    continue
                swing_high = event['swing_high_prev']
                print(f"\n[Signal] {ticker} DAILY 9/18 crossover at {when.date()}, prior swing high {swing_high:.2f}")
                underlying = Stock(ticker, 'SMART', 'USD')
                ib.qualifyContracts(underlying)
                contract, entry_px, cost, iv = select_option_contract(ib, underlying, 45, swing_high, max_alloc_val)
                if not all([contract, entry_px, cost, iv]):
                    print(f"[Skip] No trade for {ticker} daily (invalid contract or price).")
                    continue
                size = int(max_alloc_val // (entry_px*100))
                if size < 1:
                    print(f"[Alloc] Position too small, skipping.")
                    continue
                if size > 10: size = 10
                profit_tgt, stop_loss, _ = calc_risk_targets(entry_px, iv)
                allocation_pct = round(size * entry_px * 100 / account_bal * 100, 2)
                print("\n[Order Preview]")
                print(f" Ticker: {ticker}")
                print(f" Contract: {contract.symbol} {contract.lastTradeDateOrContractMonth} {contract.strike}C")
                print(f" Quantity: {size}")
                print(f" Premium per contract: {entry_px:.2f}")
                print(f" Option IV: {iv:.2f}")
                print(f" Total allocation: {size * entry_px * 100:.2f} USD ({allocation_pct}%)")
                print(f" Dynamic profit target: {profit_tgt:.2f}")
                print(f" Dynamic stop loss: {stop_loss:.2f}")
                print(f" Timeframe: DAILY 9/18 | Signal date: {when.date()} | Swing high strike: {swing_high:.2f}")
                answer = input("Place this trade? (y/n): ").strip().lower()
                if answer != 'y':
                    print(f"[Input] Trade declined.")
                    continue
                place_entry_with_dynamic_exits(ib, contract, entry_px, size, profit_tgt, stop_loss)
                filled_trades.append({
                    "symbol": ticker,
                    "contract": f"{contract.symbol} {contract.lastTradeDateOrContractMonth} {contract.strike}C",
                    "qty": size,
                    "premium": entry_px,
                    "iv": iv,
                    "target": profit_tgt,
                    "stop": stop_loss,
                    "alloc_pct": allocation_pct,
                    "timeframe": "DAILY 9/18",
                    "signal_date": when.strftime('%Y-%m-%d'),
                    "entry_time": datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                })
            print(f"[Daily Analysis] {ticker} finished.")

        # 4H 26/150
        bars_4h = None
        for exch in exchanges:
            bars_4h = fetch_bars(ib, ticker, lookback, "4 hours", exch)
            if bars_4h: break
        if bars_4h:
            print(f"[4H Analysis] {ticker}: Analyzing 4H bars ...")
            df_4h = util.df(bars_4h)
            df_4h['date'] = pd.to_datetime(df_4h['date'])
            df_4h = df_4h.set_index('date')
            crosses = analyze_crosses_and_swing_highs(df_4h, 26, 150)
            for ix, event in enumerate(crosses):
                when = event['date']
                if (datetime.now() - when.tz_localize(None)) > pd.Timedelta(days=10):
                    continue
                swing_high = event['swing_high_prev']
                print(f"\n[Signal] {ticker} 4H 26/150 crossover at {when.date()}, prior swing high {swing_high:.2f}")
                underlying = Stock(ticker, 'SMART', 'USD')
                ib.qualifyContracts(underlying)
                contract, entry_px, cost, iv = select_option_contract(ib, underlying, 30, swing_high, max_alloc_val)
                if not all([contract, entry_px, cost, iv]):
                    print(f"[Skip] No trade for {ticker} 4H (invalid contract or price).")
                    continue
                size = int(max_alloc_val // (entry_px*100))
                if size < 1:
                    print(f"[Alloc] Position too small, skipping.")
                    continue
                if size > 10: size = 10
                profit_tgt, stop_loss, _ = calc_risk_targets(entry_px, iv)
                allocation_pct = round(size * entry_px * 100 / account_bal * 100, 2)
                print("\n[Order Preview]")
                print(f" Ticker: {ticker}")
                print(f" Contract: {contract.symbol} {contract.lastTradeDateOrContractMonth} {contract.strike}C")
                print(f" Quantity: {size}")
                print(f" Premium per contract: {entry_px:.2f}")
                print(f" Option IV: {iv:.2f}")
                print(f" Total allocation: {size * entry_px * 100:.2f} USD ({allocation_pct}%)")
                print(f" Dynamic profit target: {profit_tgt:.2f}")
                print(f" Dynamic stop loss: {stop_loss:.2f}")
                print(f" Timeframe: 4H 26/150 | Signal date: {when.date()} | Swing high strike: {swing_high:.2f}")
                answer = input("Place this trade? (y/n): ").strip().lower()
                if answer != 'y':
                    print(f"[Input] Trade declined.")
                    continue
                place_entry_with_dynamic_exits(ib, contract, entry_px, size, profit_tgt, stop_loss)
                filled_trades.append({
                    "symbol": ticker,
                    "contract": f"{contract.symbol} {contract.lastTradeDateOrContractMonth} {contract.strike}C",
                    "qty": size,
                    "premium": entry_px,
                    "iv": iv,
                    "target": profit_tgt,
                    "stop": stop_loss,
                    "alloc_pct": allocation_pct,
                    "timeframe": "4H 26/150",
                    "signal_date": when.strftime('%Y-%m-%d'),
                    "entry_time": datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                })
            print(f"[4H Analysis] {ticker} finished.")

    ib.disconnect()
    print("\n[Result] Script finished scanning all tickers.")

    # Summary Table of Filled Trades
    if filled_trades:
        print("\n==== SUMMARY OF TRADES EXECUTED ====")
        print("Idx | Symbol | Contract           | Qty | EntryPx | IV    | Target  | Stop    | Alloc% | Timeframe   | SignalDate | EntryTime")
        print("----|--------|--------------------|-----|---------|-------|---------|---------|--------|-------------|------------|-------------------")
        for i, t in enumerate(filled_trades, 1):
            print(f"{i:<3} | {t['symbol']:<6} | {t['contract']:<18} | {t['qty']:<3} | {t['premium']:<7.2f} | {t['iv']:<5.2f} | {t['target']:<7.2f} | {t['stop']:<7.2f} | {t['alloc_pct']:<6.2f} | {t['timeframe']:<11} | {t['signal_date']:<10} | {t['entry_time']}")
    else:
        print("\n[Result] No trades were filled in this session.")

if __name__ == "__main__":
    main()
