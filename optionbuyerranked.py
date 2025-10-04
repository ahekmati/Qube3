# =============================================================================
# QUBE3 OPTIONS TRADING SCRIPT FOR IBKR WITH SWING HIGH STRIKE TARGETS
# Now with Rapid Option Ranking for Top 5 Signals
# =============================================================================

import numpy as np
import pandas as pd
import math
from datetime import datetime, timedelta
from ib_insync import *

ATR_PERIOD = 10
MAX_ALLOC = 0.10
TARGET_RR = 2
STOP_RR = 1
TRAILING_PCT = 0.20
MIN_OTM = 0.90
TARGET_IV = 0.40          # Ideal IV for swing trades (tunable)
MAX_SHOW_TRADES = 5       # Show top N ranked options

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

def get_liquidity_score(bid, ask, open_interest):
    # Tight spread, high open interest => high score (0-1 normalized)
    if (bid is None) or (ask is None): return 0
    spread = ask - bid
    spread_score = 1 - min(spread / max(bid, 0.01), 0.5)  # 0.5 is a reasonable max spread
    oi_score = min(open_interest / 100, 1)                # Normalize assuming 100+ is optimal
    return round(0.7 * spread_score + 0.3 * oi_score, 2)

def get_freshness_score(signal_date):
    # Signals from today = 1, yesterday = 0.9, ... max 10 days ago = 0.7
    delta = (datetime.now().date() - signal_date.date()).days
    return max(1.0 - delta * 0.03, 0.7)

def get_cost_score(cost, max_alloc_val):
    # Cheaper contracts score higher, normalized to allocation limit
    return max(1.0 - (cost / max_alloc_val), 0)

def get_iv_quality_score(iv, target_iv=TARGET_IV):
    # Score closer IVs higher, penalize extreme IV
    return max(1.0 - abs(iv - target_iv) / target_iv, 0)

def select_option_contracts_and_rank(
    ib, underlying, expiry_after_days, swing_high, price_limit, signal_date):
    chains = ib.reqSecDefOptParams(underlying.symbol, '', underlying.secType, underlying.conId)
    chain = next((c for c in chains if c.tradingClass == underlying.symbol or c.exchange in ['SMART', 'CBOE']), None)
    if not chain or not chain.expirations:
        print(f"[Options] No option chain/expirations for {underlying.symbol}")
        return []
    today = datetime.now().date()
    expiry_dates = sorted([datetime.strptime(d, "%Y%m%d").date() for d in chain.expirations])
    expiry_dates_sorted = sorted(expiry_dates, key=lambda d: abs((d - today).days - expiry_after_days))
    try_expiries = expiry_dates_sorted[:2] if len(expiry_dates_sorted) > 1 else expiry_dates_sorted
    strikes = sorted(chain.strikes)
    underlying_px = get_underlying_price(ib, underlying)
    if not underlying_px:
        print("[Options] No underlying price.")
        return []
    filtered_strikes = find_filtered_strikes(strikes, swing_high, underlying_px)
    option_candidates = []
    for expiry in try_expiries:
        for strike in filtered_strikes:
            contract = Option(underlying.symbol, expiry.strftime("%Y%m%d"), strike, 'C', 'SMART')
            try:
                ib.qualifyContracts(contract)
                ticker = ib.reqMktData(contract, '', False, False)
                ib.sleep(2)
                price = (ticker.bid + ticker.ask) / 2 if ticker.bid and ticker.ask else None
                cost = price * 100 if price else None
                open_interest = getattr(ticker, "openInterest", 0)
                iv = getattr(ticker.modelGreeks, 'impliedVol', None)
                iv = float(iv) if iv else TARGET_IV
                if price is None or math.isnan(price) or cost is None:
                    continue
                if cost > price_limit:
                    continue
                liquidity_score = get_liquidity_score(ticker.bid, ticker.ask, open_interest)
                freshness_score = get_freshness_score(signal_date)
                cost_score = get_cost_score(cost, price_limit)
                iv_score = get_iv_quality_score(iv)
                profit_tgt, stop_loss, risk = calc_risk_targets(price, iv)
                rr_score = profit_tgt / risk if risk > 0 else 0
                # Composite score (weights: RR 0.4, freshness 0.2, IV 0.15, liquidity 0.15, cost 0.1)
                total_score = (
                    0.4 * rr_score +
                    0.2 * freshness_score +
                    0.15 * iv_score +
                    0.15 * liquidity_score +
                    0.1 * cost_score
                )
                option_candidates.append({
                    "score": round(total_score, 4),
                    "contract": contract,
                    "price": price,
                    "cost": cost,
                    "iv": iv,
                    "profit_tgt": profit_tgt,
                    "stop_loss": stop_loss,
                    "risk": risk,
                    "strike": strike,
                    "expiry": expiry,
                    "liquidity_score": liquidity_score,
                    "freshness_score": freshness_score,
                    "cost_score": cost_score,
                    "iv_score": iv_score,
                    "rr_score": rr_score
                })
            except Exception as e:
                continue
    # Rank all candidates
    option_candidates.sort(key=lambda x: x['score'], reverse=True)
    return option_candidates[:MAX_SHOW_TRADES]

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
        # ... tickers list here as in your script ...
        "AAPL", "NVDA", "TSLA", "AMZN", "MSFT", "TSLL", "TQQQ", "NUGT", "EEM", "SOXL", "IBIT", "ARKK", "QTUM", "SVXY",
        "STX", "PLTR", "WDC", "NEM", "MU", "GEV", "ORCL", "WBD", "NRG", "GE", "APH", "LRCX", "TPR", "HWM", "GLW",
        "CVS", "KLAC", "UBER", "IDXX", "DASH", "JBL", "MPWR", "VST", "TEL", "WYNN", "APP", "ZS", "FAST", "AVGO",
        "CEG", "MDB", "MELI", "ORLY", "TTWO", "NFLX", "AMD", "PDD", "META", "CRWD", "AXON", "GOOGL", "GOOG",
        "TDUP", "OPEN", "LEU", "IMRX", "AMPX", "CELC", "AEVA", "BE", "OPRX", "CMCL", "AMLX", "FUBO", "PGEN", "MASS",
        "APPS", "MLYS", "SATS", "COMM", "KTOS", "LCTX", "CDE", "LASR", "METC", "UUUU", "CPS", "SSO", "QQQ", "SPY",
        "SMCI", "APLD", "ILMN", "NBIS", "IONQ",
        "BAP", "ITUB", "BSBR", "ING", "DB", "BBVA", "NU", "CIB", "SMFG", "LYG",
        "CANG", "BILI", "TV", "PAGS", "TME", "VIPS", "CZR", "YNDX", "FANUY",
        "SCPH", "GDS", "GH", "JAZZ", "QGEN", "CPNG", "PRGO", "NVS", "RPRX", "TEVA",
        "VALE", "SBSW", "EC", "YPF", "SHEL", "SSL", "SCCO", "RIO", "PBR", "STM", 'GDXJ'
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
            option_candidates = []
            for event in crosses:
                when = event['date']
                if (datetime.now().date() - when.date()) > timedelta(days=10):
                    continue
                swing_high = event['swing_high_prev']
                underlying = Stock(ticker, 'SMART', 'USD')
                ib.qualifyContracts(underlying)
                ranked_options = select_option_contracts_and_rank(
                    ib, underlying, 45, swing_high, max_alloc_val, when
                )
                for opt in ranked_options:
                    option_candidates.append({
                        **opt,
                        "symbol": ticker,
                        "timeframe": "DAILY 9/18",
                        "signal_date": when
                    })
            # Sorting and presenting only the top MAX_SHOW_TRADES:
            option_candidates.sort(key=lambda x: x['score'], reverse=True)
            top_options = option_candidates[:MAX_SHOW_TRADES]

            # Display the top choices
            for i, opt in enumerate(top_options, 1):
                print("\n[TOP {}] {} {} strike {}C exp {} | Score {:.3f}".format(
                    i, opt['symbol'], opt['timeframe'], opt['strike'], opt['expiry'], opt['score']))
                print(f"  Price: {opt['price']} | IV: {opt['iv']:.2f} | RR: {opt['rr_score']:.2f} | Freshness: {opt['freshness_score']:.2f}")
                print(f"  Liquidity: {opt['liquidity_score']:.2f} | Cost: {opt['cost']} | Profit tgt: {opt['profit_tgt']} | Stop: {opt['stop_loss']}")
                answer = input("Place this trade? (y/n): ").strip().lower()
                if answer == 'y':
                    size = int(max_alloc_val // (opt['price']*100))
                    if size < 1:
                        print(f"[Alloc] Position too small, skipping.")
                        continue
                    if size > 10: size = 10
                    place_entry_with_dynamic_exits(
                        ib, opt['contract'], opt['price'], size,
                        opt['profit_tgt'], opt['stop_loss']
                    )
                    filled_trades.append({
                        "symbol": opt['symbol'],
                        "contract": f"{opt['contract'].symbol} {opt['contract'].lastTradeDateOrContractMonth} {opt['contract'].strike}C",
                        "qty": size,
                        "premium": opt['price'],
                        "iv": opt['iv'],
                        "target": opt['profit_tgt'],
                        "stop": opt['stop_loss'],
                        "score": opt['score'],
                        "alloc_pct": round(size * opt['price'] * 100 / account_bal * 100, 2),
                        "timeframe": "DAILY 9/18",
                        "signal_date": opt['signal_date'].strftime('%Y-%m-%d'),
                        "entry_time": datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                    })

    ib.disconnect()
    print("\n[Result] Script finished scanning all tickers.")

    # Summary Table of Filled Trades
    if filled_trades:
        print("\n==== SUMMARY OF TRADES EXECUTED ====")
        print("Idx | Symbol | Contract           | Qty | EntryPx | IV    | Target  | Stop    | Score  | Alloc% | Timeframe   | SignalDate | EntryTime")
        print("----|--------|--------------------|-----|---------|-------|---------|---------|--------|--------|-------------|------------|-------------------")
        for i, t in enumerate(filled_trades, 1):
            print(f"{i:<3} | {t['symbol']:<6} | {t['contract']:<18} | {t['qty']:<3} | {t['premium']:<7.2f} | {t['iv']:<5.2f} | {t['target']:<7.2f} | {t['stop']:<7.2f} | {t['score']:<6.3f} | {t['alloc_pct']:<6.2f} | {t['timeframe']:<11} | {t['signal_date']:<10} | {t['entry_time']}")
    else:
        print("\n[Result] No trades were filled in this session.")

if __name__ == "__main__":
    main()
