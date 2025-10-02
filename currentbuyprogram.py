import numpy as np
import pandas as pd
from datetime import datetime
from math import sqrt
from ib_insync import *
import re

def smma(series, window):
    return pd.Series(series).ewm(alpha=1/window, adjust=False).mean()

def fetch_bars(ib, ticker, duration_str, bar_size, exchange):
    contract = Stock(ticker, exchange, 'USD')
    try:
        bars = ib.reqHistoricalData(
            contract,
            endDateTime='',
            durationStr=duration_str,
            barSizeSetting=bar_size,
            whatToShow='TRADES',
            useRTH=False
        )
        return bars
    except Exception:
        return None

def analyze_crosses_and_reentries(df, fast, slow, ticker, timeframe, lookback=10):
    df['smma_fast'] = smma(df['close'], fast)
    df['smma_slow'] = smma(df['close'], slow)
    candidates = []
    cross_events = []
    swing_high = None
    price_below = False
    for i in range(1, len(df)):
        if df['smma_fast'].iloc[i] > df['smma_slow'].iloc[i]:
            if swing_high is None or df['high'].iloc[i] > swing_high:
                swing_high = df['high'].iloc[i]
        if df['smma_fast'].iloc[i] > df['smma_slow'].iloc[i] and df['smma_fast'].iloc[i-1] <= df['smma_slow'].iloc[i-1]:
            cross_events.append({"type": "bull", "cross_date": df.index[i]})
            if i >= len(df) - lookback:
                candidates.append({"type": "bull", "cross_date": df.index[i], "swing_high": df['high'].iloc[i]})
            swing_high = df['high'].iloc[i]
            price_below = False
        if df['smma_fast'].iloc[i] < df['smma_slow'].iloc[i] and df['smma_fast'].iloc[i-1] >= df['smma_slow'].iloc[i-1]:
            cross_events.append({"type": "bear", "cross_date": df.index[i]})
            swing_high = None
            price_below = False
        if df['smma_fast'].iloc[i] > df['smma_slow'].iloc[i]:
            if df['close'].iloc[i-1] > df['smma_slow'].iloc[i-1] and df['close'].iloc[i] < df['smma_slow'].iloc[i]:
                price_below = True
            if price_below and df['close'].iloc[i] > df['smma_slow'].iloc[i]:
                if i >= len(df) - lookback:
                    candidates.append({"type": "reentry", "cross_date": df.index[i], "swing_high": swing_high})
                price_below = False
    return candidates, cross_events

def select_option_contract(ib, underlying, expiry_after_days, strike_target):
    try:
        chains = ib.reqSecDefOptParams(underlying.symbol, '', underlying.secType, underlying.conId)
        chain = next((c for c in chains if c.tradingClass == underlying.symbol or c.exchange in ['SMART', 'CBOE']), None)
        if not chain: return None
        today = datetime.now().date()
        expiry_dates = sorted([datetime.strptime(d, "%Y%m%d").date() for d in chain.expirations])
        if not expiry_dates: return None
        strikes = sorted(chain.strikes)
        if not strikes: return None
        if strike_target < strikes[0]: strike_target = strikes[0]
        if strike_target > strikes[-1]: strike_target = strikes[-1]
        target_expiry = min(expiry_dates, key=lambda d: abs((d - today).days - expiry_after_days))
        target_expiry_str = target_expiry.strftime("%Y%m%d")
        target_strike = min(strikes, key=lambda x: abs(x - strike_target))
        contract = Option(underlying.symbol, target_expiry_str, target_strike, 'C', 'SMART')
        ib.qualifyContracts(contract)
        if not getattr(contract, 'conId', None) or contract.conId == 0:
            return None
        return contract
    except Exception:
        return None

def get_option_metrics(ib, contract):
    try:
        ticker = ib.reqMktData(contract, "106", False, False)  # 106 for Greeks
        ib.sleep(2)
        delta = ticker.modelGreeks.delta
        iv = ticker.modelGreeks.impliedVol
        bid, ask = ticker.bid, ticker.ask
        price = (bid + ask) / 2 if bid is not None and ask is not None else None
        return delta, iv, price
    except Exception:
        return None, None, None

def multifactor_score(delta, iv, momentum):
    sc = 0
    if delta is not None: sc += abs(delta) * 0.5
    if iv is not None: sc += (1.0 - min(iv, 2.0) / 2.0) * 0.25
    sc += momentum * 0.25
    return round(sc, 3)

def place_order(ib, contract, ticker, reason='', action='BUY'):
    print(f"[ORDER] {action} 1 {contract.symbol} {contract.lastTradeDateOrContractMonth} {contract.strike} {contract.right} ({reason})")
    order = MarketOrder(action, 1)
    trade = ib.placeOrder(contract, order)
    while not trade.isDone():
        ib.waitOnUpdate(timeout=1)
    fill_px = trade.log[-1].price if trade.log else 'unknown'
    print(f"[EXECUTED] {action} {contract.symbol} {contract.strike} @ {fill_px}")
    return trade

def find_open_option_positions(ib):
    open_positions = []
    positions = ib.positions()
    for pos in positions:
        con = pos.contract
        if isinstance(con, Option) and pos.position > 0:
            try:
                delta, iv, price = get_option_metrics(ib, con)
                expiry_dt = datetime.strptime(con.lastTradeDateOrContractMonth[:8], "%Y%m%d")
                days_to_exp = (expiry_dt.date() - datetime.now().date()).days
                pct_otm = 100 * ((con.strike - pos.averageCost) / pos.averageCost)
                exp_move = pos.averageCost * (iv if iv is not None else 0.3) * sqrt(days_to_exp / 365) if iv is not None and days_to_exp > 0 else None
                in_exp_move = abs(con.strike - pos.averageCost) <= exp_move if exp_move is not None else None
                pnl = (price - pos.averageCost) * 100 * pos.position if (price is not None and pos.averageCost is not None) else None
            except Exception:
                delta, iv, price, days_to_exp, pct_otm, exp_move, in_exp_move, pnl = None, None, None, None, None, None, None, None
            open_positions.append({
                'ticker': con.symbol,
                'expiry': con.lastTradeDateOrContractMonth,
                'strike': con.strike,
                'right': con.right,
                'position': pos.position,
                'avg_cost': pos.averageCost,
                'delta': delta,
                'iv': iv,
                'price': price,
                'days_to_exp': days_to_exp,
                'pct_otm': pct_otm,
                'exp_move': exp_move,
                'in_exp_move': in_exp_move,
                'pnl': pnl
            })
    return open_positions

def fmt(v):
    try:
        if v is None: return 'NA'
        return f"{float(v):.2f}"
    except:
        return str(v) if v is not None else 'NA'

def main():
    print("\n=== SMMA MULTI-TICKER OPTION SIGNAL RANKER (EXCLUDES EXISTING CALL POSITIONS) ===")
    tickers = [
        "AAPL", "AMZN", "AMD", "APP", "ARKK", "DUST", "GOOGL", "HOOD", "IBIT", "MASS",
        "META", "MSFT", "NFLX", "NVDA", "NUGT", "PLTR", "QCOM", "QQQ", "QTUM", "SARK",
        "SOXL", "SOXS", "SPXS", "SPY", "SQQQ", "SSO", "TQQQ", "TSLA", "TSLL", "VXX"
    ]
    exchanges = ['ARCA', 'NASDAQ', 'SMART']
    lookback = '180 D'
    bar_lookback = 10
    ib = IB()
    try:
        ib.connect('127.0.0.1', 4001, clientId=101)
    except Exception as e:
        print(f"Could not connect to IBKR: {str(e)}")
        return

    today = datetime.now().date()
    all_signals = []
    buy_candidates_map = {}

    # --- Get open positions, build a set for fast lookup ---
    open_positions = find_open_option_positions(ib)
    current_calls = set()
    for p in open_positions:
        if p['right'] == 'C':
            current_calls.add((p['ticker'], p['strike'], p['expiry']))

    for ticker in tickers:
        # DAILY scan
        bars_daily = None
        for exch in exchanges:
            bars_daily = fetch_bars(ib, ticker, lookback, "1 day", exch)
            if bars_daily: break
        if bars_daily and len(bars_daily) > 0:
            df_daily = util.df(bars_daily)
            df_daily['date'] = pd.to_datetime(df_daily['date'])
            df_daily = df_daily.set_index('date')
            candidates, cross_events = analyze_crosses_and_reentries(df_daily, 9, 18, ticker, "DAILY 9/18", lookback=bar_lookback)
            last_close = df_daily['close'].iloc[-1]
            momentum = (df_daily['close'].iloc[-1] - df_daily['close'].iloc[-10]) / df_daily['close'].iloc[-10] if len(df_daily) > 10 else 0.0
            for signal in candidates:
                underlying = Stock(ticker, 'SMART', 'USD')
                ib.qualifyContracts(underlying)
                oc = select_option_contract(ib, underlying, 45, signal['swing_high'])
                if not oc:
                    continue
                # --- SKIP if already holding this call contract ---
                if (ticker, oc.strike, oc.lastTradeDateOrContractMonth) in current_calls:
                    continue
                delta, iv, price = get_option_metrics(ib, oc)
                try:
                    expiry_dt = datetime.strptime(oc.lastTradeDateOrContractMonth[:8], "%Y%m%d")
                    days_to_exp = (expiry_dt.date() - today).days
                except Exception:
                    days_to_exp = None
                try:
                    pct_otm = 100*((oc.strike - last_close)/last_close)
                except Exception:
                    pct_otm = None
                prob_itm = abs(delta) if delta is not None else None
                prob_touch = min(2*abs(delta), 1.0) if delta is not None else None
                exp_move = last_close * iv * sqrt(days_to_exp/365) if iv is not None and days_to_exp is not None and days_to_exp > 0 else None
                in_exp_move = abs(oc.strike - last_close) <= exp_move if exp_move is not None else None
                sc = multifactor_score(delta, iv, momentum)
                sigdict = {
                    'ticker': ticker,
                    'type': 'daily-' + signal['type'],
                    'underlying_price': last_close,
                    'option_price': price,
                    'delta': delta,
                    'iv': iv,
                    'momentum': momentum,
                    'strike': oc.strike,
                    'expiry': oc.lastTradeDateOrContractMonth,
                    'days_to_exp': days_to_exp,
                    'pct_otm': pct_otm,
                    'prob_itm': prob_itm,
                    'prob_touch': prob_touch,
                    'exp_move': exp_move,
                    'in_exp_move': in_exp_move,
                    'score': sc,
                    'swing_high': signal['swing_high'],
                    'cross_date': str(signal['cross_date'].date()) if 'cross_date' in signal else ''
                }
                all_signals.append(sigdict)
                buy_candidates_map[(ticker, oc.strike, oc.lastTradeDateOrContractMonth)] = sigdict
            for ce in cross_events:
                if ce['type'] == 'bear':
                    cross_day = ce['cross_date'].date()
                    if ce['cross_date'] >= df_daily.index[-bar_lookback]:
                        for pos in open_positions:
                            con = Option(pos['ticker'], pos['expiry'], pos['strike'], 'C', 'SMART')
                            if (
                                pos['ticker'] == ticker
                                and pos['position'] > 0
                                and (ticker, pos['strike'], pos['expiry']) in buy_candidates_map
                            ):
                                print(f"\n[SELL SIGNAL] {ticker} {pos['strike']} {pos['expiry']} -- detected bear cross {cross_day}.")
                                inp = input(f"SELL {pos['position']} contracts ({pos['ticker']} {pos['strike']} {pos['expiry']}) now? (y/n): ").strip().lower()
                                if inp == "y":
                                    place_order(ib, con, ticker, f"Bear SMA cross ({cross_day})", action='SELL')
        # 4H scan
        bars_4h = None
        for exch in exchanges:
            bars_4h = fetch_bars(ib, ticker, lookback, "4 hours", exch)
            if bars_4h: break
        if bars_4h and len(bars_4h) > 0:
            df_4h = util.df(bars_4h)
            df_4h['date'] = pd.to_datetime(df_4h['date'])
            df_4h = df_4h.set_index('date')
            candidates4, cross_events4 = analyze_crosses_and_reentries(df_4h, 26, 150, ticker, "4H 26/150", lookback=bar_lookback)
            last_close_4h = df_4h['close'].iloc[-1]
            momentum_4h = (df_4h['close'].iloc[-1] - df_4h['close'].iloc[-10]) / df_4h['close'].iloc[-10] if len(df_4h) > 10 else 0.0
            for signal in candidates4:
                underlying = Stock(ticker, 'SMART', 'USD')
                ib.qualifyContracts(underlying)
                oc = select_option_contract(ib, underlying, 45, signal['swing_high'])
                if not oc:
                    continue
                # --- SKIP if already holding this call contract ---
                if (ticker, oc.strike, oc.lastTradeDateOrContractMonth) in current_calls:
                    continue
                delta, iv, price = get_option_metrics(ib, oc)
                try:
                    expiry_dt = datetime.strptime(oc.lastTradeDateOrContractMonth[:8], "%Y%m%d")
                    days_to_exp = (expiry_dt.date() - today).days
                except Exception:
                    days_to_exp = None
                try:
                    pct_otm = 100*((oc.strike - last_close_4h)/last_close_4h)
                except Exception:
                    pct_otm = None
                prob_itm = abs(delta) if delta is not None else None
                prob_touch = min(2*abs(delta), 1.0) if delta is not None else None
                exp_move = last_close_4h * iv * sqrt(days_to_exp/365) if iv is not None and days_to_exp is not None and days_to_exp > 0 else None
                in_exp_move = abs(oc.strike - last_close_4h) <= exp_move if exp_move is not None else None
                sc = multifactor_score(delta, iv, momentum_4h)
                sigdict = {
                    'ticker': ticker,
                    'type': '4h-' + signal['type'],
                    'underlying_price': last_close_4h,
                    'option_price': price,
                    'delta': delta,
                    'iv': iv,
                    'momentum': momentum_4h,
                    'strike': oc.strike,
                    'expiry': oc.lastTradeDateOrContractMonth,
                    'days_to_exp': days_to_exp,
                    'pct_otm': pct_otm,
                    'prob_itm': prob_itm,
                    'prob_touch': prob_touch,
                    'exp_move': exp_move,
                    'in_exp_move': in_exp_move,
                    'score': sc,
                    'swing_high': signal['swing_high'],
                    'cross_date': str(signal['cross_date'].date()) if 'cross_date' in signal else ''
                }
                all_signals.append(sigdict)
                buy_candidates_map[(ticker, oc.strike, oc.lastTradeDateOrContractMonth)] = sigdict
            for ce in cross_events4:
                if ce['type'] == 'bear':
                    cross_day = ce['cross_date'].date()
                    if ce['cross_date'] >= df_4h.index[-bar_lookback]:
                        for pos in open_positions:
                            con = Option(pos['ticker'], pos['expiry'], pos['strike'], 'C', 'SMART')
                            if (
                                pos['ticker'] == ticker
                                and pos['position'] > 0
                                and (ticker, pos['strike'], pos['expiry']) in buy_candidates_map
                            ):
                                print(f"\n[SELL SIGNAL] {ticker} {pos['strike']} {pos['expiry']} -- detected bear 4H cross {cross_day}.")
                                inp = input(f"SELL {pos['position']} contracts ({pos['ticker']} {pos['strike']} {pos['expiry']}) now? (y/n): ").strip().lower()
                                if inp == "y":
                                    place_order(ib, con, ticker, f"Bear SMA 4H cross ({cross_day})", action='SELL')

    ranked = sorted(all_signals, key=lambda x: x['score'], reverse=True)
    headers = [
        "Idx", "Ticker", "Type", "UndPx", "OptPx", "Delta", "IV", "%OTM", "DTE", "ProbITM",
        "ProbTouch", "ExpMove", "Strike", "InExpMv", "Score", "Mom", "Expiry", "SwingHi", "CrossDate"
    ]
    print("\n=== BUY SIGNALS (EXCLUDING CURRENT HOLDINGS) ===")
    print(" | ".join(f"{h:<9}" for h in headers))
    print("-" * 170)
    for i, r in enumerate(ranked):
        print(
            f"{i:<9}{r['ticker']:<9}{r['type']:<9}"
            f"{fmt(r['underlying_price']):<9}"
            f"{fmt(r['option_price']):<9}"
            f"{fmt(r['delta']):<9}"
            f"{fmt(r['iv']):<9}"
            f"{fmt(r['pct_otm']):<9}"
            f"{r['days_to_exp'] if r['days_to_exp'] is not None else 'NA':<9}"
            f"{fmt(r['prob_itm']):<9}"
            f"{fmt(r['prob_touch']):<9}"
            f"{fmt(r['exp_move']):<9}"
            f"{fmt(r['strike']):<9}"
            f"{str(r['in_exp_move']) if r['in_exp_move'] is not None else 'NA':<9}"
            f"{fmt(r['score']):<9}"
            f"{fmt(r['momentum']):<9}"
            f"{r['expiry']:<9}"
            f"{fmt(r['swing_high']):<9}"
            f"{r['cross_date']}"
        )
    if ranked:
        sel = input("\nEnter row numbers of buys to execute (comma/space): ")
        idxs = [int(v) for v in re.findall(r'\d+', sel)]
        for idx in idxs:
            if 0 <= idx < len(ranked):
                r = ranked[idx]
                underlying = Stock(r['ticker'], 'SMART', 'USD')
                ib.qualifyContracts(underlying)
                oc = select_option_contract(ib, underlying, r['days_to_exp'], r['strike'])
                if oc:
                    place_order(ib, oc, r['ticker'], f"{r['type']} {r['cross_date']}")
    else:
        print("No buy signals found.")

    # --- ENHANCED OPEN POSITION TABLE ---
    open_positions = find_open_option_positions(ib)
    if open_positions:
        print("\n=== OPEN OPTION POSITIONS (WITH ENTRY PRICE, LAST PRICE, AND P/L) ===")
        print(" | ".join([
            f"{h:<9}" for h in [
                "Ticker", "Expiry", "Strike", "Type", "Qty", "EntryPx", "LastPx",
                "Delta", "IV", "DTE", "%OTM", "ExpMove", "InExpMv", "OpenPnL"
            ]
        ]))
        print("-" * 145)
        for p in open_positions:
            print(
                f"{p['ticker']:<9}{p['expiry']:<9}{fmt(p['strike']):<9}"
                f"{p['right']:<9}{p['position']:<9}{fmt(p['avg_cost']):<9}"
                f"{fmt(p['price']):<9}{fmt(p['delta']):<9}{fmt(p['iv']):<9}"
                f"{p['days_to_exp'] if p['days_to_exp'] is not None else 'NA':<9}"
                f"{fmt(p['pct_otm']):<9}{fmt(p['exp_move']):<9}"
                f"{str(p['in_exp_move']) if p['in_exp_move'] is not None else 'NA':<9}"
                f"{fmt(p['pnl']):<9}"
            )
    else:
        print("\nNo open option positions detected.")

    ib.disconnect()
    print("\n[Result] Script finished: disconnected from IBKR.")

if __name__ == "__main__":
    main()


# =============================================================================
# Script Summary:
#
# This script is an options signal scanner and semi-automated trade assistant
# for Interactive Brokers (IBKR). It analyzes a set of equity tickers by:
#
#   - Connecting to IBKR and fetching recent daily and 4-hour price bars.
#   - Calculating fast/slow Smoothed Moving Averages (SMMA) to identify 
#     bullish and re-entry signals for each ticker and timeframe.
#   - Evaluating suitable call option contracts (strike/expiry) for each signal.
#   - Computing option metrics (Greeks, implied volatility, probabilistic stats)
#     and a multifactor score for ranking signals by trade quality.
#   - Excluding trades for contracts already held, to avoid duplicates.
#   - Presenting ranked buy signals and allowing the user to select which 
#     trades to execute interactively.
#   - Detecting recent bearish cross signals and offering to close any 
#     corresponding open call option positions.
#   - Displaying all current open option positions with key statistics such as 
#     entry price, last price, Greeks, IV, DTE, %OTM, expected move, and P&L.
#   - Disconnecting cleanly from IBKR after processing.
#
# The workflow is interactive, combining systematic signal scanning and
# position management in a user-driven, risk-controlled manner.
# =============================================================================
