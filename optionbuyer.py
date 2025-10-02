import numpy as np
import pandas as pd
from datetime import datetime
from ib_insync import *
import re

def smma(series, window):
    return pd.Series(series).ewm(alpha=1/window, adjust=False).mean()

def fetch_bars(ib, ticker, duration_str, bar_size, exchange):
    print(f"[Process] Fetching {bar_size} bars for {ticker} on {exchange} over {duration_str}...")
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
    except Exception as e:
        print(f"[Error] Could not fetch bars for {ticker} on {exchange}: {str(e)}")
        return None

def analyze_crosses_and_reentries(df, fast, slow, ticker, timeframe):
    print(f"[Process] Analyzing SMMA crosses and re-entry signals for {ticker} [{timeframe}]: fast={fast}, slow={slow}...")
    today = datetime.now().date()
    df['smma_fast'] = smma(df['close'], fast)
    df['smma_slow'] = smma(df['close'], slow)
    prev_bull_high, prev_bull_date = None, None
    prev_bear_low, prev_bear_date = None, None
    swing_high = None
    cross_results = []
    events = []
    reentry_signals = []
    price_below = False

    for i in range(1, len(df)):
        if df['smma_fast'].iloc[i] > df['smma_slow'].iloc[i]:
            if swing_high is None or df['high'].iloc[i] > swing_high:
                swing_high = df['high'].iloc[i]
        if df['smma_fast'].iloc[i] > df['smma_slow'].iloc[i] and df['smma_fast'].iloc[i-1] <= df['smma_slow'].iloc[i-1]:
            cross_date = df.index[i]
            if prev_bull_high is not None:
                msg = f"{ticker} [{timeframe}] BULL cross at {cross_date.date()} -> Previous bull cross HIGH: {prev_bull_high:.2f} on {prev_bull_date.date()}"
                cross_results.append((cross_date.date(), msg))
                events.append({"type": "bull", "cross_date": cross_date, "prev_bull_high": prev_bull_high, "prev_bull_date": prev_bull_date})
            prev_bull_high = df['high'].iloc[i]
            prev_bull_date = df.index[i]
            swing_high = df['high'].iloc[i]
            price_below = False
        if df['smma_fast'].iloc[i] < df['smma_slow'].iloc[i] and df['smma_fast'].iloc[i-1] >= df['smma_slow'].iloc[i-1]:
            cross_date = df.index[i]
            if prev_bear_low is not None:
                msg = f"{ticker} [{timeframe}] BEAR cross at {cross_date.date()} -> Previous bear cross LOW: {prev_bear_low:.2f} on {prev_bear_date.date()}"
                cross_results.append((cross_date.date(), msg))
                events.append({"type": "bear", "cross_date": cross_date, "prev_bull_high": prev_bull_high, "prev_bear_low": prev_bear_low, "prev_bear_date": prev_bear_date})
            prev_bear_low = df['low'].iloc[i]
            prev_bear_date = df.index[i]
            swing_high = None
            price_below = False
        if df['smma_fast'].iloc[i] > df['smma_slow'].iloc[i]:
            if df['close'].iloc[i-1] > df['smma_slow'].iloc[i-1] and df['close'].iloc[i] < df['smma_slow'].iloc[i]:
                price_below = True
            if price_below and df['close'].iloc[i] > df['smma_slow'].iloc[i]:
                msg = f"{ticker} [{timeframe}] RE-ENTRY: Close back above slow smma at {df.index[i].date()} (bull regime). Strike ≈ swing high {swing_high:.2f}"
                reentry_signals.append((i, df.index[i], swing_high, msg))
                price_below = False
    print(f"[Result] SMMA analysis for {ticker} [{timeframe}] complete.")
    return cross_results, events, reentry_signals

def select_option_contract(ib, underlying, expiry_after_days, strike_target):
    print(f"[Process] Selecting option contract for {underlying.symbol}: expiry ≈ {expiry_after_days} days, strike ≈ {strike_target:.2f}")
    chains = ib.reqSecDefOptParams(underlying.symbol, '', underlying.secType, underlying.conId)
    chain = next((c for c in chains if c.tradingClass == underlying.symbol or c.exchange in ['SMART', 'CBOE']), None)
    if not chain:
        print(f"[Error] No option chain found for {underlying.symbol}. Cannot select contract.")
        return None
    today = datetime.now().date()
    expiry_dates = sorted([datetime.strptime(d, "%Y%m%d").date() for d in chain.expirations])
    if not expiry_dates:
        print(f"[Error] No expiries found for {underlying.symbol}.")
        return None
    strikes = sorted(chain.strikes)
    if not strikes:
        print(f"[Error] No strikes available for {underlying.symbol}.")
        return None
    print(f"[Diagnostic] Available expiries: {expiry_dates}")
    print(f"[Diagnostic] Available strikes: {strikes}")
    target_expiry = min(expiry_dates, key=lambda d: abs((d - today).days - expiry_after_days))
    target_expiry_str = target_expiry.strftime("%Y%m%d")
    target_strike = min(strikes, key=lambda x: abs(x - strike_target))
    contract = Option(underlying.symbol, target_expiry_str, target_strike, 'C', 'SMART')
    try:
        ib.qualifyContracts(contract)
        print(f"[Result] Selected contract: {contract.symbol} {contract.lastTradeDateOrContractMonth} {contract.right} Strike: {contract.strike}")
        return contract
    except Exception as e:
        print(f"[Error] Option contract qualification failed: {str(e)}")
        return None

def get_option_metrics(ib, contract):
    contract.exchange = "SMART"
    contract.primaryExchange = "SMART"
    ticker = ib.reqMktData(contract, "106", False, False)
    ib.sleep(2)
    try:
        delta = ticker.modelGreeks.delta
        iv = ticker.modelGreeks.impliedVol
        bid, ask = ticker.bid, ticker.ask
        price = (bid + ask) / 2 if bid is not None and ask is not None else None
        return delta, iv, price
    except Exception:
        return None, None, None

def fmt(v):
    try:
        if v is None:
            return 'NA'
        return f"{float(v):.2f}"
    except:
        return str(v) if v is not None else 'NA'

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
                avg_cost = getattr(pos, 'avgCost', None)
                pnl = (price - avg_cost) * 100 * pos.position if (price is not None and avg_cost is not None) else None
            except Exception:
                delta, iv, price, days_to_exp, pnl, avg_cost = None, None, None, None, None, None
            open_positions.append({
                'ticker': con.symbol,
                'expiry': con.lastTradeDateOrContractMonth,
                'strike': con.strike,
                'right': con.right,
                'position': pos.position,
                'avg_cost': avg_cost,
                'price': price,
                'days_to_exp': days_to_exp,
                'delta': delta,
                'iv': iv,
                'pnl': pnl
            })
    return open_positions

def market_close_option(ib, contract, qty):
    contract.exchange = "SMART"
    contract.primaryExchange = "SMART"
    print(f"[Process] Submitting market SELL order for {qty} {contract.symbol} {contract.lastTradeDateOrContractMonth} {contract.right} {contract.strike}...")
    order = MarketOrder('SELL', qty)
    trade = ib.placeOrder(contract, order)
    while not trade.isDone():
        ib.waitOnUpdate(timeout=1)
    print(f"[Result] MARKET CLOSE: Filled {trade.orderStatus.filled} {contract.symbol} {contract.strike} at status {trade.orderStatus.status}")
    return trade

def confirm_trade(action, contract, ticker, reason=''):
    print(f"[Prompt] About to show confirmation for: {action} 1 {contract.symbol} {contract.lastTradeDateOrContractMonth} {contract.right} Strike: {contract.strike} Ticker: {ticker} [Reason: {reason}]")
    msg = (
        f"\nDo you want to take this trade? "
        f"{action} 1 {contract.symbol} {contract.lastTradeDateOrContractMonth} "
        f"{contract.right} Strike: {contract.strike} Ticker: {ticker}"
    )
    if reason:
        msg += f" [Reason: {reason}]"
    response = input(msg + " (y/n): ").strip().lower()
    return response == "y"

def option_position_open(ib, contract):
    positions = ib.positions()
    for pos in positions:
        con = pos.contract
        if (
            isinstance(con, Option)
            and con.symbol == contract.symbol
            and con.lastTradeDateOrContractMonth == contract.lastTradeDateOrContractMonth
            and abs(con.strike - contract.strike) < 0.01
            and con.right == contract.right
            and pos.position > 0
        ):
            return True
    return False

def place_order(ib, contract, action, ticker, reason=''):
    print(f"[Intent] Requesting trade: {action} 1 {contract.symbol} {contract.lastTradeDateOrContractMonth} {contract.right} Strike: {contract.strike} Ticker: {ticker} [Reason: {reason}]")
    if contract is None:
        print(f"[Info] Skipping order: contract object is None, cannot place order for {ticker}.")
        return None
    if action == 'BUY':
        if option_position_open(ib, contract):
            print(f"[Info] Skipping BUY: open call already detected for {ticker} {contract.strike} {contract.lastTradeDateOrContractMonth} {contract.right}.")
            return None
    elif action == 'SELL':
        if not option_position_open(ib, contract):
            print(f"[Info] Skipping SELL: no open call position found for {ticker} {contract.strike} {contract.lastTradeDateOrContractMonth} {contract.right}.")
            return None
    if confirm_trade(action, contract, ticker, reason):
        print(f"[Process] Placing {action} order...")
        order = MarketOrder(action, 1)
        trade = ib.placeOrder(contract, order)
        print(f"[Process] Order submitted: {action} 1 {contract.symbol} {contract.lastTradeDateOrContractMonth} {contract.strike} {contract.right}")
        while not trade.isDone():
            ib.waitOnUpdate(timeout=1)
        print(f"[Result] TRADE EXECUTED: {action} 1 {contract.symbol} {contract.lastTradeDateOrContractMonth} {contract.right} Strike: {contract.strike}")
        print(f"[Result] Order status: {trade.orderStatus.status}, filled: {trade.orderStatus.filled}")
        return trade
    else:
        print("[Info] Trade cancelled by user.")
        return None

def main():
    print("\n=== SMMA MULTI-TICKER OPTION TRADER ===")
    tickers = [
        "AAPL", "AMZN", "AMD", "APP", "ARKK", "DUST", "GOOGL", "HOOD", "IBIT", "MASS",
        "META", "MSFT", "NFLX", "NVDA", "NUGT", "PLTR", "QCOM", "QQQ", "QTUM", "SARK",
        "SOXL", "SOXS", "SPXS", "SPY", "SQQQ", "SSO", "TQQQ", "TSLA", "TSLL", "VXX"
    ]
    exchanges = ['ARCA', 'NASDAQ', 'SMART']
    lookback = '180 D'
    ib = IB()
    try:
        ib.connect('127.0.0.1', 4001, clientId=101)
        print("[Result] Connected to IBKR.")
    except Exception as e:
        print(f"[Error] Could not connect to IBKR: {str(e)}")
        return

    today = datetime.now().date()
    trades_taken = False

    for ticker in tickers:
        position_is_open_daily = False
        daily_contract = None
        position_is_open_4h = False
        contract_4h = None

        bars_daily = None
        for exch in exchanges:
            bars_daily = fetch_bars(ib, ticker, lookback, "1 day", exch)
            if bars_daily:
                break
        if not bars_daily or len(bars_daily) == 0:
            print(f"[Error] No valid daily data for {ticker} on any supported exchange.")
        else:
            df_daily = util.df(bars_daily)
            df_daily['date'] = pd.to_datetime(df_daily['date'])
            df_daily = df_daily.set_index('date')
            cross_results_daily, events_daily, reentries_daily = analyze_crosses_and_reentries(df_daily, 9, 18, ticker, "DAILY 9/18")
            if cross_results_daily:
                for _, line in cross_results_daily:
                    print(line)
            for event in events_daily:
                if event["type"] == "bull":
                    cross_day = event["cross_date"].date()
                    if (today - cross_day).days <= 3 and not position_is_open_daily:
                        if event["prev_bull_high"] is not None:
                            underlying = Stock(ticker, 'SMART', 'USD')
                            ib.qualifyContracts(underlying)
                            option_contract = select_option_contract(ib, underlying, 45, event["prev_bull_high"])
                            if option_contract:
                                trade = place_order(ib, option_contract, 'BUY', ticker, "bullish crossover (daily)")
                                if trade is not None:
                                    trades_taken = True
                                position_is_open_daily = True
                                daily_contract = option_contract
            for idx, bar_date, swing_high, msg in reentries_daily:
                if (today - bar_date.date()).days <= 3 and not position_is_open_daily:
                    print("[Signal] " + msg)
                    underlying = Stock(ticker, 'SMART', 'USD')
                    ib.qualifyContracts(underlying)
                    if swing_high is not None:
                        option_contract = select_option_contract(ib, underlying, 45, swing_high)
                        if option_contract:
                            trade = place_order(ib, option_contract, 'BUY', ticker, "reentry after close above slow SMMA (daily)")
                            if trade is not None:
                                trades_taken = True
                            position_is_open_daily = True
                            daily_contract = option_contract
            for event in events_daily:
                if event["type"] == "bear" and position_is_open_daily and daily_contract is not None:
                    if event.get("prev_bull_high") is not None:
                        trade = place_order(ib, daily_contract, 'SELL', ticker, "bearish crossover (daily)")
                        if trade is not None:
                            trades_taken = True
                        position_is_open_daily = False
                        daily_contract = None

        bars_4h = None
        for exch in exchanges:
            bars_4h = fetch_bars(ib, ticker, lookback, "4 hours", exch)
            if bars_4h:
                break
        if not bars_4h or len(bars_4h) == 0:
            print(f"[Error] No valid 4-hour data for {ticker} on any supported exchange.")
        else:
            df_4h = util.df(bars_4h)
            df_4h['date'] = pd.to_datetime(df_4h['date'])
            df_4h = df_4h.set_index('date')
            cross_results_4h, events_4h, reentries_4h = analyze_crosses_and_reentries(df_4h, 26, 150, ticker, "4H 26/150")
            if cross_results_4h:
                for _, line in cross_results_4h:
                    print(line)
            total_bars_4h = len(df_4h)
            for event in events_4h:
                if event["type"] == "bull":
                    bar_idx = df_4h.index.get_loc(event["cross_date"])
                    if total_bars_4h - bar_idx <= 8 and not position_is_open_4h:
                        if event["prev_bull_high"] is not None:
                            underlying = Stock(ticker, 'SMART', 'USD')
                            ib.qualifyContracts(underlying)
                            option_contract = select_option_contract(ib, underlying, 45, event["prev_bull_high"])
                            if option_contract:
                                trade = place_order(ib, option_contract, 'BUY', ticker, "bullish crossover (4-hour)")
                                if trade is not None:
                                    trades_taken = True
                                position_is_open_4h = True
                                contract_4h = option_contract
            for idx, bar_date, swing_high, msg in reentries_4h:
                bar_int = df_4h.index.get_loc(bar_date)
                if (total_bars_4h - bar_int) <= 8 and not position_is_open_4h:
                    print("[Signal] " + msg)
                    underlying = Stock(ticker, 'SMART', 'USD')
                    ib.qualifyContracts(underlying)
                    if swing_high is not None:
                        option_contract = select_option_contract(ib, underlying, 45, swing_high)
                        if option_contract:
                            trade = place_order(ib, option_contract, 'BUY', ticker, "reentry after close above slow SMMA (4-hour)")
                            if trade is not None:
                                trades_taken = True
                            position_is_open_4h = True
                            contract_4h = option_contract
            for event in events_4h:
                if event["type"] == "bear" and position_is_open_4h and contract_4h is not None:
                    if event.get("prev_bull_high") is not None:
                        trade = place_order(ib, contract_4h, 'SELL', ticker, "bearish crossover (4-hour)")
                        if trade is not None:
                            trades_taken = True
                        position_is_open_4h = False
                        contract_4h = None

    # === ADDED: Print open options positions and allow user to close them ===
    open_positions = find_open_option_positions(ib)
    if open_positions:
        print("\n=== OPEN OPTION POSITIONS (WITH UNREALIZED P/L) ===")
        print("Idx | Ticker | Expiry   | Strike   | Type | Qty | EntryPx | LastPx | OpenPnL")
        print("----------------------------------------------------------------------------")
        for i, p in enumerate(open_positions):
            print(f"{i:<4}| {p['ticker']:<6}| {p['expiry']:<9}| {fmt(p['strike']):<8}| {p['right']:<5}| {fmt(p['position']):<4}| {fmt(p['avg_cost']):<8}| {fmt(p['price']):<7}| {fmt(p['pnl']):<8}")
        sel = input("\nEnter row numbers of positions to CLOSE (comma/space), or press Enter to skip: ").strip()
        idxs = [int(v) for v in re.findall(r'\d+', sel)]
        for idx in idxs:
            if 0 <= idx < len(open_positions):
                pos = open_positions[idx]
                contract = Option(pos['ticker'], pos['expiry'], pos['strike'], pos['right'], "SMART")
                contract.exchange = "SMART"
                contract.primaryExchange = "SMART"
                market_close_option(ib, contract, int(pos['position']))
    else:
        print("\nNo open option positions detected.")

    ib.disconnect()
    print("[Result] Script finished: disconnected from IBKR.")
    if trades_taken:
        print("\nAt least one trade was placed during this run.")
    else:
        print("\nNo trades placed during this run.")

if __name__ == "__main__":
    main()



    # =============================================================================
# Script Summary:
#
# This script is an automated options trading assistant for Interactive Brokers.
# It processes a defined list of stock tickers, analyzing both daily and 4-hour 
# price data for each stock across major US exchanges. 
#
# The script performs the following:
#   - Connects to the Interactive Brokers API (IBKR).
#   - Fetches historical price data for each ticker.
#   - Calculates fast and slow Smoothed Moving Averages (SMMA) on the price data.
#   - Scans for bullish/bearish crossovers and "re-entry" signals that indicate 
#     potential trade opportunities.
#   - When a signal is detected, it selects a suitable options contract with 
#     strike and expiry near the signal's reference point, prompts the user for 
#     confirmation, and places BUY/SELL orders for options accordingly.
#   - Tracks and reports all trades taken during the session.
#   - At the end, displays any open option positions in the IBKR account and 
#     gives the user the choice to close them.
#   - Cleanly disconnects from IBKR at completion.
#
# The script is interactive: it requires user confirmation for any real trade, 
# making it a semi-automated, risk-controlled trading workflow.
# =============================================================================

