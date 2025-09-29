import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from ib_insync import *
import re

def smma(series, window):
    s = pd.Series(series)
    out = s.copy()
    out.iloc[:window] = s.iloc[:window].mean()
    for i in range(window, len(out)):
        out.iloc[i] = (out.iloc[i-1]*(window-1)+s.iloc[i])/window
    return out

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

def analyze_crosses_and_reentries(df, fast, slow, ticker):
    print(f"[Process] Analyzing SMMA crosses and re-entry signals for {ticker}: fast={fast}, slow={slow}...")
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
                msg = f"{ticker} BULL cross at {cross_date.date()} -> Previous bull cross HIGH: {prev_bull_high:.2f} on {prev_bull_date.date()}"
                cross_results.append((cross_date.date(), msg))
                events.append({
                    "type": "bull",
                    "cross_date": cross_date,
                    "prev_bull_high": prev_bull_high,
                    "prev_bull_date": prev_bull_date
                })
            prev_bull_high = df['high'].iloc[i]
            prev_bull_date = df.index[i]
            swing_high = df['high'].iloc[i]
            price_below = False
        if df['smma_fast'].iloc[i] < df['smma_slow'].iloc[i] and df['smma_fast'].iloc[i-1] >= df['smma_slow'].iloc[i-1]:
            cross_date = df.index[i]
            if prev_bear_low is not None:
                msg = f"{ticker} BEAR cross at {cross_date.date()} -> Previous bear cross LOW: {prev_bear_low:.2f} on {prev_bear_date.date()}"
                cross_results.append((cross_date.date(), msg))
                events.append({
                    "type": "bear",
                    "cross_date": cross_date,
                    "prev_bull_high": prev_bull_high,
                    "prev_bear_low": prev_bear_low,
                    "prev_bear_date": prev_bear_date
                })
            prev_bear_low = df['low'].iloc[i]
            prev_bear_date = df.index[i]
            swing_high = None
            price_below = False
        if df['smma_fast'].iloc[i] > df['smma_slow'].iloc[i]:
            if df['close'].iloc[i-1] > df['smma_slow'].iloc[i-1] and df['close'].iloc[i] < df['smma_slow'].iloc[i]:
                price_below = True
            if price_below and df['close'].iloc[i] > df['smma_slow'].iloc[i]:
                msg = f"{ticker} RE-ENTRY: Close back above slow smma at {df.index[i].date()} (bull regime). Strike ≈ swing high {swing_high:.2f}"
                reentry_signals.append((i, df.index[i], swing_high, msg))
                price_below = False
    print(f"[Result] SMMA analysis for {ticker} complete.")
    return cross_results, events, reentry_signals

def parse_duration(user_input):
    user_input = user_input.replace(" ", "").upper()
    match = re.match(r"^(\d+)([YMDW])$", user_input)
    if match:
        return f"{match.group(1)} {match.group(2)}"
    else:
        print("[Error] Invalid lookback format. Defaulting to '1 Y'.")
        return "1 Y"

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
    print("\n=== SMMA Option Strategy Script Starting ===")
    ticker = input("Enter ticker symbol (e.g., TSLA): ").strip().upper()
    lookback = input("Enter lookback period (e.g., 2Y or 180D): ").strip()
    duration_str = parse_duration(lookback)
    today = datetime.now().date()
    print("[Process] Connecting to IBKR Gateway...")
    ib = IB()
    try:
        ib.connect('127.0.0.1', 4001, clientId=1001)
        print("[Result] Connected to IBKR.")
    except Exception as e:
        print(f"[Error] Could not connect to IBKR: {str(e)}")
        return
    exchanges = ['ARCA', 'NASDAQ', 'SMART']

    # For status summary
    buy_actions = []
    sell_actions = []

    position_is_open_daily = False
    daily_contract = None
    position_is_open_4h = False
    contract_4h = None

    # DAILY LOGIC
    print(f"\n--- {ticker} Daily analysis 9/18 SMMA ---")
    bars_daily = None
    for exch in exchanges:
        bars_daily = fetch_bars(ib, ticker, duration_str, "1 day", exch)
        if bars_daily:
            break
    if not bars_daily:
        print(f"[Error] No valid daily data for {ticker} on any supported exchange.")
    else:
        df_daily = util.df(bars_daily)
        df_daily['date'] = pd.to_datetime(df_daily['date'])
        df_daily = df_daily.set_index('date')
        cross_results_daily, events_daily, reentries_daily = analyze_crosses_and_reentries(df_daily, 9, 18, ticker)
        print("[Result] Daily SMMA cross analysis complete.")
        if cross_results_daily:
            for _, line in cross_results_daily:
                print(line)
        else:
            print("[Info] No daily cross events found.")

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
                            if trade:
                                position_is_open_daily = True
                                daily_contract = option_contract
                                buy_actions.append(
                                    f"DAILY {ticker} {option_contract.lastTradeDateOrContractMonth} {option_contract.right} "
                                    f"Strike: {option_contract.strike} [reason: bullish crossover (daily)]"
                                )
                    else:
                        print(f"[Info] Skipping trade for {ticker} at {event['cross_date'].date()} (bull cross): no previous bull high for strike.")

        for idx, bar_date, swing_high, msg in reentries_daily:
            if (today - bar_date.date()).days <= 3 and not position_is_open_daily:
                print("[Signal] " + msg)
                underlying = Stock(ticker, 'SMART', 'USD')
                ib.qualifyContracts(underlying)
                if swing_high is not None:
                    option_contract = select_option_contract(ib, underlying, 45, swing_high)
                    if option_contract:
                        trade = place_order(ib, option_contract, 'BUY', ticker, "reentry after close above slow SMMA (daily)")
                        if trade:
                            position_is_open_daily = True
                            daily_contract = option_contract
                            buy_actions.append(
                                f"DAILY {ticker} {option_contract.lastTradeDateOrContractMonth} {option_contract.right} "
                                f"Strike: {option_contract.strike} [reason: reentry daily]"
                            )
                else:
                    print(f"[Info] Skipping reentry trade for {ticker} at {bar_date.date()}: no swing high for strike.")

        for event in events_daily:
            if event["type"] == "bear" and position_is_open_daily and daily_contract is not None:
                if event.get("prev_bull_high") is not None:
                    trade = place_order(ib, daily_contract, 'SELL', ticker, "bearish crossover (daily)")
                    if trade:
                        position_is_open_daily = False
                        daily_contract = None
                        sell_actions.append(
                            f"DAILY {ticker} {option_contract.lastTradeDateOrContractMonth} {option_contract.right} "
                            f"Strike: {option_contract.strike} [reason: bearish crossover (daily)]"
                        )
                else:
                    print(f"[Info] Skipping sell trade for {ticker} at {event['cross_date'].date()} (bear cross): no previous bull high for strike.")

    # 4-HOUR LOGIC
    print(f"\n--- {ticker} 4-Hour analysis 26/150 SMMA ---")
    bars_4h = None
    for exch in exchanges:
        bars_4h = fetch_bars(ib, ticker, duration_str, "4 hours", exch)
        if bars_4h:
            break
    if not bars_4h:
        print(f"[Error] No valid 4-hour data for {ticker} on any supported exchange.")
    else:
        df_4h = util.df(bars_4h)
        df_4h['date'] = pd.to_datetime(df_4h['date'])
        df_4h = df_4h.set_index('date')
        cross_results_4h, events_4h, reentries_4h = analyze_crosses_and_reentries(df_4h, 26, 150, ticker)
        print("[Result] 4-hour SMMA cross analysis complete.")
        if cross_results_4h:
            for _, line in cross_results_4h:
                print(line)
        else:
            print("[Info] No 4-hour cross events found.")

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
                            if trade:
                                position_is_open_4h = True
                                contract_4h = option_contract
                                buy_actions.append(
                                    f"4-HOUR {ticker} {option_contract.lastTradeDateOrContractMonth} {option_contract.right} "
                                    f"Strike: {option_contract.strike} [reason: bullish crossover (4-hour)]"
                                )
                    else:
                        print(f"[Info] Skipping trade for {ticker} at {event['cross_date'].date()} (bull cross): no previous bull high for strike.")

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
                        if trade:
                            position_is_open_4h = True
                            contract_4h = option_contract
                            buy_actions.append(
                                f"4-HOUR {ticker} {option_contract.lastTradeDateOrContractMonth} {option_contract.right} "
                                f"Strike: {option_contract.strike} [reason: reentry 4-hour]"
                            )
                else:
                    print(f"[Info] Skipping reentry trade for {ticker} at {bar_date.date()}: no swing high for strike.")

        for event in events_4h:
            if event["type"] == "bear" and position_is_open_4h and contract_4h is not None:
                if event.get("prev_bull_high") is not None:
                    trade = place_order(ib, contract_4h, 'SELL', ticker, "bearish crossover (4-hour)")
                    if trade:
                        position_is_open_4h = False
                        contract_4h = None
                        sell_actions.append(
                            f"4-HOUR {ticker} {option_contract.lastTradeDateOrContractMonth} {option_contract.right} "
                            f"Strike: {option_contract.strike} [reason: bearish crossover (4-hour)]"
                        )
                else:
                    print(f"[Info] Skipping sell trade for {ticker} at {event['cross_date'].date()} (bear cross): no previous bull high for strike.")

    ib.disconnect()

    print("\n=========== FINAL STATUS ===========")
    if not buy_actions and not sell_actions:
        print("No trade signals detected—no trades executed.")
    else:
        if buy_actions:
            for detail in buy_actions:
                print(f"[FINAL] Bought: {detail}")
        if sell_actions:
            for detail in sell_actions:
                print(f"[FINAL] Closed/Sold: {detail}")
    print("[Result] Script finished: disconnected from IBKR.")

"""
===============================================================================
STEP-BY-STEP SCRIPT DESCRIPTION AND DIAGNOSTIC NOTES
===============================================================================

1. Prompts user for ticker symbol and lookback period.
2. Connects to IBKR API and prints status.
3. Fetches daily (9/18 SMMA) and 4-hour (26/150 SMMA) bars; analyzes for bullish/bearish crossovers and 'shakeout' reentry signals.
4. For entry:
   - BUY trade is only attempted (and prompted to user) if:
     a) A bullish crossover (daily, last 3 days; 4h, last 8 bars) or...
     b) A reentry event (price closes below then back above slow SMMA in a bullish regime) occurs in the entry window
     c) There is currently NO open position (tracked internally)
   - User always sees intent and prompt for confirmed entries. If no matching contract or signal, details are printed and move on.
5. For exit:
   - SELL is **only processed** IF the script tracks that there is an open position (from prior buy), i.e., after successful entry!
   - Position/broker state is always checked before actual order placement. If SELL is skipped, reason is printed.
6. All IBKR contract errors, skips due to missing data, and position logic are **logged at every step**.
7. Diagnostic: All available expiries and strikes for each contract search are shown if a contract cannot be built.
8. At the end, disconnects from IBKR, prints completion, and prints a summary status:
   - If no buys or sells, prints "No trade signals detected—no trades executed."
   - Else, prints summary for all buys and sells that occurred this run.
9. User confirmation is enforced for every trade and all attempts/skips are visible in output.

===============================================================================
"""

if __name__ == "__main__":
    main()
