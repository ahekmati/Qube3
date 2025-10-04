from ib_insync import IB, Stock, Option, util, Order
import numpy as np
import pandas as pd
import os
import datetime

TRADES_FILE = 'open_option_trades.csv'
IB_PORT = 4001
CLIENT_ID = 1
SMART_EXCH = 'SMART'
CASH_CURR = 'USD'
SIGNAL_WINDOW_DAYS = 14

def calc_smma(prices, period):
    smma = [np.mean(prices[:period])]
    for i in range(period, len(prices)):
        prev = smma[-1]
        curr = ((prev * (period - 1)) + prices[i]) / period
        smma.append(curr)
    return np.array([np.nan] * (period - 1) + smma)

def prompt_tickers_params():
    tickers = input("Tickers (comma separated): ").strip().upper().split(",")
    tickers = [x.strip() for x in tickers if x.strip()]
    ticker_params = {}
    for ticker in tickers:
        print(f"\n--- Ticker: {ticker} ---")
        fast = int(input('SMMA Fast period: '))
        slow = int(input('SMMA Slow period: '))
        ticker_params[ticker] = {'fast': fast, 'slow': slow}
    timeframe = input('\nTimeframe (e.g. "1d", "1h", "5 mins"): ').strip()
    lookback_years = float(input('Lookback (years): '))
    return tickers, ticker_params, timeframe, lookback_years

def fetch_history(ib, ticker, timeframe, lookback_years):
    contract = Stock(ticker, SMART_EXCH, CASH_CURR)
    endDateTime = ''
    durationStr = f'{int(lookback_years)} Y' if lookback_years >= 1 else f'{int(lookback_years*365)} D'
    bars = ib.reqHistoricalData(contract, endDateTime, durationStr, timeframe, 'TRADES', useRTH=True, formatDate=1)
    if not bars:
        return None
    df = util.df(bars)
    if df is None or df.empty:
        return None
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date')
    return df

def get_atm_option(ib, ticker, expiry, right='C'):
    stk_contract = Stock(ticker, SMART_EXCH, CASH_CURR)
    [ib.qualifyContracts(stk_contract)]
    market = ib.reqMktData(stk_contract, '', False, False)
    ib.sleep(2)
    last = market.last if market.last else market.close
    chains = ib.reqSecDefOptParams(ticker, '', SMART_EXCH, stk_contract.conId)
    if not chains:
        raise Exception("No option chain for %s" % ticker)
    strikes = sorted([x for x in chains[0].strikes if abs(x - last) < last*0.1])
    if not strikes:
        raise Exception("No nearby strikes found for %s" % ticker)
    atm_strike = min(strikes, key=lambda x: abs(x - last))
    expiry_fmt = pd.to_datetime(expiry).strftime('%Y%m%d')
    optc = Option(ticker, expiry_fmt, atm_strike, right, SMART_EXCH, tradingClass=stk_contract.symbol)
    ib.qualifyContracts(optc)
    return optc

def place_bracket_order(ib, contract, quantity, stop_loss_pct, profit_tgt_pct):
    market = ib.reqMktData(contract, '', False, False)
    ib.sleep(2)
    entry_price = market.last if market.last else (market.close if market.close else 1.0)
    limit_price = round(entry_price * (1 + profit_tgt_pct/100), 2)
    stop_price = round(entry_price * (1 - stop_loss_pct/100), 2)
    parent = Order(action='BUY', orderType='MKT', totalQuantity=quantity, transmit=False)
    take_profit = Order(action='SELL', orderType='LMT', totalQuantity=quantity, lmtPrice=limit_price,
                        parentId=parent.orderId, transmit=False)
    stop_loss = Order(action='SELL', orderType='STP', totalQuantity=quantity, auxPrice=stop_price,
                      parentId=parent.orderId, transmit=True)
    return parent, take_profit, stop_loss

def save_trade(trade):
    df = pd.DataFrame([trade])
    if not os.path.isfile(TRADES_FILE):
        df.to_csv(TRADES_FILE, index=False)
    else:
        df.to_csv(TRADES_FILE, mode='a', header=False, index=False)

def load_trades():
    if not os.path.isfile(TRADES_FILE):
        return pd.DataFrame()
    return pd.read_csv(TRADES_FILE, parse_dates=['entry_date', 'expiry_date'])

def mark_trade_closed(trade_id):
    df = load_trades()
    df.loc[df['trade_id'] == trade_id, 'active'] = False
    df.to_csv(TRADES_FILE, index=False)

def analyze_crossovers(df, fast, slow):
    closes = df['close'].values
    smma_fast = calc_smma(closes, fast)
    smma_slow = calc_smma(closes, slow)
    dates = df['date'].values
    today = pd.Timestamp.now()
    cutoff = today - pd.Timedelta(days=SIGNAL_WINDOW_DAYS)
    cross_dates = []
    cross_to_high_days = []
    cross_idxs = []
    for i in range(1, len(closes)):
        if smma_fast[i-1] <= smma_slow[i-1] and smma_fast[i] > smma_slow[i]:
            cross_date = pd.to_datetime(dates[i])
            if cross_date >= cutoff:
                cross_idxs.append(i)
                cross_dates.append(cross_date)
                idx_close_below = None
                for j in range(i+1, len(closes)):
                    if closes[j] < smma_slow[j]:
                        idx_close_below = j
                        break
                search_end = idx_close_below if idx_close_below else len(closes)
                if search_end > i:
                    segment = closes[i:search_end]
                    max_idx = np.argmax(segment)
                    idx_high = i + max_idx
                    cross_to_high_days.append((df['date'].iloc[idx_high] - df['date'].iloc[i]).days)
    avg_days = np.mean(cross_to_high_days) if cross_to_high_days else 10
    return cross_dates, int(round(avg_days))

def monitor_open_trades(ib):
    print("\n=== ACTIVE TRADES MONITORING ===")
    df = load_trades()
    today = pd.Timestamp.now()
    any_alerts = False
    if df.empty:
        print("No open trades found.")
        return
    for idx, row in df[df['active'] == True].iterrows():
        entry = row['entry_date']
        expiry = row['expiry_date']
        days_held = (today - entry).days
        total_days = row['avg_days']
        days_left = int((expiry - today).days)
        print("-"*60)
        print(f"Trade ID: {row['trade_id']}")
        print(f"  Ticker: {row['ticker']}")
        print(f"  Contract: {row['contract']}")
        print(f"  Entry Date: {entry.strftime('%Y-%m-%d')}")
        print(f"  Expiry Target: {expiry.strftime('%Y-%m-%d')}")
        print(f"  Days Held: {days_held}")
        print(f"  Max (Avg) Hold Days: {total_days}")
        print(f"  Days Remaining: {max(0, days_left)}")
        if days_left <= 0:
            print("  --> ALERT: EXCEEDED TARGET HOLD DAYS! <--")
            any_alerts = True
    # Prompt to close expired trades
    for idx, row in df[df['active'] == True].iterrows():
        expiry = row['expiry_date']
        days_left = int((expiry - today).days)
        if days_left <= 0:
            ans = input(f"Trade {row['trade_id']} reached target days. Market close now? (y/n): ")
            if ans.lower() == 'y':
                # Submit market sell order
                sec_fields = eval(row['contract'].replace('Option', 'dict').replace('=', ':'))
                contract = Option(row['ticker'], sec_fields['lastTradeDateOrContractMonth'],
                                  float(sec_fields['strike']), sec_fields['right'],
                                  SMART_EXCH, tradingClass=row['ticker'])
                ib.qualifyContracts(contract)
                close_order = Order(action='SELL', orderType='MKT', totalQuantity=row['quantity'])
                print(f"Closing {row['trade_id']} ...")
                ib.placeOrder(contract, close_order)
                mark_trade_closed(row['trade_id'])
                print("Closed and logged.")
    if not any_alerts:
        print("No trades at or past their max hold yet.")

def main():
    ib = IB()
    ib.connect('127.0.0.1', IB_PORT, clientId=CLIENT_ID)
    # Monitor old trades
    monitor_open_trades(ib)

    # New signals
    tickers, ticker_params, timeframe, lookback_years = prompt_tickers_params()
    for ticker in tickers:
        print(f"\n--- {ticker} ---")
        fast = ticker_params[ticker]['fast']
        slow = ticker_params[ticker]['slow']
        df = fetch_history(ib, ticker, timeframe, lookback_years)
        if df is None or df.empty:
            print("No historical data for", ticker)
            continue
        cross_dates, avg_days = analyze_crossovers(df, fast, slow)
        if not cross_dates:
            print(f"No SMMA cross detected in last {SIGNAL_WINDOW_DAYS} days for {ticker}")
            continue
        for last_cross in cross_dates:
            print(f"Recent crossover: {last_cross.strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"Average days-to-high: {avg_days}")
            user = input(f"Enter ATM call for {ticker}? Expire {avg_days} days? (y/n): ")
            if user.lower() == 'y':
                stop = float(input("Stop loss %: "))
                tgt  = float(input("Profit target %: "))
                qty  = int(input("Contracts: "))
                expiry = (last_cross + pd.Timedelta(days=avg_days))

                opt_contract = get_atm_option(ib, ticker, expiry)
                print(f"ATM strike to be bought: {opt_contract.strike}, Expiry: {expiry.strftime('%Y-%m-%d')}")
                parent, take_profit, stop_loss = place_bracket_order(ib, opt_contract, qty, stop, tgt)
                print("Placing bracket order...")
                ib.placeOrder(parent.orderId, opt_contract, parent)
                ib.placeOrder(take_profit.orderId, opt_contract, take_profit)
                ib.placeOrder(stop_loss.orderId, opt_contract, stop_loss)
                trade_data = {
                    'trade_id': f"{ticker}_{last_cross.strftime('%Y%m%d%H%M')}",
                    'ticker': ticker,
                    'contract': str(opt_contract),
                    'entry_date': pd.Timestamp.now(),
                    'expiry_date': expiry,
                    'quantity': qty,
                    'active': True,
                    'stop_loss_pct': stop,
                    'profit_target_pct': tgt,
                    'avg_days': avg_days
                }
                save_trade(trade_data)
                print("Order placed and trade logged.")
    ib.disconnect()

if __name__ == '__main__':
    main()
