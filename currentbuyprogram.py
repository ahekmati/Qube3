import logging
import json
import pytz
import pandas as pd
import numpy as np
from ib_insync import IB, Stock, util, Order



CONFIG_VERSION = "1.0"
ATR_PERIOD = 10
ATR_MULTIPLIER = 2.0
TP_MULTIPLIER = 2.0
SUPERTREND_PERIOD = 10
SUPERTREND_MULTIPLIER = 2.0
SMMA_FAST = 10
SMMA_SLOW = 21
EXCEPTION_TICKERS = {"SQQQ", "VXX", "TLT", "GLD", "SOXS", "SPXS", "FAZ", "SARK"}



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



def supertrend(df, period=10, multiplier=2.0):
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



class State:
    def __init__(self, fname):
        self.fname = fname
        try:
            with open(fname, 'r') as f: self.data = json.load(f)
        except Exception: self.data = {}
        if "positions" not in self.data:
            self.data["positions"] = {}
        if "reentries" not in self.data:
            self.data["reentries"] = {}
        self.data["max_reentries"] = 3
    def save(self):
        with open(self.fname, 'w') as f: json.dump(self.data, f, indent=2)



def fetch_daily_df(ib, symbol, days=120):
    print(f"Fetching daily data for {symbol}...")
    contract = Stock(symbol, 'SMART', 'USD')
    bars = ib.reqHistoricalData(contract, '', f'{days} D', '1 day', 'TRADES', useRTH=True, formatDate=1)
    df = util.df(bars)
    if len(df) == 0:
        raise ValueError('No data for '+symbol)
    print(f"Fetched {len(df)} daily bars for {symbol}.")
    return df



def get_momentum_rvol_score(df, lookback=20):
    if len(df) < lookback + 2:
        return None, None, None
    momentum = (df['close'].iloc[-1] - df['close'].iloc[-lookback]) / df['close'].iloc[-lookback]
    avg_vol = df['volume'].iloc[-lookback-1:-1].mean()
    today_vol = df['volume'].iloc[-1]
    rvol = today_vol / avg_vol if avg_vol > 0 else 0
    score = momentum * 0.6 + rvol * 0.4
    return score, rvol, momentum



def get_trade_size(available_cap, price, max_alloc=0.10):
    raw = int((available_cap * max_alloc) // price)
    return max(raw, 1)



def cancel_existing_stop(ib, symbol):
    contract = Stock(symbol, 'SMART', 'USD')
    for trade in ib.trades():
        if trade.contract.symbol == symbol and trade.order.orderType == 'STP' and trade.order.status not in ('Filled', 'Cancelled'):
            ib.cancelOrder(trade.order)




def main():
    print("Starting trading script...")
    logging.basicConfig(level=logging.INFO)
    ib = IB()
    print("Connecting to IB...")
    ib.connect('127.0.0.1', 4001, clientId=1)
    print("Connected to IB.")
    config = json.load(open('config.json'))
    state = State('state.json')
    if "positions" not in state.data:
        state.data["positions"] = {}
    if "reentries" not in state.data:
        state.data["reentries"] = {}
    state.data["max_reentries"] = 3



    symbols = config['user_tickers']
    max_positions = config.get("max_stocks", 5)
    max_alloc = 0.15
    max_re = state.data["max_reentries"]



    print("Retrieving account balance...")
    account_bal = float([a.value for a in ib.accountSummary() if a.tag == 'NetLiquidation' and a.currency == 'USD'][0])
    print(f"Account balance: ${account_bal:.2f}")



    print("Computing SPY filter (9 SMMA vs 18 SMMA)...")
    spy_df = fetch_daily_df(ib, "SPY")
    spy_df["smma9"] = smma(spy_df['close'], 9)
    spy_df["smma18"] = smma(spy_df['close'], 18)
    spy_9 = spy_df["smma9"].iloc[-1]
    spy_18 = spy_df["smma18"].iloc[-1]
    spy_long_allowed = spy_9 > spy_18
    print(f"SPY 9 SMMA: {spy_9:.2f} | 18 SMMA: {spy_18:.2f} | Long positions allowed for most: {spy_long_allowed}")


    if not spy_long_allowed:
        print("SPY filter triggered! Protective stops activated.")
        for symbol in list(state.data['positions'].keys()):
            if symbol in EXCEPTION_TICKERS:
                continue
            pos = state.data['positions'][symbol]
            try:
                df = fetch_daily_df(ib, symbol)
                recent_low = df['close'].iloc[-5:].min()  # lowest close in last 5 days
                two_pct_below = df['close'].iloc[-1] * 0.98  # 2% below last close
                protective_stop = min(recent_low, two_pct_below)
                cancel_existing_stop(ib, symbol)
                contract = Stock(symbol, 'SMART', 'USD')
                stp_order = Order(
                    action="SELL", 
                    orderType="STP", 
                    auxPrice=round(protective_stop, 2),
                    totalQuantity=pos['shares'],
                    tif='GTC'
                )
                ib.placeOrder(contract, stp_order)
                pos["active_stop"] = protective_stop
                pos["stop_type"] = "protective"
                print(f"Protective stop set for {symbol} at {protective_stop:.2f}")
            except Exception as e:
                logging.error(f"Error setting protective stop for {symbol}: {e}")
                print(f"Error setting protective stop for {symbol}: {e}")



    # Trailing stop update for all open positions
    for symbol, pos in state.data["positions"].items():
        try:
            df = fetch_daily_df(ib, symbol)
            if symbol in EXCEPTION_TICKERS:
                # ATR Trailing Stop for exception tickers
                high_since_entry = pos.get("high_since_entry", pos["entry_price"])
                atr_latest = atr(df, ATR_PERIOD).iloc[-1]
                new_high = max(high_since_entry, df["close"].iloc[-1])
                move = new_high - high_since_entry
                if move >= ATR_MULTIPLIER * atr_latest:
                    # Move stop up by 2 ATR
                    # Stop is new_high - 2*ATR
                    stop_price = new_high - ATR_MULTIPLIER * atr_latest
                    if stop_price > pos.get("active_stop", -np.inf):
                        print(f"ATR trailing stop raised for {symbol} from {pos.get('active_stop', 'None'):.2f} to {stop_price:.2f}")
                        cancel_existing_stop(ib, symbol)
                        contract = Stock(symbol, 'SMART', 'USD')
                        stp_order = Order(action="SELL", orderType="STP", auxPrice=round(stop_price, 2),
                                          totalQuantity=pos['shares'], tif='GTC')
                        ib.placeOrder(contract, stp_order)
                        pos["active_stop"] = stop_price
                        pos["stop_type"] = "atr"
                        pos["high_since_entry"] = new_high
                else:
                    # Update high water mark if price increases
                    pos["high_since_entry"] = new_high
            else:
                # Only update SuperTrend stop if direction is green
                st, direction = supertrend(df, SUPERTREND_PERIOD, SUPERTREND_MULTIPLIER)
                if direction.iloc[-1] == "green":
                    supertrend_stop = float(st.iloc[-1])
                    if supertrend_stop > pos.get("active_stop", -np.inf):
                        print(f"Raising SuperTrend stop for {symbol} from {pos.get('active_stop', 'None'):.2f} to {supertrend_stop:.2f}")
                        cancel_existing_stop(ib, symbol)
                        contract = Stock(symbol, 'SMART', 'USD')
                        stp_order = Order(action="SELL", orderType="STP", auxPrice=round(supertrend_stop, 2),
                                          totalQuantity=pos['shares'], tif='GTC')
                        ib.placeOrder(contract, stp_order)
                        pos["active_stop"] = supertrend_stop
                        pos["stop_type"] = "supertrend"
                else:
                    print(f"SuperTrend is red for {symbol}; stop not updated.")
        except Exception as e:
            print(f"Failed stop update for {symbol}: {e}")



    qualified = []
    for symbol in symbols:
        if symbol in state.data['positions']:
            continue
        if not spy_long_allowed and symbol not in EXCEPTION_TICKERS:
            continue
        try:
            df = fetch_daily_df(ib, symbol)
            df["smma_fast"] = smma(df['close'], SMMA_FAST)
            df["smma_slow"] = smma(df['close'], SMMA_SLOW)
            curr, prev = -1, -2
            cross_bull = (
                df["smma_fast"].iloc[prev] <= df["smma_slow"].iloc[prev] and
                df["smma_fast"].iloc[curr] > df["smma_slow"].iloc[curr]
            )
            above_slow = df["smma_fast"].iloc[curr] > df["smma_slow"].iloc[curr]
            recross_above_slow = (
                above_slow and
                df["close"].iloc[prev] < df["smma_slow"].iloc[prev] and
                df["close"].iloc[curr] > df["smma_slow"].iloc[curr]
            )
            reentries = state.data['reentries'].get(symbol, 0)
            if cross_bull or recross_above_slow:
                score, rvol, momentum = get_momentum_rvol_score(df)
                if score is not None:
                    qualified.append({'symbol': symbol, 'score': score, 'rvol': rvol,
                                      'momentum': momentum, 'price': float(df["close"].iloc[curr]),
                                      'df': df, 'reentries': reentries})
        except Exception as e:
            print(f"Error scanning {symbol} for entry: {e}")



    qualified_sorted = sorted(qualified, key=lambda x: x['score'], reverse=True)
    top_qualified = qualified_sorted[:max_positions]
    print("\nThe top 5 momentum ranking that also meet the criteria are as follows:")
    for i, q in enumerate(top_qualified):
        print(f"{i+1}. {q['symbol']} | Score={q['score']:.3f} | RVOL={q['rvol']:.2f} | Momentum={q['momentum']:.3f}")



    for q in top_qualified:
        if len(state.data['positions']) >= max_positions:
            print("Max positions reached.")
            break
        if q['reentries'] < max_re:
            df = q['df']
            if q['symbol'] in EXCEPTION_TICKERS:
                # ATR-based stop for exception ticker
                atr_val = atr(df, ATR_PERIOD).iloc[-1]
                stop_price = q['price'] - ATR_MULTIPLIER * atr_val
                size = get_trade_size(account_bal, q['price'], max_alloc)
                contract = Stock(q['symbol'], 'SMART', 'USD')
                ib.qualifyContracts(contract)
                parent_ord = Order(action="BUY", orderType="MKT", totalQuantity=size, tif='DAY', transmit=False)
                ib.placeOrder(contract, parent_ord)
                tp_ord = Order(action="SELL", orderType="LMT", totalQuantity=size, lmtPrice=round(q['price'] + TP_MULTIPLIER * atr_val, 2),
                               parentId=parent_ord.orderId, tif='GTC', transmit=False)
                stp_ord = Order(action="SELL", orderType="STP", totalQuantity=size, auxPrice=round(stop_price, 2),
                                parentId=parent_ord.orderId, tif='GTC', transmit=True)
                ib.placeOrder(contract, tp_ord)
                ib.placeOrder(contract, stp_ord)
                print(f"Market bracket order (ATR stop) placed for {q['symbol']}, waiting for execution...")
                state.data['positions'][q['symbol']] = {
                    'entry_price': q['price'],
                    'shares': size,
                    'init_stop': stop_price,
                    'active_stop': stop_price,
                    'take_profit': q['price'] + TP_MULTIPLIER * atr_val,
                    'stop_type': "atr",
                    'cross_date': str(df.index[-1]),
                    'high_since_entry': q['price']
                }
            else:
                # SuperTrend-based stop for normal ticker (initial)
                st, direction = supertrend(df, SUPERTREND_PERIOD, SUPERTREND_MULTIPLIER)
                supertrend_stop = float(st.iloc[-1])
                size = get_trade_size(account_bal, q['price'], max_alloc)
                contract = Stock(q['symbol'], 'SMART', 'USD')
                ib.qualifyContracts(contract)
                parent_ord = Order(action="BUY", orderType="MKT", totalQuantity=size, tif='DAY', transmit=False)
                ib.placeOrder(contract, parent_ord)
                tp_ord = Order(action="SELL", orderType="LMT", totalQuantity=size, lmtPrice=round(q['price'] + TP_MULTIPLIER * atr(df, ATR_PERIOD).iloc[-1], 2),
                               parentId=parent_ord.orderId, tif='GTC', transmit=False)
                stp_ord = Order(action="SELL", orderType="STP", totalQuantity=size, auxPrice=round(supertrend_stop, 2),
                                parentId=parent_ord.orderId, tif='GTC', transmit=True)
                ib.placeOrder(contract, tp_ord)
                ib.placeOrder(contract, stp_ord)
                print(f"Market bracket order placed for {q['symbol']}, waiting for execution...")
                state.data['positions'][q['symbol']] = {
                    'entry_price': q['price'],
                    'shares': size,
                    'init_stop': supertrend_stop,
                    'active_stop': supertrend_stop,
                    'take_profit': q['price'] + TP_MULTIPLIER * atr(df, ATR_PERIOD).iloc[-1],
                    'stop_type': "supertrend",
                    'cross_date': str(df.index[-1])
                }
            state.data['reentries'][q['symbol']] = q['reentries'] + 1
            logging.info(f"BUY {q['symbol']} {size} @ {q['price']:.2f}, TP {state.data['positions'][q['symbol']]['take_profit']:.2f}, stop {state.data['positions'][q['symbol']]['active_stop']:.2f}")
            print(f"Entry position recorded for {q['symbol']}.")



    state.save()
    print("State saved.")
    logging.info("End of trading cycle.")



    positions = ib.positions()
    trades = ib.trades()
    stop_orders = {}
    for trade in trades:
        order = trade.order
        contract = trade.contract
        if order.orderType == 'STP' and order.tif == 'GTC' and order.action == 'SELL':
            stop_orders[contract.symbol] = order.auxPrice


    print("\nCurrent Open Positions and Their Stop Loss Levels:")
    print(f"{'Symbol':6} | {'Qty':>6} | {'AvgPx':>8} | {'CurPx':>8} | {'StopLoss(GTC)':>13}")
    print("-" * 60)


    for pos in positions:
        contract = pos.contract
        symbol = contract.symbol
        qty = pos.position
        avg_cost = pos.avgCost
        stop = stop_orders.get(symbol, "None")


        # Fetch current market price
        market_data = ib.reqMktData(contract, '', False, False)
        ib.sleep(1)  # give IB a moment to fill in values
        cur_price = market_data.marketPrice()
        # fallback to last price if market price isn't available
        if np.isnan(cur_price):
            cur_price = market_data.last
        if np.isnan(cur_price):
            cur_price = "N/A"
        else:
            cur_price = f"{cur_price:.2f}"


        print(f"{symbol:6} | {qty:6} | {avg_cost:8.2f} | {cur_price:8} | {stop:>13}")


    print("Trading cycle complete. Disconnecting IB...")
    ib.disconnect()
    print("Disconnected from IB.")



if __name__ == "__main__":
    main()
