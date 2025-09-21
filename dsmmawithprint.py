import logging, os, json, csv, threading, time
from datetime import datetime, timedelta
import pytz
import pandas as pd
import numpy as np
from ib_insync import IB, Stock, util

CONFIG_VERSION = "1.0"

def smma(series, window):
    s = pd.Series(series)
    if len(s) < window:
        return s.copy()
    result = s.copy()
    result.iloc[:window] = s.iloc[:window].mean()
    for i in range(window, len(result)):
        result.iloc[i] = (result.iloc[i-1]*(window-1) + s.iloc[i])/window
    return result

def atr(df, window=14):
    high_low = df['high'] - df['low']
    high_close = np.abs(df['high'] - df['close'].shift(1))
    low_close  = np.abs(df['low'] - df['close'].shift(1))
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    return tr.rolling(window).mean()

def supertrend(df, period=22, multiplier=2):
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

def get_trade_size(available_cap, price, max_alloc=0.10):
    raw = int((available_cap * max_alloc) // price)
    return max(raw, 1)

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
    tz = pytz.timezone(config.get("trade_timezone", "US/Eastern"))

    print("Retrieving account balance...")
    account_bal = float([a.value for a in ib.accountSummary() if a.tag == 'NetLiquidation' and a.currency == 'USD'][0])
    print(f"Account balance: ${account_bal:.2f}")
    open_pos = list(state.data['positions'].keys())
    num_live = len(open_pos)

    print(f"Scanning tickers: {symbols}")
    for symbol in symbols:
        print(f"\nChecking ticker: {symbol}")
        if len(state.data['positions']) >= max_positions: 
            print("Max positions reached.")
            break
        try:
            df = fetch_daily_df(ib, symbol)
            print(f"Calculating indicators for {symbol}...")
            df["smma12"] = smma(df['close'], 12)
            df["smma40"] = smma(df['close'], 40)
            df["atr"] = atr(df)
            df["supertrend"], df["superdir"] = supertrend(df)
            curr, prev = -1, -2
            inpos = symbol in state.data['positions']
            cross_bull = df["smma12"].iloc[prev] <= df["smma40"].iloc[prev] and df["smma12"].iloc[curr] > df["smma40"].iloc[curr]
            above40 = df["smma12"].iloc[curr] > df["smma40"].iloc[curr]
            price = float(df["close"].iloc[curr])
            stop = df["smma40"].iloc[curr] - df["atr"].iloc[curr]
            reentries = state.data['reentries'].get(symbol, 0)
            print(f"{symbol}: inpos={inpos}, price={price:.2f}, stop={stop:.2f}, reentries={reentries}")

            recross_above_40 = (
                above40 and
                df["close"].iloc[prev] < df["smma40"].iloc[prev] and
                df["close"].iloc[curr] > df["smma40"].iloc[curr]
            )

            if not inpos and (cross_bull or recross_above_40):
                print(f"Trade entry signal detected for {symbol} (cross_bull={cross_bull}, recross_above_40={recross_above_40}).")
                if reentries < max_re:
                    size = get_trade_size(account_bal, price, max_alloc)
                    print(f"Placing buy order for {symbol}: {size} shares at ${price:.2f}")
                    order = ib.placeOrder(Stock(symbol, 'SMART', 'USD'), ib.marketOrder('BUY', size))
                    print(f"BUY order placed for {symbol}, waiting for execution...")
                    state.data['positions'][symbol] = {
                        'entry_price': price,
                        'shares': size,
                        'init_stop': stop,
                        'active_stop': stop,
                        'stop_type': "init",
                        'cross_date': str(df.index[curr]),
                        'supertrend_last': float(df["supertrend"].iloc[curr]),
                        'superdir_last': df["superdir"].iloc[curr]
                    }
                    state.data['reentries'][symbol] = reentries + 1
                    logging.info(f"BUY {symbol} {size} @ {price:.2f}, stop {stop:.2f} (cross_bull={cross_bull}, recross_above_40={recross_above_40})")
                    print(f"Entry position recorded for {symbol}.")

            # POSITION MANAGEMENT (Trailing and Stopping)
            if inpos:
                pos = state.data['positions'][symbol]
                prev_superdir = pos['superdir_last']
                curr_superdir = df["superdir"].iloc[curr]
                curr_st = float(df["supertrend"].iloc[curr])
                stop = pos['active_stop']
                got_stopped = False

                # Update trailing stop if supertrend flips from red to green again
                if prev_superdir == "red" and curr_superdir == "green":
                    print(f"Supertrend flip detected for {symbol}: stop moved from {stop:.2f} to {curr_st:.2f}")
                    stop = curr_st
                    pos['active_stop'] = stop
                    pos['stop_type'] = "supertrend"
                    logging.info(f"TRAIL {symbol}: Trailing stop updated to SuperTrend {stop:.2f}")

                # Check for stop out
                if price < stop:
                    print(f"Price dropped below stop for {symbol}! Selling {pos['shares']} shares at ${price:.2f}.")
                    size = pos['shares']
                    order = ib.placeOrder(Stock(symbol, 'SMART', 'USD'), ib.marketOrder('SELL', size))
                    print(f"SELL order placed for {symbol}, waiting for execution...")
                    logging.info(f"SELL {symbol} {size} @ {price:.2f} STOPPED OUT.")
                    del state.data['positions'][symbol]
                    got_stopped = True
                    print(f"Stopped out of {symbol}.")

                if got_stopped:
                    if above40:
                        print(f"{symbol} remains in bullish regime after stop. Ready for potential reentry.")
                    else:
                        state.data['reentries'][symbol] = 0
                        print(f"{symbol} left bullish regime. Reentry counter reset.")
                else:
                    pos['supertrend_last'] = curr_st
                    pos['superdir_last'] = curr_superdir

        except Exception as e:
            logging.error(f"{symbol}: {e}")
            print(f"Error processing {symbol}: {e}")

    state.save()
    print("State saved.")
    logging.info("End of trading cycle.")
    print("Trading cycle complete. Disconnecting IB...")
    ib.disconnect()
    print("Disconnected from IB.")

if __name__ == "__main__":
    main()
