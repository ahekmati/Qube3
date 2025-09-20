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
    contract = Stock(symbol, 'SMART', 'USD')
    bars = ib.reqHistoricalData(contract, '', f'{days} D', '1 day', 'TRADES', useRTH=True, formatDate=1)
    df = util.df(bars)
    if len(df) == 0: raise ValueError('No data for '+symbol)
    return df

def get_trade_size(available_cap, price, max_alloc=0.15):
    raw = int((available_cap * max_alloc) // price)
    return max(raw, 1)

def main():
    logging.basicConfig(level=logging.INFO)
    ib = IB()
    ib.connect('127.0.0.1', 4001, clientId=1)
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

    account_bal = float([a.value for a in ib.accountSummary() if a.tag == 'NetLiquidation' and a.currency == 'USD'][0])
    open_pos = list(state.data['positions'].keys())
    num_live = len(open_pos)

    for symbol in symbols:
        if len(state.data['positions']) >= max_positions: break
        try:
            df = fetch_daily_df(ib, symbol)
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

            # --- Track price recross for pullback entry logic
            recross_above_40 = (
                # only if in bullish regime
                above40 and
                # previous day closed below the 40 SMMA
                df["close"].iloc[prev] < df["smma40"].iloc[prev] and
                # current day closed above
                df["close"].iloc[curr] > df["smma40"].iloc[curr]
            )

            # ENTRY: Either first cross OR price recross above 40 during bullish regime
            if not inpos and (cross_bull or recross_above_40):
                if reentries < max_re:
                    size = get_trade_size(account_bal, price, max_alloc)
                    order = ib.placeOrder(Stock(symbol, 'SMART', 'USD'), ib.marketOrder('BUY', size))
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
                    stop = curr_st
                    pos['active_stop'] = stop
                    pos['stop_type'] = "supertrend"
                    logging.info(f"TRAIL {symbol}: Trailing stop updated to SuperTrend {stop:.2f}")

                # Check for stop out
                if price < stop:
                    size = pos['shares']
                    order = ib.placeOrder(Stock(symbol, 'SMART', 'USD'), ib.marketOrder('SELL', size))
                    logging.info(f"SELL {symbol} {size} @ {price:.2f} STOPPED OUT.")
                    del state.data['positions'][symbol]
                    got_stopped = True

                # If stopped out, allow possible re-entry per regime logic
                if got_stopped:
                    if above40:
                        pass
                    else:
                        state.data['reentries'][symbol] = 0

                else:
                    pos['supertrend_last'] = curr_st
                    pos['superdir_last'] = curr_superdir

        except Exception as e:
            logging.error(f"{symbol}: {e}")

    state.save()
    logging.info("End of trading cycle.")
    ib.disconnect()

if __name__ == "__main__":
    main()
