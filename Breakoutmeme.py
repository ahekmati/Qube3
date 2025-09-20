import pandas as pd
import numpy as np
import json
import csv
import time
from datetime import datetime
from ib_insync import IB, util, Stock, ScannerSubscription

STATE_FILE = "positions_state.json"
TRADELOG = "trade_log.csv"
MAX_STOCKS = 10
POSITION_ALLOC = 0.05
ATR_MULT = 2

def get_live_account_balance(ib):
    summary = ib.accountSummary()
    for x in summary:
        if x.tag == 'NetLiquidation' and x.currency == 'USD':
            return float(x.value)
    return 2500.00  # fallback

def save_state(positions, filename=STATE_FILE):
    with open(filename, "w") as f:
        json.dump(positions, f)

def load_state(filename=STATE_FILE):
    try:
        with open(filename, "r") as f:
            return json.load(f)
    except Exception:
        return {}

def log_trade_csv(symbol, action, shares, price, trailing_stop, reason="", stop_order_id=None):
    with open(TRADELOG, "a", newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([
            datetime.now().isoformat(), symbol, action, shares,
            f"{price:.2f}", f"{trailing_stop:.2f}", reason, stop_order_id
        ])

def notify_trade(symbol, action, shares, price, trailing_stop, reason="", stop_order_id=None):
    line = (f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | "
            f"{action} {shares} {symbol} @ ${price:.2f} | "
            f"Trailing/Stop: ${trailing_stop:.2f} | IBKR Stop Order ID: {stop_order_id} "
            f"{('('+reason+')') if reason else ''}")
    print(line)
    log_trade_csv(symbol, action, shares, price, trailing_stop, reason, stop_order_id)

def calculate_atr(df, window=14):
    high_low = df['high'] - df['low']
    high_close = np.abs(df['high'] - df['close'].shift(1))
    low_close = np.abs(df['low'] - df['close'].shift(1))
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    atr = tr.rolling(window).mean()
    return atr

def fetch_ohlcv(ib, symbol):
    contract = Stock(symbol, 'SMART', 'USD')
    bars = ib.reqHistoricalData(
        contract,
        endDateTime='',
        durationStr='60 D',
        barSizeSetting='1 day',
        whatToShow='TRADES',
        useRTH=True,
        formatDate=1)
    if not bars:
        return None
    df = util.df(bars)
    df = df[['date','open','high','low','close','volume']]
    df['date'] = pd.to_datetime(df['date'])
    df.set_index('date', inplace=True)
    df.rename(columns={'open':'Open','high':'High','low':'Low','close':'Close','volume':'Volume'}, inplace=True)
    df.columns = df.columns.str.lower()
    return df

def scan_for_stocks(ib):
    scan_sub = ScannerSubscription(
        instrument='STK',
        locationCode='STK.US.MAJOR',
        scanCode='TOP_PERC_GAIN'
    )
    scan_results = ib.reqScannerData(scan_sub, [
        dict(tag='priceAbove', value='3'),
        dict(tag='volumeAbove', value='100000'),
        dict(tag='avgVolumeAbove', value='100000'),
        dict(tag='marketCapAbove', value='500000000')
    ])
    time.sleep(3)
    tickers = [x.contract.symbol for x in scan_results[:MAX_STOCKS]]
    return tickers

def place_ibkr_market_order(ib, symbol, shares):
    contract = Stock(symbol, 'SMART', 'USD')
    order = ib.marketOrder('BUY', shares)
    t = ib.placeOrder(contract, order)
    ib.sleep(3)
    print(f"Placed BUY order for {shares} shares of {symbol}.")
    return int(t.order.permId) if t.order.permId else None

def place_ibkr_stop_order(ib, symbol, shares, stop_price):
    contract = Stock(symbol, 'SMART', 'USD')
    stop_order = ib.stopOrder('SELL', shares, round(stop_price,2))
    t = ib.placeOrder(contract, stop_order)
    ib.sleep(3)
    print(f"Placed STOP order for {shares} shares of {symbol} at ${stop_price:.2f}.")
    return int(t.order.permId) if t.order.permId else None

def modify_ibkr_stop_order(ib, symbol, stop_order_id, shares, new_stop_price):
    # For true production, must track/cancel/replace STOP orders. This is a stub.
    print(f"Modify stop for {symbol}: New stop ${new_stop_price:.2f} (order_id {stop_order_id})")
    # You can use ib.cancelOrder() and place a new one.

def cancel_ibkr_order(ib, order_id):
    print(f"Cancelling IBKR order ID {order_id}.")

def check_and_trade_positions(ib, open_positions, candidates, alloc_per_stock):
    for symbol in candidates:
        if symbol in open_positions: continue
        df = fetch_ohlcv(ib, symbol)
        if df is None or len(df)<21: continue
        df['atr'] = calculate_atr(df)
        df['20d_high'] = df['close'].rolling(20).max()
        price = df.iloc[-1]['close']
        recent_20d_low = df['low'].rolling(20).min().iloc[-1]
        ma5 = df['close'].rolling(5).mean().iloc[-1]
        if price > recent_20d_low * 1.3:
            print(f"SKIP {symbol}: too extended above 20-day low.")
            continue
        if price > ma5 * 1.15:
            print(f"SKIP {symbol}: too extended above 5-day avg.")
            continue
        if price > df.iloc[-2]['20d_high']:
            if price <= 3: continue
            shares = int(alloc_per_stock // price)
            if shares == 0: continue
            atr = df.iloc[-1]['atr']
            trailing_stop = price - ATR_MULT * atr
            buy_order_id = place_ibkr_market_order(ib, symbol, shares)
            stop_order_id = place_ibkr_stop_order(ib, symbol, shares, trailing_stop)
            open_positions[symbol] = {
                "entry_price": float(price),
                "shares": shares,
                "entry_date": str(df.index[-1]),
                "high_since_entry": float(price),
                "trailing_stop": float(trailing_stop),
                "last_atr": float(atr),
                "buy_order_id": buy_order_id,
                "stop_order_id": stop_order_id
            }
            notify_trade(symbol, "BUY", shares, price, trailing_stop, stop_order_id=stop_order_id)
    # Trailing stops & exits
    to_close = []
    for symbol in list(open_positions.keys()):
        df = fetch_ohlcv(ib, symbol)
        if df is None or len(df) < 2: continue
        pos = open_positions[symbol]
        since_entry = df[df.index >= pos['entry_date']]
        if since_entry.empty: continue
        recent_high = since_entry['close'].max()
        last_close = df.iloc[-1]['close']
        atr = calculate_atr(df).iloc[-1]
        if recent_high > pos['high_since_entry']:
            pos['high_since_entry'] = float(recent_high)
            new_stop = float(recent_high - ATR_MULT * atr)
            pos['trailing_stop'] = new_stop
            pos['last_atr'] = float(atr)
            modify_ibkr_stop_order(ib, symbol, pos['stop_order_id'], pos['shares'], new_stop)
        if last_close <= pos['trailing_stop']:
            notify_trade(symbol, "SELL", pos['shares'], last_close, pos['trailing_stop'],
                         reason="stop hit", stop_order_id=pos['stop_order_id'])
            cancel_ibkr_order(ib, pos['stop_order_id'])
            to_close.append(symbol)
    for symbol in to_close:
        del open_positions[symbol]
    return open_positions

def main():
    ib = IB()
    ib.connect('127.0.0.1', 7497, clientId=1)
    open_positions = load_state()
    print("Restored open positions:", open_positions)
    need_new_picks = len(open_positions) < MAX_STOCKS
    candidates = []
    if need_new_picks:
        candidates = scan_for_stocks(ib)
    if need_new_picks:
        picks = [x for x in candidates if x not in open_positions]
        picks = picks[:MAX_STOCKS - len(open_positions)]
    else:
        picks = []
    capital = get_live_account_balance(ib)
    alloc_per_stock = capital * POSITION_ALLOC
    open_positions = check_and_trade_positions(ib, open_positions, picks, alloc_per_stock)
    save_state(open_positions)
    print("Positions after update:", open_positions)
    ib.disconnect()

if __name__ == "__main__":
    main()
