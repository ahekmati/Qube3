from ib_insync import *
import pandas as pd
import numpy as np

# --- CONFIG ---
tickers = ['TQQQ', 'SSO']
exchange = 'ARCA'
currency = 'USD'
period = '60 D'
candle_sizes = ['8 hours']
ema_fast = 20
ema_slow = 50
ema_short_slow = 70
z_window = 20
z_thresh_long = -0.5
z_thresh_short = 0.7
take_profit_pct = .12      # +12% Take Profit
stop_loss_pct = .04        # -4% Stop Loss
max_hold = 50
quantity = 100             # Position size for live order
signal_max_age = 10        # Only signals in last N candles result in trade
backtest_start = '2025-03-01'
backtest_end = '2025-10-29'

ib = IB()
ib.connect('127.0.0.1', 4001, clientId=1)

def get_data(symbol, size, period):
    contract = Stock(symbol, exchange, currency)
    bars = ib.reqHistoricalData(
        contract,
        endDateTime='',
        durationStr=period,
        barSizeSetting=size,
        whatToShow='TRADES',
        useRTH=False,
        formatDate=1
    )
    df = util.df(bars)
    df.set_index('date', inplace=True)
    df = df.sort_index()
    df['close'] = df['close'].astype(float)
    return df

def compute_signals(df):
    # Long logic
    df['ema_fast'] = df['close'].ewm(span=ema_fast, adjust=False).mean()
    df['ema_slow'] = df['close'].ewm(span=ema_slow, adjust=False).mean()
    std20 = df['close'].rolling(z_window).std()
    dip_z = (df['close'] - df['ema_fast']) / std20
    dip_signal = dip_z < z_thresh_long
    buy_signals = (df['close'] < df['ema_slow']) & dip_signal
    df['buy_the_dip'] = buy_signals.fillna(False)
    
    # Short logic
    df['ema_70'] = df['close'].ewm(span=ema_short_slow, adjust=False).mean()
    rally_z = (df['close'] - df['ema_fast']) / std20
    short_signal = (
        (df['ema_fast'] < df['ema_70'])
        & (df['close'] > df['ema_70'])
        & (rally_z >= z_thresh_short)
    )
    df['sell_the_rally'] = short_signal.fillna(False)
    
    # Compose universal signal column for easy processing
    df['signal'] = ''
    in_long = False
    in_short = False
    i = 0
    while i < len(df):
        idx = df.index[i]
        long_sig = df.at[idx, 'buy_the_dip']
        short_sig = df.at[idx, 'sell_the_rally']
        if not in_long and long_sig:
            entry_price = df.at[idx, 'close']
            df.at[idx, 'signal'] = 'BUY'
            in_long = True
            # exit logic as before
            exit_found = False
            for j in range(i+1, min(i+max_hold+1, len(df))):
                tp_price = entry_price * (1 + take_profit_pct)
                closej = df['close'].iloc[j]
                idxj = df.index[j]
                if closej >= tp_price:
                    df.at[idxj, 'signal'] = 'SELL (TP)'
                    i = j + 1
                    in_long = False
                    exit_found = True
                    break
            if not exit_found and (i+max_hold < len(df)):
                idxj = df.index[i+max_hold]
                df.at[idxj, 'signal'] = 'SELL (TIME)'
                i = i + max_hold + 1
                in_long = False
            elif not exit_found:
                i += 1
        elif not in_short and short_sig:
            entry_price = df.at[idx, 'close']
            df.at[idx, 'signal'] = 'SELL'
            in_short = True
            # exit logic as before but for short side
            exit_found = False
            for j in range(i+1, min(i+max_hold+1, len(df))):
                tp_price = entry_price * (1 - take_profit_pct)
                closej = df['close'].iloc[j]
                idxj = df.index[j]
                if closej <= tp_price:
                    df.at[idxj, 'signal'] = 'COVER (TP)'
                    i = j + 1
                    in_short = False
                    exit_found = True
                    break
            if not exit_found and (i+max_hold < len(df)):
                idxj = df.index[i+max_hold]
                df.at[idxj, 'signal'] = 'COVER (TIME)'
                i = i + max_hold + 1
                in_short = False
            elif not exit_found:
                i += 1
        elif in_long or in_short:
            i += 1
        else:
            i += 1
    return df

def place_market_bracket_order(symbol, side, quantity, take_profit_pct, stop_loss_pct):
    contract = Stock(symbol, exchange, currency)
    ib.qualifyContracts(contract)
    got_fill = ''
    if side == 'BUY':
        parent_order = MarketOrder('BUY', quantity, tif='GTC')
    else:
        parent_order = MarketOrder('SELL', quantity, tif='GTC')
    parent_trade = ib.placeOrder(contract, parent_order)
    while parent_trade.orderStatus.status not in ['Filled', 'Cancelled']:
        ib.waitOnUpdate()
    if parent_trade.orderStatus.status == 'Filled':
        avg_fill_price = parent_trade.orderStatus.avgFillPrice
        if side == 'BUY':
            tp_price = round(avg_fill_price * (1 + take_profit_pct), 2)
            sl_price = round(avg_fill_price * (1 - stop_loss_pct), 2)
            tp_order = LimitOrder('SELL', quantity, tp_price, tif='GTC', parentId=parent_order.orderId)
            sl_order = StopOrder('SELL', quantity, sl_price, tif='GTC', parentId=parent_order.orderId)
        else:
            tp_price = round(avg_fill_price * (1 - take_profit_pct), 2)
            sl_price = round(avg_fill_price * (1 + stop_loss_pct), 2)
            tp_order = LimitOrder('BUY', quantity, tp_price, tif='GTC', parentId=parent_order.orderId)
            sl_order = StopOrder('BUY', quantity, sl_price, tif='GTC', parentId=parent_order.orderId)
        ib.placeOrder(contract, tp_order)
        ib.placeOrder(contract, sl_order)
        if side == 'BUY':
            print(f"Bracket LONG: BUY {symbol} @ {avg_fill_price:.2f}, TP={tp_price}, SL={sl_price} (both GTC)")
        else:
            print(f"Bracket SHORT: SELL {symbol} @ {avg_fill_price:.2f}, TP={tp_price}, SL={sl_price} (both GTC)")
        got_fill = avg_fill_price
    else:
        print(f"{side} order for {symbol} was not filled.")
    return got_fill

for symbol in tickers:
    for size in candle_sizes:
        print(f"\n===== {symbol}: {size.upper()} bars =====")
        df = get_data(symbol, size, period)
        df = df[(df.index >= backtest_start) & (df.index <= backtest_end)]
        df = compute_signals(df)
        signals = df[df['signal'] != ''][['signal', 'close', 'ema_fast', 'ema_slow', 'ema_70']]

        recent_candles = df.index[-signal_max_age:]
        found_long = False
        found_short = False
        latest_long_fill = None
        latest_short_fill = None

        for idx, row in signals.iterrows():
            is_recent_long = (row['signal'] == 'BUY') and (idx in recent_candles)
            is_recent_short = (row['signal'] == 'SELL') and (idx in recent_candles)
            if is_recent_long or is_recent_short:
                print(f"Signal found: {row['signal']} at {idx.strftime('%Y-%m-%d %H:%M')}, price={row['close']:.2f} | EMA20={row['ema_fast']:.2f}, EMA50={row['ema_slow']:.2f}, EMA70={row['ema_70']:.2f}")
            else:
                print(f"{idx.strftime('%Y-%m-%d %H:%M')}: {row['signal']} @ {row['close']:.2f} | Not recent (<{signal_max_age} candles) - no trade placed.")

            if is_recent_long:
                latest_long_fill = place_market_bracket_order(symbol, 'BUY', quantity, take_profit_pct, stop_loss_pct)
                found_long = True
            elif is_recent_short:
                latest_short_fill = place_market_bracket_order(symbol, 'SELL', quantity, take_profit_pct, stop_loss_pct)
                found_short = True

        if found_long and latest_long_fill:
            print(f"Long trade executed for {symbol} at {latest_long_fill:.2f}")
        elif not found_long:
            print(f"No recent LONG signals for {symbol}/{size} in last {signal_max_age} candles.")
        if found_short and latest_short_fill:
            print(f"Short trade executed for {symbol} at {latest_short_fill:.2f}")
        elif not found_short:
            print(f"No recent SHORT signals for {symbol}/{size} in last {signal_max_age} candles.")

        # Open position status
        contract = Stock(symbol, exchange, currency)
        ib.qualifyContracts(contract)
        positions = ib.positions()
        open_longs = [p for p in positions if p.contract.symbol == symbol and p.position > 0]
        open_shorts = [p for p in positions if p.contract.symbol == symbol and p.position < 0]
        if open_longs:
            avg_cost = open_longs[0].avgCost
            market_price = ib.reqMktData(contract, "", False, False).last
            print(f"Open LONG position in {symbol}: Avg price {avg_cost:.2f}, Current price {market_price:.2f}")
        elif open_shorts:
            avg_cost = open_shorts[0].avgCost
            market_price = ib.reqMktData(contract, "", False, False).last
            print(f"Open SHORT position in {symbol}: Avg price {avg_cost:.2f}, Current price {market_price:.2f}")
        else:
            print(f"No open position in {symbol}.")

ib.disconnect()
