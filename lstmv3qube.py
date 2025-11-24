import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from ib_insync import *
import pandas as pd
import numpy as np
import pandas_ta as ta
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, LSTM, Conv1D, MaxPooling1D, Flatten, Dense
from colorama import init, Fore, Style

init(autoreset=True)

IB_HOST = '127.0.0.1'
IB_PORT = 4001
IB_CLIENT_ID = 123
ACCOUNT = 'U22816462'

tickers = ["TQQQ", "SSO", "UDOW", "TNA", "QTUM", "NVDL", "SVXY", "QDTE", "SVOL", "VXX"]
exchange = "SMART"
currency = "USD"
lookback = 30
target_pct = 0.75
stop_loss_pct = 0.05
max_hold_bars = 40
order_size = 1

duration_daily = "3 Y"
duration_4h = "90 D"

# Only require essential features for merge
ESSENTIAL_FEATURES = [
    'rsi', 'ema10', 'vwap', 'position_vs_low',
    'rsi_4h', 'ema10_4h', 'vwap_4h', 'position_vs_low_4h'
]

def localize_and_sort_index(df):
    df.index = pd.to_datetime(df.index)
    if hasattr(df.index, 'tz') and df.index.tz is not None:
        df.index = df.index.tz_convert(None)
    return df.sort_index()

def fetch_tf_ohlcv(ib, ticker, exchange, currency, duration, bar_size):
    contract = Stock(ticker, exchange, currency)
    bars = ib.reqHistoricalData(
        contract,
        endDateTime='',
        durationStr=duration,
        barSizeSetting=bar_size,
        whatToShow='TRADES',
        useRTH=False,
        formatDate=1
    )
    if not bars:
        return None
    df = util.df(bars)
    df = df.rename(columns=str.lower).set_index('date')
    df = localize_and_sort_index(df)
    return df

def robust_indicators(df, ma_list_override=None):
    ma_list = ma_list_override if ma_list_override is not None else [5, 9, 10, 13, 20]
    ma_dict = {}
    for w in ma_list:
        ma_dict[f'ema{w}'] = df['close'].ewm(span=w).mean()
        ma_dict[f'sma{w}'] = df['close'].rolling(w).mean()
        ma_dict[f'wma{w}'] = ta.wma(df['close'], length=w)
        ma_dict[f'hma{w}'] = ta.hma(df['close'], length=w)
        ma_dict[f'tema{w}'] = ta.tema(df['close'], length=w)
    ma_df = pd.DataFrame(ma_dict, index=df.index)
    df = pd.concat([df, ma_df], axis=1)

    ma_pairs = [(fast, slow) for fast in ma_list for slow in ma_list if fast < slow]
    crossover_dict = {}
    for mtype in ["ema", "sma", "wma"]:
        for fast, slow in ma_pairs:
            try:
                key_up = f"{mtype}{fast}_{slow}_up"
                key_dn = f"{mtype}{fast}_{slow}_dn"
                crossover_dict[key_up] = ((df[f"{mtype}{fast}"].shift(1) < df[f"{mtype}{slow}"].shift(1)) & (df[f"{mtype}{fast}"] > df[f"{mtype}{slow}"])).astype(int)
                crossover_dict[key_dn] = ((df[f"{mtype}{fast}"].shift(1) > df[f"{mtype}{slow}"].shift(1)) & (df[f"{mtype}{fast}"] < df[f"{mtype}{slow}"])).astype(int)
            except KeyError:
                continue
    crossover_df = pd.DataFrame(crossover_dict, index=df.index)
    df = pd.concat([df, crossover_df], axis=1)

    fragments = []
    fragments.append(pd.DataFrame({"rsi": ta.rsi(df["close"], 14)}, index=df.index))
    fragments.append(pd.DataFrame({"cmo": ta.cmo(df["close"], 14)}, index=df.index))
    fragments.append(pd.DataFrame({"willr": ta.willr(df["high"], df["low"], df["close"], 14)}, index=df.index))
    fragments.append(pd.DataFrame({"roc": ta.roc(df["close"], 14)}, index=df.index))
    stoch = ta.stoch(df["high"], df["low"], df["close"])
    fragments.append(stoch)
    macd = ta.macd(df["close"])
    fragments.append(macd)
    ppo = ta.ppo(df["close"])
    fragments.append(ppo)
    fragments.append(pd.DataFrame({"mfi": ta.mfi(df["high"], df["low"], df["close"], df["volume"], 14)}, index=df.index))
    fragments.append(ta.adx(df["high"], df["low"], df["close"]))
    fragments.append(pd.DataFrame({"atr": ta.atr(df["high"], df["low"], df["close"])}, index=df.index))
    fragments.append(pd.DataFrame({"obv": ta.obv(df["close"], df["volume"])}, index=df.index))
    fragments.append(pd.DataFrame({"vwap": ta.vwap(df["high"], df["low"], df["close"], df["volume"])}, index=df.index))
    bb = ta.bbands(df["close"])
    fragments.append(bb)
    kc = ta.kc(df["high"], df["low"], df["close"])
    fragments.append(kc.add_prefix("kc_"))
    dc = ta.donchian(df["high"], df["low"], df["close"], length=20)
    fragments.append(dc.add_prefix("dc_"))
    fragments.append(pd.DataFrame({"realized_vol": df["close"].rolling(10).std() / df["close"].rolling(10).mean()}, index=df.index))
    fragments.append(pd.DataFrame({"position_vs_high": (df["close"] - df["high"].rolling(20).max()) / df["high"].rolling(20).max()}, index=df.index))
    fragments.append(pd.DataFrame({"position_vs_low": (df["close"] - df["low"].rolling(20).min()) / df["low"].rolling(20).min()}, index=df.index))
    fragments.append(pd.DataFrame({"overbought": ((ta.rsi(df["close"], 14) > 78) | 
                                                  (ta.cmo(df["close"], 14) > 70) | 
                                                  (ta.willr(df["high"], df["low"], df["close"], 14) > -10)).astype(int)}, index=df.index))
    fragments.append(pd.DataFrame({"oversold": ((ta.rsi(df["close"], 14) < 22) | 
                                                (ta.cmo(df["close"], 14) < -70) | 
                                                (ta.willr(df["high"], df["low"], df["close"], 14) < -90)).astype(int)}, index=df.index))
    df = pd.concat([df] + fragments, axis=1)
    return df

def has_open_long_position(ib, symbol):
    positions = ib.positions()
    return any(p.contract.symbol == symbol and p.position > 0 for p in positions)

def get_symbol_stop_orders(ib, symbol, qty):
    return [
        opent for opent in ib.openTrades()
        if (opent.contract.symbol == symbol and 
            opent.order.orderType.upper() == 'STP' and
            opent.order.action == 'SELL' and
            int(opent.order.totalQuantity) == int(qty) and
            opent.order.tif.upper() == 'GTC')
    ]

def send_bracket_order(ib, symbol, qty, entry_price, take_profit_pct, stop_loss_pct, account, max_wait=5):
    contract = Stock(symbol, exchange, currency)
    ib.qualifyContracts(contract)
    mkt_order = MarketOrder('BUY', qty)
    mkt_order.account = account
    mkt_order.outsideRth = True
    trade = ib.placeOrder(contract, mkt_order)
    print(f"Placed BUY market order for {symbol}, quantity {qty}. Waiting for fill or cancellation...")
    waited = 0
    while trade.orderStatus.status not in ['Filled', 'Cancelled'] and waited < max_wait:
        print(f"  Current status: {trade.orderStatus.status} (waited {waited+1} seconds)")
        ib.sleep(1)
        waited += 1
    if trade.orderStatus.status == 'Filled':
        avg_fill = trade.orderStatus.avgFillPrice
        filled_qty = trade.orderStatus.filled
        print(f"Order FILLED for {symbol}! Filled {filled_qty} at price {avg_fill:.2f}")
        tp_price = round(avg_fill * (1 + take_profit_pct), 2)
        sl_price = round(avg_fill * (1 - stop_loss_pct), 2)
        limit_order = LimitOrder('SELL', qty, tp_price, tif='GTC')
        limit_order.account = account
        limit_order.outsideRth = True
        stop_order = StopOrder('SELL', qty, sl_price, tif='GTC')
        stop_order.account = account
        stop_order.outsideRth = True
        ib.placeOrder(contract, limit_order)
        ib.placeOrder(contract, stop_order)
        print(f"Placed GTC Limit (TP) @ {tp_price}, Stop @ {sl_price}")
        return {"avg_fill": avg_fill, "stop_price": sl_price, "stop_order": stop_order, "entry_time": pd.Timestamp.now()}
    elif trade.orderStatus.status == 'Cancelled':
        print(Fore.RED + Style.BRIGHT + f"Order CANCELLED for {symbol}.")
    else:
        print(Fore.RED + Style.BRIGHT +
              f"Order still not filled/cancelled after {max_wait} seconds. Status: {trade.orderStatus.status}")
    return None

ib = IB()
ib.connect(IB_HOST, IB_PORT, IB_CLIENT_ID)

n_bars_back = 2

for ticker in tickers:
    print(f"\n==== {ticker} ====")
    daily_df = fetch_tf_ohlcv(ib, ticker, exchange, currency, duration_daily, '1 day')
    fourh_df = fetch_tf_ohlcv(ib, ticker, exchange, currency, duration_4h, '4 hours')
    if (daily_df is None or len(daily_df) < (lookback + 2) or fourh_df is None):
        print("No data for one or more timeframes.")
        continue

    daily_df = robust_indicators(daily_df, ma_list_override=[5, 9, 10, 13, 20])
    fourh_df = robust_indicators(fourh_df, ma_list_override=[5, 9, 10, 13, 20]).add_suffix('_4h')

    daily_df = localize_and_sort_index(daily_df)
    fourh_df = localize_and_sort_index(fourh_df)

    merge_df = daily_df.copy()
    merge_df = merge_df.join(fourh_df.resample('1D').last(), how='left')

    available_essential = [c for c in ESSENTIAL_FEATURES if c in merge_df.columns]
    merge_df = merge_df.dropna(subset=available_essential)
    if merge_df.empty or len(merge_df) < lookback + 2:
        print(f"{Fore.YELLOW + Style.BRIGHT}Not enough data after indicator prep for {ticker}")
        continue

    scaler = MinMaxScaler()
    X = scaler.fit_transform(merge_df[available_essential])
    y = merge_df["close"].values
    X_seq, y_seq = [], []
    for i in range(lookback, len(merge_df)):
        X_seq.append(X[i-lookback:i])
        y_seq.append(y[i])
    X_seq, y_seq = np.array(X_seq), np.array(y_seq)
    if len(X_seq) < 10:
        print(f"{Fore.YELLOW + Style.BRIGHT}Not enough data after indicators for ML fit for {ticker}")
        continue
    split_idx = int(len(X_seq)*0.8)
    X_train, X_test = X_seq[:split_idx], X_seq[split_idx:]
    y_train, y_test = y_seq[:split_idx], y_seq[split_idx:]
    model = Sequential()
    model.add(Input(shape=(lookback, X_seq.shape[2])))
    model.add(LSTM(40, return_sequences=True, activation='tanh'))
    model.add(Conv1D(32, 2, padding='same', activation='relu'))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Flatten())
    model.add(Dense(25, activation='relu'))
    model.add(Dense(1, activation='linear'))
    model.compile(optimizer='adam', loss='mse')
    model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test), verbose=0)
    pred = model.predict(X_test).flatten()
    trade_dates = merge_df.index[split_idx + lookback:]
    prev_pred = pred[:-1]
    curr_pred = pred[1:]

    position_open = has_open_long_position(ib, ticker)
    trade_info = None
    trade_duration = 0
    stop_order = None
    trade_taken = False

    for i in range(len(prev_pred) - n_bars_back, len(prev_pred)):
        date = trade_dates[i+1]
        price = pred[i+1]
        actual_close = merge_df.loc[date, "close"]
        signal_source = "Daily"
        # Example: if you want to check a 4h condition for the entry, you could also compare e.g. rsi_4h, etc.
        # signal_source = "4-hour" if merge_df.loc[date, "rsi_4h"] > merge_df.loc[date, "rsi"] else "Daily"
        if position_open:
            trade_duration += 1
            if merge_df.loc[date, "overbought"] == 1:
                print(f"Overbought/exhaustion detected at {date} for {ticker}. Liquidating position.")
                contract = Stock(ticker, exchange, currency)
                mkt_exit = MarketOrder('SELL', order_size)
                mkt_exit.account = ACCOUNT
                mkt_exit.outsideRth = True
                ib.placeOrder(contract, mkt_exit)
                position_open = False
                continue
            if trade_duration >= max_hold_bars:
                print(f"Max holding period ({max_hold_bars} bars) reached for {ticker}. Liquidating position.")
                contract = Stock(ticker, exchange, currency)
                mkt_exit = MarketOrder('SELL', order_size)
                mkt_exit.account = ACCOUNT
                mkt_exit.outsideRth = True
                ib.placeOrder(contract, mkt_exit)
                position_open = False
                continue
            entry_price = trade_info["avg_fill"] if trade_info else actual_close
            open_stop_orders = get_symbol_stop_orders(ib, ticker, order_size)
            if not open_stop_orders:
                stop_price = round(entry_price * (1 - stop_loss_pct), 2)
                contract = Stock(ticker, exchange, currency)
                new_stop = StopOrder('SELL', order_size, stop_price, tif='GTC')
                new_stop.account = ACCOUNT
                new_stop.outsideRth = True
                ib.placeOrder(contract, new_stop)
                stop_price_print = stop_price
                print(Fore.YELLOW + f"Placed new stop loss for {ticker} at {stop_price:.2f}")
            else:
                stop_price_print = open_stop_orders[0].order.auxPrice
            print(Fore.GREEN + Style.BRIGHT +
                  f"OPEN POSITION {ticker} | ENTRY: {entry_price:.2f} | LAST: {actual_close:.2f} | STOP: {stop_price_print}")
        else:
            if curr_pred[i] > prev_pred[i] and merge_df.loc[date, "position_vs_low"] > -0.05:
                print(Fore.CYAN + f"{date} RECENT BUY SIGNAL on {signal_source} (within last {n_bars_back} bars) @ {actual_close:.2f} for {ticker} (trend-respecting entry)")
                trade_info = send_bracket_order(ib, ticker, order_size, actual_close, target_pct, stop_loss_pct, ACCOUNT, max_wait=5)
                stop_order = trade_info["stop_order"] if trade_info else None
                if not trade_info:
                    print(Fore.RED + Style.BRIGHT + f"No trade filled for {ticker} within 5 seconds, moving to next ticker.")
                    trade_taken = False
                    break
                position_open = True
                trade_duration = 0
                trade_taken = True
                break
            else:
                print(Fore.YELLOW + Style.BRIGHT +
                      f"{date} No entry for this bar (within last {n_bars_back} bars for {ticker}).")

    if position_open and not trade_taken:
        print(f"No new trade for {ticker} because a position is already open.")

ib.disconnect()
