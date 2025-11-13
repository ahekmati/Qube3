from ib_insync import *
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Conv1D, MaxPooling1D, Flatten, Dense
from colorama import init, Fore, Style

init(autoreset=True)

IB_HOST = '127.0.0.1'
IB_PORT = 4001
IB_CLIENT_ID = 123
ACCOUNT = 'U22816462'

tickers = ["TQQQ", "SSO"]
exchange = "SMART"
currency = "USD"
lookback = 30
target_pct = .35
stop_loss_pct = 0.12
max_bars_in_trade = 180
max_hold_bars = 40
quantity = 1

bar_size = "1 day"
duration = "3 Y"
tf_name = "DAILY"
x_ATR_BE = 2

def fetch_ibkr_ohlcv(ib, ticker, exchange, currency, duration, bar_size):
    primary_map = {'TQQQ': 'NASDAQ', 'SSO': 'ARCA'}
    contract = Stock(ticker, exchange, currency, primaryExchange=primary_map.get(ticker, 'NASDAQ'))
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
    return df

def add_classic_signals(df):
    ema_fast = 20
    ema_slow = 50
    ema_short_slow = 70
    z_window = 20
    z_thresh_long = -0.5
    z_thresh_short = 0.7
    df['ema_fast'] = df['close'].ewm(span=ema_fast, adjust=False).mean()
    df['ema_slow'] = df['close'].ewm(span=ema_slow, adjust=False).mean()
    std20 = df['close'].rolling(z_window).std()
    dip_z = (df['close'] - df['ema_fast']) / std20
    df['buy_the_dip'] = ((df['close'] < df['ema_slow']) & (dip_z < z_thresh_long)).astype(int)
    df['ema_70'] = df['close'].ewm(span=ema_short_slow, adjust=False).mean()
    rally_z = (df['close'] - df['ema_fast']) / std20
    df['sell_the_rally'] = ((df['ema_fast'] < df['ema_70']) & (df['close'] > df['ema_70']) & (rally_z >= z_thresh_short)).astype(int)
    return df

def indicator_pipeline(df):
    def wma(series, period):
        weights = np.arange(1, period+1)
        return series.rolling(period).apply(lambda x: np.dot(x, weights)/weights.sum(), raw=True)
    def hma(series, period):
        wmaf = wma(series, period//2)
        wmas = wma(series, period)
        raw_hma = 2 * wmaf - wmas
        return wma(raw_hma, int(np.sqrt(period)))
    def tema(series, period):
        ema1 = series.ewm(span=period, adjust=False).mean()
        ema2 = ema1.ewm(span=period, adjust=False).mean()
        ema3 = ema2.ewm(span=period, adjust=False).mean()
        return 3*ema1 - 3*ema2 + ema3
    def rsi(series, period):
        delta = series.diff()
        up, down = delta.clip(lower=0), -delta.clip(upper=0)
        ma_up = up.ewm(com=(period - 1), min_periods=period).mean()
        ma_down = down.ewm(com=(period - 1), min_periods=period).mean()
        rs = ma_up / (ma_down + 1e-10)
        return 100 - (100 / (1 + rs))
    def cmo(series, period):
        diff = series.diff()
        up = diff.where(diff > 0, 0).rolling(period).sum()
        down = diff.where(diff < 0, 0).abs().rolling(period).sum()
        denominator = up + down
        return 100 * (up - down) / (denominator + 1e-10)
    def willr(df, period):
        high = df["high"]
        low = df["low"]
        close = df["close"]
        return (high.rolling(period).max() - close) / (high.rolling(period).max() - low.rolling(period).min()) * -100
    def roc(series, period):
        return 100 * (series - series.shift(period)) / (series.shift(period) + 1e-10)
    def adx(df, period):
        high = df["high"]
        low = df["low"]
        close = df["close"]
        plus_dm = high.diff()
        minus_dm = low.diff().abs()
        tr1 = high - low
        tr2 = (high - close.shift()).abs()
        tr3 = (low - close.shift()).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window=period).mean()
        plus_di = 100 * (plus_dm.ewm(span=period, min_periods=period).mean() / atr)
        minus_di = 100 * (minus_dm.ewm(span=period, min_periods=period).mean() / atr)
        dx = (abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)) * 100
        adx = dx.ewm(span=period, min_periods=period).mean()
        return adx
    def pline(series, period):
        rises = (series.diff() > 0).astype(int)
        return rises.rolling(period).sum() / period * 100

    df["wma"] = wma(df["close"], 14)
    df["ema"] = df["close"].ewm(span=14).mean()
    df["rsi"] = rsi(df["close"], 14)
    df["cmo"] = cmo(df["close"], 14)
    df["willr"] = willr(df, 14)
    df["roc"] = roc(df["close"], 14)
    df["hma"] = hma(df["close"], 14)
    df["tema"] = tema(df["close"], 14)
    df["adx"] = adx(df, 14)
    df["pline"] = pline(df["close"], 14)
    atr_period = 14
    df["ATR"] = (df["high"] - df["low"]).rolling(atr_period).mean()
    lookback_high = 20
    price_percent_above_high = ((df["close"] / df["high"].rolling(lookback_high).max()) - 1) * 100
    df["overbought"] = ((df["rsi"] > 78) | (price_percent_above_high > 7)).astype(int)
    return df

def has_open_long_position(ib, symbol):
    positions = ib.positions()
    return any(p.contract.symbol == symbol and p.position > 0 for p in positions)

def get_symbol_stop_orders(ib, symbol, qty):
    """
    Returns all active/working GTC stop orders for that symbol and quantity.
    """
    stop_orders = []
    for opent in ib.openTrades():
        if (opent.contract.symbol == symbol and 
            opent.order.orderType.upper() == 'STP' and
            opent.order.action == 'SELL' and
            int(opent.order.totalQuantity) == int(qty) and
            opent.order.tif.upper() == 'GTC'):
            stop_orders.append(opent)
    return stop_orders

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
        print(f"Order CANCELLED for {symbol}.")
    else:
        print(f"Order still not filled/cancelled after {max_wait} seconds. Status: {trade.orderStatus.status}")
    return None

def add_trend_filter(df, n_days=20, low_thresh=0.07):
    recent_low = df['close'].rolling(n_days).min()
    rel_to_low = (df['close'] - recent_low) / recent_low
    df['not_too_far_from_low'] = (rel_to_low < low_thresh).astype(int)
    return df

def cancel_and_replace_stop(ib, contract, old_stop_order, qty, new_stop_price, account):
    if old_stop_order:
        ib.cancelOrder(old_stop_order)
    stop_order = StopOrder('SELL', qty, new_stop_price, tif='GTC')
    stop_order.account = account
    stop_order.outsideRth = True
    trade = ib.placeOrder(contract, stop_order)
    print(f"Trailing stop moved to {new_stop_price:.2f}")
    return trade

ib = IB()
ib.connect(IB_HOST, IB_PORT, IB_CLIENT_ID)

n_bars_back = 2

for ticker in tickers:
    print(f"\n==== {ticker} ====")
    df = fetch_ibkr_ohlcv(ib, ticker, exchange, currency, duration, bar_size)
    if df is None or len(df) < (lookback + 2):
        print("No data for this timeframe.")
        continue
    df = indicator_pipeline(df)
    df = add_classic_signals(df)
    df = add_trend_filter(df, n_days=20, low_thresh=0.07)
    features = [
        "wma","ema","rsi","cmo","willr","roc","hma","tema","adx","pline",
        "ema_fast","ema_slow","ema_70","buy_the_dip","sell_the_rally","not_too_far_from_low"
    ]
    df = df.dropna(subset=features)
    scaler = MinMaxScaler()
    X_ind = scaler.fit_transform(df[features])
    pca = PCA(n_components=3)
    X_pca = pca.fit_transform(X_ind)
    X_seq, y_seq = [], []
    close = df["close"].values
    for i in range(lookback, len(X_pca)):
        seq_x = X_pca[i - lookback:i]
        X_seq.append(seq_x)
        y_seq.append(close[i])
    X_seq, y_seq = np.array(X_seq), np.array(y_seq)
    if len(X_seq) < 10:
        print("Not enough data after indicators for ML fit.")
        continue
    split_idx = int(len(X_seq) * 0.8)
    X_train, X_test = X_seq[:split_idx], X_seq[split_idx:]
    y_train, y_test = y_seq[:split_idx], y_seq[split_idx:]
    model = Sequential()
    model.add(LSTM(50, input_shape=(lookback, 3), return_sequences=True, activation='tanh'))
    model.add(Conv1D(50, kernel_size=2, padding='same', activation='relu'))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Flatten())
    model.add(Dense(25, activation='relu'))
    model.add(Dense(1, activation='linear'))
    model.compile(optimizer='adam', loss='mse')
    model.fit(X_train, y_train, epochs=12, batch_size=32, validation_data=(X_test, y_test), verbose=0)

    pred = model.predict(X_test).flatten()
    trade_dates = df.index[split_idx + lookback:]
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
        actual_close = df.loc[date, "close"]
        if position_open:
            trade_duration += 1
            if df.loc[date, "overbought"] == 1:
                print(f"Overbought/exhaustion detected at {date} for {ticker}. Liquidating position.")
                contract = Stock(ticker, exchange, currency)
                mkt_exit = MarketOrder('SELL', quantity)
                mkt_exit.account = ACCOUNT
                mkt_exit.outsideRth = True
                ib.placeOrder(contract, mkt_exit)
                position_open = False
                continue
            if trade_duration >= max_hold_bars:
                print(f"Max holding period ({max_hold_bars} bars) reached for {ticker}. Liquidating position.")
                contract = Stock(ticker, exchange, currency)
                mkt_exit = MarketOrder('SELL', quantity)
                mkt_exit.account = ACCOUNT
                mkt_exit.outsideRth = True
                ib.placeOrder(contract, mkt_exit)
                position_open = False
                continue
            entry_price = trade_info["avg_fill"] if trade_info else actual_close
            atr = df.loc[date, "ATR"]
            # Robust stop-loss: ONLY 1 stop order for this position
            open_stop_orders = get_symbol_stop_orders(ib, ticker, quantity)
            if not open_stop_orders:
                stop_price = round(entry_price * (1 - stop_loss_pct), 2)
                contract = Stock(ticker, exchange, currency)
                new_stop = StopOrder('SELL', quantity, stop_price, tif='GTC')
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
            if curr_pred[i] > prev_pred[i] and df.loc[date, "not_too_far_from_low"]:
                print(f"{date} RECENT BUY SIGNAL (within last {n_bars_back} bars) @ {actual_close:.2f} for {ticker} (trend-respecting entry)")
                trade_info = send_bracket_order(ib, ticker, quantity, actual_close, target_pct, stop_loss_pct, ACCOUNT, max_wait=5)
                stop_order = trade_info["stop_order"] if trade_info else None
                if not trade_info:
                    print(f"No trade filled for {ticker} within 5 seconds, moving to next ticker.")
                    trade_taken = False
                    break
                position_open = True
                trade_duration = 0
                trade_taken = True
                break
            else:
                print(f"{date} No entry for this bar (within last {n_bars_back} bars for {ticker}).")

    if position_open and not trade_taken:
        print(f"No new trade for {ticker} because a position is already open.")

ib.disconnect()
