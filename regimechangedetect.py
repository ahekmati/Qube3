import numpy as np
import pandas as pd
from ib_insync import IB, Stock
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import ruptures as rpt

def regime_label(mean_return, threshold=0.001):
    if mean_return > threshold:
        return "bullish"
    elif mean_return < -threshold:
        return "bearish"
    else:
        return "consolidation"

def fetch_ibkr_ohlcv(ib, ticker, exchange, currency, duration, bar_size):
    contract = Stock(ticker, exchange, currency)
    bars = ib.reqHistoricalData(
        contract, endDateTime='',
        durationStr=duration, barSizeSetting=bar_size,
        whatToShow='TRADES', useRTH=True, formatDate=1)
    if not bars or len(bars) < 15:
        return None
    df = pd.DataFrame(bars)
    df = df.rename(columns=str.lower)
    df['returns'] = df['close'].pct_change()
    df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=['returns']).reset_index(drop=True)
    return df

def get_last_regime(df):
    signal = df['returns'].values
    if len(signal) < 10 or len(np.unique(signal)) < 3:
        return "unknown"
    model = "l2"
    algo = rpt.Pelt(model=model, min_size=5, jump=1).fit(signal)
    result = algo.predict(pen=5)
    cp_idx = result[-2] if len(result) > 1 else 0
    mean_ret = df.loc[cp_idx:, 'returns'].mean()
    return regime_label(mean_ret)

def build_lstm_model(X_train, y_train, lookback):
    model = Sequential()
    model.add(LSTM(24, input_shape=(lookback, X_train.shape[2]), activation='tanh', return_sequences=False))
    model.add(Dense(12, activation='relu'))
    model.add(Dense(1, activation='linear'))
    model.compile(optimizer='adam', loss='mse')
    model.fit(X_train, y_train, epochs=8, batch_size=16, verbose=0)
    return model

def make_lstm_features(df, features, lookback=30):
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(df[features])
    X_seq, y_seq = [], []
    close = df["close"].values
    for i in range(lookback, len(X_scaled)):
        seq_x = X_scaled[i - lookback:i]
        X_seq.append(seq_x)
        y_seq.append(close[i])
    X_seq, y_seq = np.array(X_seq), np.array(y_seq)
    return X_seq, y_seq

def has_open_long_position(ib, symbol):
    positions = ib.positions()
    return any(p.contract.symbol == symbol and p.position > 0 for p in positions)

def send_bracket_order(ib, symbol, qty, entry_price, target_pct, stop_loss_pct, exchange, currency):
    contract = Stock(symbol, exchange, currency)
    ib.qualifyContracts(contract)
    mkt_order = ib.marketOrder('BUY', qty)
    trade = ib.placeOrder(contract, mkt_order)
    print(f"Placed BUY market order {symbol} qty {qty}")
    while not trade.orderStatus.status in ['Filled', 'Cancelled']:
        ib.sleep(1)
    if trade.orderStatus.status == 'Filled':
        avg_fill = trade.orderStatus.avgFillPrice
        tp_price = round(avg_fill * (1 + target_pct), 2)
        sl_price = round(avg_fill * (1 - stop_loss_pct), 2)
        limit_order = ib.limitOrder('SELL', qty, tp_price, tif='GTC')
        stop_order = ib.stopOrder('SELL', qty, sl_price, tif='GTC')
        ib.placeOrder(contract, limit_order)
        ib.placeOrder(contract, stop_order)
        print(f"Placed GTC Limit (TP) @ {tp_price}, Stop @ {sl_price}")
        return True
    else:
        print("Order not filled or cancelled.")
        return False

IB_HOST = '127.0.0.1'
IB_PORT = 4001
IB_CLIENT_ID = 123

tickers = ['TQQQ','SSO','UDOW']
exchange = 'SMART'
currency = 'USD'
lookback = 30
quantity = 1
duration = '1 Y'
bar_size = '1 day'
recent_candles = 5  # Only review last 5 candles for new signals

feature_cols = ['close', 'returns']

ib = IB()
ib.connect(IB_HOST, IB_PORT, IB_CLIENT_ID)

for ticker in tickers:
    print(f"\n==== {ticker} ====")
    df = fetch_ibkr_ohlcv(ib, ticker, exchange, currency, duration, bar_size)
    if df is None or len(df) < (lookback + 2):
        print("No data for this timeframe.")
        continue

    regime = get_last_regime(df)
    print(f"Latest regime classification: {regime}")

    # LSTM pipeline
    df_lstm = df.tail(1000).copy()
    X_seq, y_seq = make_lstm_features(df_lstm, feature_cols, lookback=lookback)
    if len(X_seq) < 20:
        print("Not enough LSTM data.")
        continue
    split_idx = int(len(X_seq) * 0.8)
    X_train, X_test = X_seq[:split_idx], X_seq[split_idx:]
    y_train, y_test = y_seq[:split_idx], y_seq[split_idx:]
    model = build_lstm_model(X_train, y_train, lookback)
    pred = model.predict(X_test).flatten()
    close_test = y_test

    # Use date column if available, else use DataFrame index as fallback
    if "date" in df_lstm.columns:
        trade_dates = list(df_lstm["date"].iloc[split_idx + lookback:])
    else:
        trade_dates = list(df_lstm.index[split_idx + lookback:])

    position_open = has_open_long_position(ib, ticker)
    prompted_trade = False

    # Iterate over the last N candles; only prompt once per ticker
    start_idx = max(1, len(X_test) - recent_candles)
    for i in range(start_idx, len(X_test)):
        date = trade_dates[i]
        curr_pred = pred[i]
        prev_pred = pred[i-1]
        actual_close = close_test[i]
        if prompted_trade:
            # Only prompt for one new trade per ticker per run
            break
        if not position_open and curr_pred > prev_pred and regime == "bullish":
            print(f"\nTrade candidate for {ticker} on {date}:")
            print(f"  Forecast upturn. Current: {curr_pred:.2f}, Previous: {prev_pred:.2f}, Actual: {actual_close:.2f}, Regime: {regime}")
            take_trade = input("Take trade? [Y/n]: ")
            if take_trade.strip().lower() in ['', 'y', 'yes']:
                send_bracket_order(ib, ticker, quantity, actual_close, 0.015, 0.12, exchange, currency)
                position_open = True
            else:
                print("Trade skipped by user.")
            prompted_trade = True  # Prevent further prompts for this ticker
        else:
            print(f"{date}: No new entry (pred: {curr_pred:.2f}, prev: {prev_pred:.2f}, regime: {regime})")

ib.disconnect()
