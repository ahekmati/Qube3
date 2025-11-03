from ib_insync import *
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Conv1D, MaxPooling1D, Flatten, Dense

# IBKR CONNECTION SETTINGS
IB_HOST = '127.0.0.1'
IB_PORT = 4001
IB_CLIENT_ID = 123

tickers = ["TQQQ", "SSO"]
exchange = "ARCA"
currency = "USD"
lookback = 30
target_pct = 1.4
stop_loss_pct = 0.12
max_bars_in_trade = 150

bar_size = "1 day"
duration = "3 Y"
tf_name = "DAILY"

def fetch_ibkr_ohlcv(ib, ticker, exchange, currency, duration, bar_size):
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

    # Extended feature set:
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
    df["macd"] = df["close"].ewm(span=12).mean() - df["close"].ewm(span=26).mean()
    df["macd_signal"] = df["macd"].ewm(span=9).mean()
    df["volatility"] = df["close"].pct_change().rolling(14).std() * np.sqrt(252)
    df["volume_sma14"] = df["volume"].rolling(14).mean()
    df["volume_ratio"] = df["volume"] / df["volume_sma14"]
    # Regime/trend filter: 200-day MA
    df["sma200"] = df["close"].rolling(200).mean()
    df["regime"] = (df["close"] > df["sma200"]).astype(int)
    return df.dropna()

ib = IB()
ib.connect(IB_HOST, IB_PORT, IB_CLIENT_ID)

for ticker in tickers:
    print(f"\n==== {ticker} ====")
    print(f"\n-- {tf_name} ({bar_size}, {duration}) --")
    df = fetch_ibkr_ohlcv(ib, ticker, exchange, currency, duration, bar_size)
    if df is None or len(df) < (lookback+2):
        print("No data for this timeframe.")
        continue
    df = indicator_pipeline(df)
    features = [
        "wma","ema","rsi","cmo","willr","roc","hma","tema","adx","pline",
        "macd","macd_signal","volatility","volume_ratio","regime"
    ]
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

    open_trade = False
    entry_price = 0
    entry_date = None
    bars_in_trade = 0

    print("\nTrade log:")
    for i in range(len(prev_pred)):
        date = trade_dates[i+1]
        price = pred[i+1]
        actual_close = df.loc[date, "close"]
        if not open_trade:
            if curr_pred[i] > prev_pred[i]:
                entry_price = actual_close
                entry_date = date
                bars_in_trade = 0
                open_trade = True
                stop_loss = entry_price * (1 - stop_loss_pct)
                target = entry_price * (1 + target_pct)
                print(f"{date} BUY @ {entry_price:.2f}")
        else:
            bars_in_trade += 1
            if actual_close <= stop_loss:
                print(f"{date} SELL (STOP) @ {actual_close:.2f} | Entry {entry_date} @ {entry_price:.2f}")
                open_trade = False
            elif actual_close >= target:
                print(f"{date} SELL (TARGET) @ {actual_close:.2f} | Entry {entry_date} @ {entry_price:.2f}")
                open_trade = False
            elif bars_in_trade >= max_bars_in_trade:
                print(f"{date} SELL (EXPIRE) @ {actual_close:.2f} | Entry {entry_date} @ {entry_price:.2f}")
                open_trade = False

ib.disconnect()
print("\nBacktest complete.")
