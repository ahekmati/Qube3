from ib_insync import *
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, LSTM, Conv1D, MaxPooling1D, Flatten, Dense
from colorama import init, Fore, Style

init(autoreset=True)

IB_HOST = '127.0.0.1'
IB_PORT = 4001
IB_CLIENT_ID = 123

tickers = ['QQQ','SPY','UDOW','TNA','NVDL','SVXY','SVOL','VXX']
exchange = "SMART"
currency = "USD"
lookback = 30
bar_size = "1 day"
duration = "3 Y"

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
        adxv = dx.ewm(span=period, min_periods=period).mean()
        return adxv
    def rsi(series, period):
        delta = series.diff()
        up, down = delta.clip(lower=0), -delta.clip(upper=0)
        ma_up = up.ewm(com=(period - 1), min_periods=period).mean()
        ma_down = down.ewm(com=(period - 1), min_periods=period).mean()
        rs = ma_up / (ma_down + 1e-10)
        return 100 - (100 / (1 + rs))
    df["wma"] = wma(df["close"], 14)
    df["ema"] = df["close"].ewm(span=14).mean()
    df["roc"] = roc(df["close"], 14)
    df["adx"] = adx(df, 14)
    df["rsi"] = rsi(df["close"], 14)
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

def resample_4h(df):
    df_4h = df.resample('4h').agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum'
    }).dropna()
    df_4h = indicator_pipeline(df_4h)
    return df_4h

ib = IB()
ib.connect(IB_HOST, IB_PORT, IB_CLIENT_ID)
lookback = 30

for ticker in tickers:
    print(f"\n==== {ticker} ====")
    df = fetch_ibkr_ohlcv(ib, ticker, exchange, currency, duration, bar_size)
    if df is None or len(df) < (lookback + 2):
        print("No data for this timeframe.")
        continue
    df.index = pd.to_datetime(df.index)
    df = indicator_pipeline(df)
    df = add_classic_signals(df)
    df_4h = resample_4h(df)
    df_4h_ff = df_4h.reindex(df.index, method='ffill')
    for col in ['wma', 'ema', 'roc', 'adx', 'rsi']:
        df[f'4h_{col}'] = df_4h_ff[col]
    features = [
        "wma","ema","roc","adx","ema_fast","ema_slow","ema_70",
        "buy_the_dip","sell_the_rally","rsi",
        "4h_wma","4h_ema","4h_roc","4h_adx","4h_rsi"
    ]
    df = df.dropna(subset=features)
    scaler = MinMaxScaler()
    X_ind = scaler.fit_transform(df[features])
    X_seq, y_seq = [], []
    close = df["close"].values
    for i in range(lookback, len(X_ind)):
        seq_x = X_ind[i - lookback:i]
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
    model.add(Input(shape=(lookback, X_seq.shape[2])))
    model.add(LSTM(50, return_sequences=True, activation='tanh'))
    model.add(Conv1D(25, kernel_size=2, padding='same', activation='relu'))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Flatten())
    model.add(Dense(15, activation='relu'))
    model.add(Dense(1, activation='linear'))
    model.compile(optimizer='adam', loss='mse')
    model.fit(X_train, y_train, epochs=12, batch_size=32, validation_data=(X_test, y_test), verbose=0)
    proba = model.predict(X_test[-1:])[0][0]
    last_close = y_test[-1]
    signals = []
    tf_labels = []
    prices = []

    # Daily timeframe logic
    if (df['rsi'].iloc[-1] < 15):
        signals.append("EXTREME OVERSOLD (daily)")
        tf_labels.append("daily")
        prices.append(f"{last_close:.2f}")
    if (df['rsi'].iloc[-1] > 80):
        signals.append("EXTREME OVERBOUGHT (daily)")
        tf_labels.append("daily")
        prices.append(f"{last_close:.2f}")
    if (df['buy_the_dip'].iloc[-1] == 1 and proba > last_close*1.01):
        signals.append("BUY SIGNAL (daily)")
        tf_labels.append("daily")
        prices.append(f"{last_close:.2f}")
    if (df['sell_the_rally'].iloc[-1] == 1 and proba < last_close*0.99):
        signals.append("EXTREME SELL SIGNAL (daily)")
        tf_labels.append("daily")
        prices.append(f"{last_close:.2f}")
    # 4H logic
    if (df['4h_rsi'].iloc[-1] < 15):
        signals.append("EXTREME OVERSOLD (4h)")
        tf_labels.append("4h")
        prices.append(f"{df['close'].iloc[-1]:.2f}")
    if (df['4h_rsi'].iloc[-1] > 80):
        signals.append("EXTREME OVERBOUGHT (4h)")
        tf_labels.append("4h")
        prices.append(f"{df['close'].iloc[-1]:.2f}")
    if (df['4h_ema'].iloc[-1] and last_close > (df['4h_ema'].iloc[-1] + 2*df['4h_ema'].rolling(10).std().iloc[-1]) and proba < last_close*0.98):
        signals.append("EXTREME SELL SIGNAL (4h)")
        tf_labels.append("4h")
        prices.append(f"{last_close:.2f}")
    # Summary printing
    if signals:
        for i, sig in enumerate(signals):
            print(Fore.YELLOW + Style.BRIGHT + f"{sig} at price {prices[i]}")
        print("Triggered timeframes: " + ", ".join(sorted(set(tf_labels))))
    else:
        print("No strong buy/sell or extreme condition today.")

ib.disconnect()
