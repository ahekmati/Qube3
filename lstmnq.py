import yfinance as yf
import pandas as pd
import numpy as np
import collections
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from ta.volatility import BollingerBands
from ta.momentum import RSIIndicator
from ta.trend import MACD
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

def run_lstm_regime(ticker, period, interval, lstm_window):
    df = yf.download(ticker, period=period, interval=interval)
    if df is None or df.empty or 'Close' not in df.columns:
        print(f"No data for {ticker} interval {interval}.")
        return None

    df = df.rename(columns={'Close': 'close', 'Volume': 'volume'})
    close_series = df['close'].squeeze()

    # Standard Indicators
    df['returns'] = close_series.pct_change()
    df['rsi'] = RSIIndicator(close_series, window=14).rsi()
    df['macd'] = MACD(close_series).macd()
    boll = BollingerBands(close_series)
    df['bbh'] = boll.bollinger_hband()
    df['bbl'] = boll.bollinger_lband()
    df['bb_width'] = df['bbh'] - df['bbl']

    # Multiple MAs
    ema_spans = [8, 10, 12, 21, 50, 100]
    sma_windows = [10, 20, 50, 200]
    for span in ema_spans:
        df[f'ema{span}'] = close_series.ewm(span=span, adjust=False).mean()
    for w in sma_windows:
        df[f'sma{w}'] = close_series.rolling(window=w, min_periods=1).mean()

    # Crossover features
    for fast, slow in [(8,21), (10,50), (12,100), (10,21), (21,50), (50,100)]:
        fcol, scol = f'ema{fast}', f'ema{slow}'
        if fcol in df.columns and scol in df.columns:
            df[f'{fcol}_above_{scol}'] = (df[fcol] > df[scol]).astype(int)
    for fast, slow in [(10,20), (20,50), (50,200)]:
        fcol, scol = f'sma{fast}', f'sma{slow}'
        if fcol in df.columns and scol in df.columns:
            df[f'{fcol}_above_{scol}'] = (df[fcol] > df[scol]).astype(int)

    # Price crossing slow MA
    for slow in [36,50,100,200]:
        ema_slow = f'ema{slow}'
        sma_slow = f'sma{slow}'
        if ema_slow in df.columns:
            df[f'close_crossed_above_{ema_slow}'] = (
                (close_series.shift(1) < df[ema_slow].shift(1)) & (close_series > df[ema_slow])
            ).astype(int)
        if sma_slow in df.columns:
            df[f'close_crossed_above_{sma_slow}'] = (
                (close_series.shift(1) < df[sma_slow].shift(1)) & (close_series > df[sma_slow])
            ).astype(int)

    df = df.dropna()

    # Features
    feature_cols = ['close', 'returns', 'rsi', 'macd', 'bbh', 'bbl', 'bb_width']
    feature_cols += [
        col for col in df.columns
        if (
            ('ema' in col and '_above_' not in col and 'close' not in col)
            or ('sma' in col and '_above_' not in col and 'close' not in col)
            or '_above_' in col or 'close_crossed_above' in col
        ) and col not in feature_cols
    ]

    X_raw = df[feature_cols].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_raw)

    # Regime labeling
    upper = df['returns'].quantile(0.7)
    lower = df['returns'].quantile(0.3)
    center = df['bb_width'].median() / 2
    labels = []
    for i in range(len(df)):
        if df['returns'].iloc[i] > upper and df['bb_width'].iloc[i] > center:
            labels.append(1)   # Bullish
        elif df['returns'].iloc[i] < lower and df['bb_width'].iloc[i] > center:
            labels.append(2)   # Bearish
        else:
            labels.append(0)   # Consolidating
    df['regime'] = labels

    # LSTM Data Prep
    window = min(lstm_window, len(X_scaled) - 1)
    X, y = [], []
    for i in range(window, len(X_scaled)):
        X.append(X_scaled[i - window:i])
        y.append(df['regime'].iloc[i])
    X, y = np.array(X), np.array(y)
    if len(X) < 2 or len(set(y)) < 2:
        print(f"{ticker}: not enough data for LSTM, skipping.")
        return None

    X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=False, test_size=0.15)

    # LSTM Model
    model = Sequential([
        LSTM(64, return_sequences=True, input_shape=(window, X.shape[2])),
        Dropout(0.3),
        LSTM(32),
        Dropout(0.3),
        Dense(16, activation='relu'),
        Dense(3, activation='softmax')
    ])
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    callbacks = [EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)]
    model.fit(X_train, y_train, validation_data=(X_test, y_test),
              epochs=60, batch_size=32, callbacks=callbacks, verbose=0)

    # Output
    y_pred = np.argmax(model.predict(X_test), axis=1)
    dates = df.index[len(df) - len(y_pred):]
    regime_names = []
    for val in y_pred[-20:]:
        if val == 0: regime_names.append("Consolidating")
        elif val == 1: regime_names.append("Bullish")
        elif val == 2: regime_names.append("Bearish")
    return {'ticker': ticker, 'dates': dates[-20:], 'regimes': regime_names}

# ------ Tickers highly correlated with QQQ ------
tickers = ['QQQ', 'SPY', 'AAPL', 'MSFT', 'NVDA', 'GOOGL','SVXY','SOXL']

period = '90d'
interval = '4h'
lstm_window = 8

results = []
for ticker in tickers:
    res = run_lstm_regime(ticker, period, interval, lstm_window)
    if res:
        results.append(res)

# -- Print per-ticker regime results --
for r in results:
    print(f"\n--- {r['ticker']} regime (last 10 bars) ---")
    for d, regime in zip(r['dates'], r['regimes']):
        print(f"{d}: {regime}")

# -- Consensus regime call for last bar --
if results:
    last_regimes = [r['regimes'][-1] for r in results]
    consensus = collections.Counter(last_regimes).most_common(1)[0][0]
    print(f"\nConsensus regime (across tickers), last bar: {consensus}")
    if consensus == "Bullish":
        print("Action: BUY (confirmed by majority of correlated tickers)")
    elif consensus == "Bearish":
        print("Action: SHORT (confirmed by majority of correlated tickers)")
    else:
        print("Action: HOLD/No Action (mixed signals)")
