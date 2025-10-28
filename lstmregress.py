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
import statsmodels.api as sm

# Select timeframes
intervals = {'1h': '1h', '4h': '4h', '1d': '1d'}
ticker = 'QQQ'
performance_summary = {}
regime_predictions = {}

for name, intvl in intervals.items():
    print(f"\n--- Analyzing {name} bars ---")
    df = yf.download(ticker, period='180d', interval=intvl)
    if df is None or df.empty or 'Close' not in df.columns:
        print(f"No data for interval {intvl}. Skipping.")
        continue
    
    df = df.rename(columns={'Close': 'close', 'Volume': 'volume'})
    close_series = df['close'].squeeze()
    df['returns'] = close_series.pct_change()
    df['rsi'] = RSIIndicator(close_series, window=14).rsi()
    df['macd'] = MACD(close_series).macd()
    boll = BollingerBands(close_series)
    df['bbh'] = boll.bollinger_hband()
    df['bbl'] = boll.bollinger_lband()
    df['bb_width'] = df['bbh'] - df['bbl']
    df = df.dropna()

    # Regime labeling with quantiles and looser thresholds
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

    print("Label distribution:", collections.Counter(df['regime']))

    # LSTM Prep
    window = 8 if intvl == '4h' else (24 if intvl == '1h' else 3)  # Adjust lookback per interval
    X_raw = df[['close','returns','rsi','macd','bbh','bbl','bb_width']].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_raw)
    X, y = [], []
    for i in range(window, len(X_scaled)):
        X.append(X_scaled[i - window:i])
        y.append(df['regime'].iloc[i])
    X, y = np.array(X), np.array(y)

    X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=False, test_size=0.15)

    # LSTM model
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
    model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=50, batch_size=32, callbacks=callbacks, verbose=0)

    # Regime prediction on latest available test set
    y_pred = np.argmax(model.predict(X_test), axis=1)
    signals = []
    for idx, val in enumerate(y_pred):
        if val == 0:
            regime = "Consolidating"
        elif val == 1:
            regime = "Bullish"
        elif val == 2:
            regime = "Bearish"
        signals.append({'date': df.index[len(df) - len(y_pred) + idx], 'regime': regime})

    regime_predictions[name] = signals
    # Example performance metric: bullish signal frequency
    performance_summary[name] = sum([s['regime']=="Bullish" for s in signals])

# Display regime predictions for each timeframe
for tf, preds in regime_predictions.items():
    print(f"\n{tf} REGIME SIGNALS:")
    for sig in preds[-20:]:  # Last 20 signals
        print(f"Date: {sig['date']} - Regime: {sig['regime']}")

# Pick best timeframe based on highest "Bullish" frequency
best_tf = max(performance_summary, key=performance_summary.get)
print(f"\nBest timeframe for current regime detection: {best_tf}")

# --- Statistical target profit analysis when bullish regime detected ---
# Use best interval's results:
df_bull = yf.download(ticker, period='180d', interval=intervals[best_tf]).rename(columns={'Close': 'close'})
df_bull['returns'] = df_bull['close'].pct_change()
close_series = df_bull['close'].squeeze()
boll = BollingerBands(close_series)
df_bull['bbh'] = boll.bollinger_hband()
df_bull['bbl'] = boll.bollinger_lband()
df_bull['bb_width'] = df_bull['bbh'] - df_bull['bbl']
upper = df_bull['returns'].quantile(0.7)
center = df_bull['bb_width'].median() / 2
labels = []
for i in range(len(df_bull)):
    if df_bull['returns'].iloc[i] > upper and df_bull['bb_width'].iloc[i] > center:
        labels.append(1)
    else:
        labels.append(0)
df_bull['regime'] = labels

# For each bullish regime, measure the subsequent move (e.g., next bar return)
bullish_moves = []
for i in range(len(df_bull)-1):
    if df_bull['regime'].iloc[i] == 1:
        move = df_bull['close'].iloc[i+1] - df_bull['close'].iloc[i]
        bullish_moves.append(move)

# Regression model predicting next-move magnitude (statistical study)
if len(bullish_moves) > 10:
    X_reg = np.arange(len(bullish_moves)).reshape(-1,1)
    y_reg = np.array(bullish_moves)
    reg_model = sm.OLS(y_reg, sm.add_constant(X_reg)).fit()
    pred_move = reg_model.predict([1,len(bullish_moves)+1])[-1]
    print(f"\nPredicted next bullish regime move magnitude: {pred_move:.2f} dollars")
    print("Bullish regime move (mean):", np.mean(bullish_moves))
    print("Bullish regime move (median):", np.median(bullish_moves))
else:
    print("\nNot enough bullish regimes for profit prediction statistics.")

    