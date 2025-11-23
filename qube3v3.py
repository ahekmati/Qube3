from ib_insync import *
import pandas as pd
import numpy as np
import pandas_ta as ta
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, LSTM, Conv1D, MaxPooling1D, Flatten, Dense
from colorama import init, Fore, Style
import csv
import os

init(autoreset=True)

# Configurable thresholds
LSTM_BUY_THRESHOLD = 0.010    # 1% up
LSTM_SELL_THRESHOLD = -0.010  # 1% down
RSI_OVERBOUGHT = 80
RSI_OVERSOLD = 20
ADX_TREND = 25
BB_PCTB_HIGH = 0.96
BB_PCTB_LOW = 0.04
CROSS_SCORE_REQUIRED = 3   # At least 2/3 agree with model for strong filter

tickers = ["QQQ", "SPY"]
exchange = "SMART"
currency = "USD"
lookback = 30
bar_size = "1 day"
duration = "3 Y"
ma_cross_pairs = [(5,20), (9,21), (10,50), (20,50), (50,200)]

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

def add_crossovers(df, ma_cross_pairs, prefix=""):
    for fast, slow in ma_cross_pairs:
        fcol, scol = f"{prefix}ema{fast}", f"{prefix}ema{slow}"
        up_col = f"{prefix}ma{fast}_{slow}_cross_up"
        dn_col = f"{prefix}ma{fast}_{slow}_cross_down"
        df[up_col] = ((df[fcol].shift(1) < df[scol].shift(1)) & (df[fcol] > df[scol])).astype(int)
        df[dn_col] = ((df[fcol].shift(1) > df[scol].shift(1)) & (df[fcol] < df[scol])).astype(int)
    return df

def indicator_pipeline(df):
    for fast, slow in ma_cross_pairs:
        df[f"ema{fast}"] = df['close'].ewm(span=fast, adjust=False).mean()
        df[f"ema{slow}"] = df['close'].ewm(span=slow, adjust=False).mean()
    df["rsi"] = ta.rsi(df["close"], 14)
    macd = ta.macd(df["close"])
    df["macd"], df["macdsignal"], df["macdhist"] = macd["MACD_12_26_9"], macd["MACDs_12_26_9"], macd["MACDh_12_26_9"]
    bb = ta.bbands(df["close"])
    try:
        bbu, bbl = [c for c in bb.columns if "BBU" in c][0], [c for c in bb.columns if "BBL" in c][0]
        df['bb_upper'], df['bb_lower'] = bb[bbu], bb[bbl]
        df['bb_pctb'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
    except Exception:
        df['bb_upper'], df['bb_lower'], df['bb_pctb'] = np.nan, np.nan, np.nan
    df['atr'] = ta.atr(df['high'], df['low'], df['close'])
    df['adx'] = ta.adx(df['high'], df['low'], df['close'])['ADX_14']
    try:
        df['obv'] = ta.obv(df['close'], df['volume'])
    except Exception:
        df['obv'] = np.nan
    try:
        df['vwap'] = ta.vwap(df['high'], df['low'], df['close'], df['volume'])
    except Exception:
        df['vwap'] = np.nan
    df["roc"] = ta.roc(df["close"], 10)
    df["volume_norm"] = df["volume"] / df["volume"].rolling(20).mean()
    df = add_crossovers(df, ma_cross_pairs)
    return df

def resample_4h_and_enrich(df):
    df_4h = df.resample('4h').agg({
        'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'
    }).dropna()
    df_4h = indicator_pipeline(df_4h)
    df_4h = add_crossovers(df_4h, ma_cross_pairs)
    df_4h = df_4h.rename(columns={c: f"4h_{c}" for c in df_4h.columns})
    return df_4h

def indicator_agreement(direction, rsi, macd, adx, bb_pctb, cross_sum):
    agree = 0
    if direction == 'buy':
        if rsi is not None and rsi < RSI_OVERSOLD:
            agree += 1
        if macd is not None and macd > 0:
            agree += 1
        if adx is not None and adx > ADX_TREND:
            agree += 1
        if bb_pctb is not None and bb_pctb < BB_PCTB_LOW:
            agree += 1
        agree += cross_sum
    elif direction == 'sell':
        if rsi is not None and rsi > RSI_OVERBOUGHT:
            agree += 1
        if macd is not None and macd < 0:
            agree += 1
        if adx is not None and adx > ADX_TREND:
            agree += 1
        if bb_pctb is not None and bb_pctb > BB_PCTB_HIGH:
            agree += 1
        agree += cross_sum
    return agree

def log_to_csv(filename, row):
    exists = os.path.isfile(filename)
    with open(filename, 'a', newline='') as f:
        w = csv.writer(f)
        if not exists:
            w.writerow(list(row.keys()))
        w.writerow(list(row.values()))

ib = IB()
IB_HOST = '127.0.0.1'
IB_PORT = 4001
IB_CLIENT_ID = 123

ib.connect(IB_HOST, IB_PORT, IB_CLIENT_ID)
for ticker in tickers:
    print(f"\n==== {ticker} ====")
    df = fetch_ibkr_ohlcv(ib, ticker, exchange, currency, duration, bar_size)
    if df is None or len(df) < (lookback + 2):
        print("No data for this timeframe.")
        continue
    df.index = pd.to_datetime(df.index)
    df = indicator_pipeline(df)
    df_4h = resample_4h_and_enrich(df)
    df_4h_ff = df_4h.reindex(df.index, method='ffill')
    for col in df_4h.columns:
        df[col] = df_4h_ff[col]
    features = [
        "close", "high", "low", "volume_norm", "obv", "vwap", "rsi", "macd", "macdsignal", "macdhist",
        "bb_upper", "bb_lower", "bb_pctb", "atr", "adx", "roc"
    ]
    for fast, slow in ma_cross_pairs:
         features.append(f"ma{fast}_{slow}_cross_up")
         features.append(f"ma{fast}_{slow}_cross_down")
         features.append(f"4h_ma{fast}_{slow}_cross_up")
         features.append(f"4h_ma{fast}_{slow}_cross_down")
    for ind in ['rsi','macd','macdsignal','macdhist','bb_upper','bb_lower','bb_pctb','atr','adx','roc','volume_norm']:
        features.append(f"4h_{ind}")
    features_present = [f for f in features if f in df.columns]
    missing = [f for f in features if f not in df.columns]
    if missing:
        print(f"Warning: Missing columns in DataFrame and will skip dropna on them: {missing}")
    df = df.dropna(subset=features_present)
    if df.empty or len(df) < lookback + 2:
        print("Not enough data after filtering and indicators.")
        continue
    scaler = MinMaxScaler()
    X = scaler.fit_transform(df[features_present])
    y = df["close"].values
    X_seq, y_seq = [], []
    for i in range(lookback, len(df)):
        X_seq.append(X[i-lookback:i])
        y_seq.append(y[i])
    X_seq, y_seq = np.array(X_seq), np.array(y_seq)
    split_idx = int(len(X_seq)*0.8)
    X_train, X_test = X_seq[:split_idx], X_seq[split_idx:]
    y_train, y_test = y_seq[:split_idx], y_seq[split_idx:]
    model = Sequential()
    model.add(Input(shape=(lookback, X_seq.shape[2])))
    model.add(LSTM(32, return_sequences=True, activation='tanh'))
    model.add(Conv1D(32, 2, padding='same', activation='relu'))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Flatten())
    model.add(Dense(20, activation='relu'))
    model.add(Dense(1, activation='linear'))
    model.compile(optimizer='adam', loss='mse')
    model.fit(X_train, y_train, epochs=8, batch_size=32, validation_data=(X_test, y_test), verbose=0)
    pred = model.predict(X_test[-2:]).flatten()
    pred_move = (pred[-1] - pred[-2]) / abs(pred[-2])
    actual_close = y_test[-1]

    # Collect latest indicator values (None if missing)
    D = df.iloc[-1]
    D4 = df.iloc[-1]  # Holds 4h columns too after ffill
    card = dict(
        Model_Pred="{:.2%}".format(pred_move), Close="{:.2f}".format(actual_close),
        RSI="{:.2f}".format(D.get('rsi', np.nan)), MACD="{:.2f}".format(D.get('macd', np.nan)),
        ADX="{:.2f}".format(D.get('adx', np.nan)), BB_PctB="{:.2f}".format(D.get('bb_pctb', np.nan)),
        RSI_4H="{:.2f}".format(D4.get('4h_rsi', np.nan)), MACD_4H="{:.2f}".format(D4.get('4h_macd', np.nan)),
        ADX_4H="{:.2f}".format(D4.get('4h_adx', np.nan)), BB_PctB_4H="{:.2f}".format(D4.get('4h_bb_pctb', np.nan))
    )

    print(Style.BRIGHT + "DASHBOARD:", " | ".join(f"{k}: {v}" for k,v in card.items()))

    # Agreement logic (daily)
    cross_agree = 0
    for fast, slow in ma_cross_pairs:
        cross_up = D.get(f"ma{fast}_{slow}_cross_up", 0)
        cross_dn = D.get(f"ma{fast}_{slow}_cross_down", 0)
        if pred_move > 0:
            cross_agree += int(cross_up)
        elif pred_move < 0:
            cross_agree += int(cross_dn)
    
    # Agreement logic (4h)
    cross_4h_agree = 0
    for fast, slow in ma_cross_pairs:
        cross_up = D4.get(f"4h_ma{fast}_{slow}_cross_up", 0)
        cross_dn = D4.get(f"4h_ma{fast}_{slow}_cross_down", 0)
        if pred_move > 0:
            cross_4h_agree += int(cross_up)
        elif pred_move < 0:
            cross_4h_agree += int(cross_dn)

    direction = 'buy' if pred_move > 0 else 'sell'
    agree_score = indicator_agreement(direction, D.get('rsi'), D.get('macd'), D.get('adx'), D.get('bb_pctb'), cross_agree)
    agree_4h_score = indicator_agreement(direction, D4.get('4h_rsi'), D4.get('4h_macd'), D4.get('4h_adx'), D4.get('4h_bb_pctb'), cross_4h_agree)
    # Print all
    print(f"Agreement score (daily): {agree_score} | Agreement score (4h): {agree_4h_score} (of ~5)")
    # Signal summary
    filter_msg = None
    if pred_move > LSTM_BUY_THRESHOLD and agree_score >= CROSS_SCORE_REQUIRED:
        print(Fore.GREEN + Style.BRIGHT + "STRONG BUY FILTER: LSTM & KEY INDICATORS AGREE (Daily)")
        filter_msg = "Strong Buy Daily"
    elif pred_move < LSTM_SELL_THRESHOLD and agree_score >= CROSS_SCORE_REQUIRED:
        print(Fore.RED + Style.BRIGHT + "STRONG SELL FILTER: LSTM & KEY INDICATORS AGREE (Daily)")
        filter_msg = "Strong Sell Daily"
    elif pred_move > LSTM_BUY_THRESHOLD and agree_4h_score >= CROSS_SCORE_REQUIRED:
        print(Fore.GREEN + Style.BRIGHT + "STRONG BUY FILTER: LSTM & INDICATORS AGREE (4H)")
        filter_msg = "Strong Buy 4h"
    elif pred_move < LSTM_SELL_THRESHOLD and agree_4h_score >= CROSS_SCORE_REQUIRED:
        print(Fore.RED + Style.BRIGHT + "STRONG SELL FILTER: LSTM & INDICATORS AGREE (4H)")
        filter_msg = "Strong Sell 4h"
    else:
        print(Fore.WHITE + "No strong agreement filter signal today.")
        filter_msg = "No strong signal"
    # CSV LOG/ROLLING TRACK
    log_to_csv(f"{ticker}_filter_signals.csv", {
        "date": D.name,
        "close": actual_close,
        "pred_move": pred_move,
        "rsi": D.get('rsi', np.nan),
        "macd": D.get('macd', np.nan),
        "adx": D.get('adx', np.nan),
        "bb_pctb": D.get('bb_pctb', np.nan),
        "rsi_4h": D4.get('4h_rsi', np.nan),
        "macd_4h": D4.get('4h_macd', np.nan),
        "adx_4h": D4.get('4h_adx', np.nan),
        "bb_pctb_4h": D4.get('4h_bb_pctb', np.nan),
        "agree_score": agree_score,
        "agree_4h_score": agree_4h_score,
        "filter_signal": filter_msg
    })

ib.disconnect()
