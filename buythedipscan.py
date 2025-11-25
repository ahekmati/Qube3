from ib_insync import *
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from colorama import init, Fore, Style

init(autoreset=True)

IB_HOST = '127.0.0.1'
IB_PORT = 4001
IB_CLIENT_ID = 13

symbols = ['TQQQ', 'SSO']
exchange = 'SMART'
currency = 'USD'

# Empirical averages from earlier study (in %), for 1D and 4H
EXPECTED = {
    ('QQQ', '1D'): {'avg_dip': -2.927018, 'avg_reb': 2.468717},
    ('QQQ', '4H'): {'avg_dip': -2.152439, 'avg_reb': 1.299490},
    ('SPY', '1D'): {'avg_dip': -2.489006, 'avg_reb': 1.993848},
    ('SPY', '4H'): {'avg_dip': -1.952473, 'avg_reb': 1.108578},
}

# Sizeable-dip thresholds (rule-based)
DIP_THRESHOLDS = {
    ('QQQ', '1D'): -2.0,
    ('QQQ', '4H'): -1.5,
    ('SPY', '1D'): -2.0,
    ('SPY', '4H'): -1.5,
}

# EMA settings (used for indicators and daily bear regimes)
EMA_FILTERS = {
    '1D': {'ema_fast': 9,  'ema_slow': 12},
    '4H': {'ema_fast': 9,  'ema_slow': 26},
}

STOP_LOSS_PCT = 2.0  # 2% below dip low
POSITION_SIZE = 100  # shares

ib = IB()
ib.connect(IB_HOST, IB_PORT, IB_CLIENT_ID)


def to_df(bars):
    df = util.df(bars)
    if df.empty:
        return df
    df.rename(columns=str.lower, inplace=True)
    df.set_index('date', inplace=True)
    df.index = pd.to_datetime(df.index)
    return df.sort_index()


def fetch_ib_history(symbol, bar_size, duration):
    contract = Stock(symbol, exchange, currency)
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
        return pd.DataFrame()
    return to_df(bars)


def add_indicators(df, tf_label):
    close = df['close']
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    roll_up = gain.rolling(14).mean()
    roll_down = loss.rolling(14).mean()
    rs = roll_up / (roll_down + 1e-9)
    rsi = 100 - (100 / (1 + rs))
    df['rsi_14'] = rsi
    df['ret_1'] = close.pct_change()
    df['vol_10'] = df['ret_1'].rolling(10).std()
    df['ma_20'] = close.rolling(20).mean()
    df['dist_ma20'] = (close - df['ma_20']) / (df['ma_20'] + 1e-9) * 100.0

    filt = EMA_FILTERS.get(tf_label)
    if filt:
        ef = filt['ema_fast']
        es = filt['ema_slow']
        df[f'ema_fast_{ef}'] = close.ewm(span=ef, adjust=False).mean()
        df[f'ema_slow_{es}'] = close.ewm(span=es, adjust=False).mean()
    return df


def build_dip_events(df, tf_label):
    """
    Build all dip→rebound events with features.
    A "dip event" is:
      - local high, then a local low that turns up,
      - drop from high to low <= -1%.
    Rebound is measured to next local high.
    """
    closes = df['close'].values
    highs = df['high'].values if 'high' in df.columns else closes
    lows = df['low'].values if 'low' in df.columns else closes
    dates = df.index.to_list()
    last_high_idx = None
    events = []

    for i in range(1, len(closes)-1):
        # local high by close
        if closes[i] > closes[i-1] and closes[i] > closes[i+1]:
            last_high_idx = i

        # local low & turn-up
        if closes[i] < closes[i-1] and closes[i] < closes[i+1] and closes[i+1] > closes[i]:
            low_idx = i
            if last_high_idx is None or last_high_idx >= low_idx:
                continue

            high_price = closes[last_high_idx]
            low_price = closes[low_idx]
            dip_pct = (low_price - high_price) / high_price * 100.0

            if dip_pct > -1.0:
                continue

            # next local high after low
            next_high_idx = None
            j = low_idx + 1
            while j < len(closes)-1:
                if closes[j] > closes[j-1] and closes[j] > closes[j+1]:
                    next_high_idx = j
                    break
                j += 1
            if next_high_idx is None:
                continue

            next_high_price = closes[next_high_idx]
            rebound_pct = (next_high_price - low_price) / low_price * 100.0

            row = df.iloc[low_idx]
            ev = {
                'tf': tf_label,
                'low_date': dates[low_idx],
                'high_date': dates[last_high_idx],
                'next_high_date': dates[next_high_idx],
                'low_price': low_price,
                'high_price': high_price,
                'next_high_price': next_high_price,
                'dip_pct': dip_pct,
                'rebound_pct': rebound_pct,
                'rsi_14': row.get('rsi_14', np.nan),
                'vol_10': row.get('vol_10', np.nan),
                'ret_1': row.get('ret_1', np.nan),
                'dist_ma20': row.get('dist_ma20', np.nan),
                'low_idx': low_idx,
                'next_high_idx': next_high_idx
            }
            events.append(ev)
    return events


def train_models(events):
    """
    Train classifier (dip quality) and regressor (rebound) from events.
    """
    df = pd.DataFrame(events)
    df = df.dropna(subset=['dip_pct', 'rebound_pct', 'rsi_14', 'vol_10', 'ret_1', 'dist_ma20'])
    if len(df) < 40:
        return None, None, None

    # classification label: good dip if rebound >= max(1%, 0.5 * |dip|)
    df['good_dip'] = (df['rebound_pct'] >= np.maximum(1.0, 0.5 * df['dip_pct'].abs())).astype(int)

    features = ['dip_pct', 'rsi_14', 'vol_10', 'ret_1', 'dist_ma20']
    X = df[features].values
    y_reg = df['rebound_pct'].values
    y_clf = df['good_dip'].values

    clf = RandomForestClassifier(
        n_estimators=200,
        max_depth=5,
        class_weight='balanced',
        random_state=42
    )
    clf.fit(X, y_clf)

    reg = RandomForestRegressor(
        n_estimators=200,
        max_depth=5,
        random_state=42
    )
    reg.fit(X, y_reg)

    return clf, reg, df


def find_latest_candidate_low(df, tf_label):
    """
    Find the latest local low + turn-up in the series, with its features.
    """
    df = add_indicators(df.copy(), tf_label)
    closes = df['close'].values
    dates = df.index.to_list()

    for i in range(len(closes)-2, 1, -1):  # scan backwards
        if closes[i] < closes[i-1] and closes[i] < closes[i+1] and closes[i+1] > closes[i]:
            low_idx = i
            # find last local high before low
            last_high_idx = None
            j = low_idx - 1
            while j > 0:
                if closes[j] > closes[j-1] and closes[j] > closes[j+1]:
                    last_high_idx = j
                    break
                j -= 1
            if last_high_idx is None:
                return None
            high_price = closes[last_high_idx]
            low_price = closes[low_idx]
            dip_pct = (low_price - high_price) / high_price * 100.0
            row = df.iloc[low_idx]
            cand = {
                'low_idx': low_idx,
                'high_idx': last_high_idx,
                'low_date': dates[low_idx],
                'high_date': dates[last_high_idx],
                'low_price': low_price,
                'high_price': high_price,
                'dip_pct': dip_pct,
                'rsi_14': row.get('rsi_14', np.nan),
                'vol_10': row.get('vol_10', np.nan),
                'ret_1': row.get('ret_1', np.nan),
                'dist_ma20': row.get('dist_ma20', np.nan)
            }
            return cand
    return None


def build_bear_ema_regimes_daily(df):
    ef, es = 9, 12
    ema_fast = df[f'ema_fast_{ef}']
    ema_slow = df[f'ema_slow_{es}']

    closes = df['close'].values
    dates = df.index.to_list()

    regimes = []
    in_regime = False
    start_idx = None

    for i in range(1, len(df)):
        if not in_regime:
            if ema_fast.iloc[i-1] >= ema_slow.iloc[i-1] and ema_fast.iloc[i] < ema_slow.iloc[i]:
                in_regime = True
                start_idx = i
        else:
            if ema_fast.iloc[i-1] <= ema_slow.iloc[i-1] and ema_fast.iloc[i] > ema_slow.iloc[i]:
                segment = df.iloc[start_idx:i+1]
                low_idx = segment['close'].idxmin()
                low_price = df.loc[low_idx, 'close']
                cross_price = closes[start_idx]
                dd_pct = (low_price - cross_price) / cross_price * 100.0
                regimes.append({
                    'cross_date': dates[start_idx],
                    'cross_price': cross_price,
                    'low_date': low_idx,
                    'low_price': low_price,
                    'dd_pct': dd_pct
                })
                in_regime = False
                start_idx = None

    return regimes


def train_bear_regime_ml(regimes, df):
    if not regimes:
        return None, None

    rows = []
    for reg in regimes:
        date = reg['cross_date']
        if date not in df.index:
            continue
        row = df.loc[date]
        rows.append({
            'dd_pct': reg['dd_pct'],
            'rsi_14': row.get('rsi_14', np.nan),
            'vol_10': row.get('vol_10', np.nan),
            'ret_1': row.get('ret_1', np.nan),
            'dist_ma20': row.get('dist_ma20', np.nan)
        })
    reg_df = pd.DataFrame(rows).dropna()
    if len(reg_df) < 20:
        return None, None

    X = reg_df[['rsi_14', 'vol_10', 'ret_1', 'dist_ma20']].values
    y = reg_df['dd_pct'].values
    model = RandomForestRegressor(
        n_estimators=200,
        max_depth=5,
        random_state=42
    )
    model.fit(X, y)
    return model, reg_df


def check_daily_bear_ema_regimes(symbol, daily_df):
    print(f"\n---- {symbol} DAILY 9<12 EMA BEAR REGIMES ----")
    df = add_indicators(daily_df.copy(), '1D')
    ef, es = 9, 12
    if f'ema_fast_{ef}' not in df.columns or f'ema_slow_{es}' not in df.columns:
        print("Daily EMAs missing, skipping bear regime analysis.")
        return

    regimes = build_bear_ema_regimes_daily(df)
    if not regimes:
        print("No completed 9<12 EMA regimes found.")
        return

    model, reg_df = train_bear_regime_ml(regimes, df)
    avg_dd_emp = np.mean([r['dd_pct'] for r in regimes])
    print(f"Empirical avg drawdown during 9<12 EMA regimes: {avg_dd_emp:.2f}% over {len(regimes)} regimes.")

    print("\nLast 3 completed 9<12→12>9 regimes (most recent last):")
    for reg in regimes[-3:]:
        cd = reg['cross_date']
        cp = reg['cross_price']
        ld = reg['low_date']
        lp = reg['low_price']
        dd = reg['dd_pct']
        if model is not None and cd in df.index:
            row = df.loc[cd]
            feat_vec = np.array([[row.get('rsi_14', np.nan),
                                  row.get('vol_10', np.nan),
                                  row.get('ret_1', np.nan),
                                  row.get('dist_ma20', np.nan)]])
            ml_dd = float(model.predict(feat_vec)[0])
            pred_low = cp * (1 + ml_dd/100.0)
            print(f"Cross {cd.date()} @ {cp:.2f} | Actual low {ld.date()} @ {lp:.2f} (DD {dd:.2f}%) | "
                  f"ML DD {ml_dd:.2f}% -> predicted low ≈ {pred_low:.2f}")
        else:
            print(f"Cross {cd.date()} @ {cp:.2f} | Actual low {ld.date()} @ {lp:.2f} (DD {dd:.2f}%) | ML: n/a")

    ema_fast = df[f'ema_fast_{ef}']
    ema_slow = df[f'ema_slow_{es}']
    in_regime = False
    start_idx = None
    for i in range(1, len(df)):
        if not in_regime:
            if ema_fast.iloc[i-1] >= ema_slow.iloc[i-1] and ema_fast.iloc[i] < ema_slow.iloc[i]:
                in_regime = True
                start_idx = i
        else:
            if ema_fast.iloc[i-1] <= ema_slow.iloc[i-1] and ema_fast.iloc[i] > ema_slow.iloc[i]:
                in_regime = False
                start_idx = None

    if in_regime and start_idx is not None:
        cross_date = df.index[start_idx]
        cross_price = df['close'].iloc[start_idx]
        current_price = df['close'].iloc[-1]
        since_dd = (current_price - cross_price) / cross_price * 100.0
        print(f"\nCurrent 9<12 EMA regime active since {cross_date.date()} @ {cross_price:.2f}.")
        print(f"Price is now {current_price:.2f}, change since cross: {since_dd:.2f}%")
        if model is not None and cross_date in df.index:
            row = df.loc[cross_date]
            feat_vec = np.array([[row.get('rsi_14', np.nan),
                                  row.get('vol_10', np.nan),
                                  row.get('ret_1', np.nan),
                                  row.get('dist_ma20', np.nan)]])
            ml_dd = float(model.predict(feat_vec)[0])
            pred_low = cross_price * (1 + ml_dd/100.0)
            print(f"Empirical avg DD from cross: {avg_dd_emp:.2f}%")
            print(f"ML-predicted DD from cross:  {ml_dd:.2f}% -> predicted swing low ≈ {pred_low:.2f}")
        else:
            print("No ML model or cross bar missing; only empirical avg DD is available.")
    else:
        print("\nNo active 9<12 EMA regime right now.")


def simulate_trades_for_dips(df, events, symbol, tf_label, clf, reg, hist_avg_reb):
    """
    For the last 5 dips:
      - Buy 100 shares at low_price.
      - Stop at 2% below low_price.
      - Two targets: empirical_avg_reb and ML-predicted reb for that dip.
    Checks which is hit first (within the window up to next_high_idx).
    Prints colored results and totals.
    """
    key = (symbol, tf_label)
    exp_entry = EXPECTED.get(key)
    if exp_entry:
        emp_reb = exp_entry['avg_reb']
    else:
        emp_reb = hist_avg_reb

    print("\n=== 2% Stop-loss Trade Simulation (100 shares) ===")
    last_events = events[-5:]
    total_emp_pnl = 0.0
    total_ml_pnl = 0.0

    closes = df['close'].values
    highs = df['high'].values if 'high' in df.columns else closes
    lows = df['low'].values if 'low' in df.columns else closes

    for ev in last_events:
        low_idx = ev['low_idx']
        end_idx = ev['next_high_idx']
        low_price = ev['low_price']
        date = ev['low_date']

        stop_price = low_price * (1 - STOP_LOSS_PCT / 100.0)
        emp_target = low_price * (1 + emp_reb / 100.0)

        # ML target for this dip
        feat_vec = np.array([[ev['dip_pct'], ev['rsi_14'], ev['vol_10'], ev['ret_1'], ev['dist_ma20']]])
        ml_reb = float(reg.predict(feat_vec)[0])
        ml_target = low_price * (1 + ml_reb / 100.0)

        # simulate empirical trade
        emp_hit = None  # 'TP', 'SL', or None
        for i in range(low_idx+1, end_idx+1):
            if lows[i] <= stop_price:
                emp_hit = 'SL'
                break
            if highs[i] >= emp_target:
                emp_hit = 'TP'
                break
        if emp_hit == 'TP':
            pnl_emp = (emp_target - low_price) * POSITION_SIZE
            total_emp_pnl += pnl_emp
            print(
                Fore.GREEN
                + f"EMP {symbol} {tf_label} {date}: "
                  f"ENTRY={low_price:.2f}, STOP={stop_price:.2f}, TARGET={emp_target:.2f}, "
                  f"RESULT=TP, PnL={pnl_emp:.2f}"
            )
        elif emp_hit == 'SL':
            pnl_emp = (stop_price - low_price) * POSITION_SIZE
            total_emp_pnl += pnl_emp
            print(
                Fore.RED
                + f"EMP {symbol} {tf_label} {date}: "
                  f"ENTRY={low_price:.2f}, STOP={stop_price:.2f}, TARGET={emp_target:.2f}, "
                  f"RESULT=STOP, PnL={pnl_emp:.2f}"
            )
        else:
            exit_price = ev['next_high_price']
            pnl_emp = (exit_price - low_price) * POSITION_SIZE
            total_emp_pnl += pnl_emp
            color = Fore.GREEN if pnl_emp >= 0 else Fore.RED
            print(
                color
                + f"EMP {symbol} {tf_label} {date}: "
                  f"ENTRY={low_price:.2f}, STOP={stop_price:.2f}, TARGET={emp_target:.2f}, "
                  f"RESULT=EXIT@{exit_price:.2f}, PnL={pnl_emp:.2f}"
            )

        # simulate ML trade
        ml_hit = None
        for i in range(low_idx+1, end_idx+1):
            if lows[i] <= stop_price:
                ml_hit = 'SL'
                break
            if highs[i] >= ml_target:
                ml_hit = 'TP'
                break
        if ml_hit == 'TP':
            pnl_ml = (ml_target - low_price) * POSITION_SIZE
            total_ml_pnl += pnl_ml
            print(
                Fore.GREEN
                + f"ML  {symbol} {tf_label} {date}: "
                  f"ENTRY={low_price:.2f}, STOP={stop_price:.2f}, TARGET={ml_target:.2f}, "
                  f"RESULT=TP, PnL={pnl_ml:.2f}"
            )
        elif ml_hit == 'SL':
            pnl_ml = (stop_price - low_price) * POSITION_SIZE
            total_ml_pnl += pnl_ml
            print(
                Fore.RED
                + f"ML  {symbol} {tf_label} {date}: "
                  f"ENTRY={low_price:.2f}, STOP={stop_price:.2f}, TARGET={ml_target:.2f}, "
                  f"RESULT=STOP, PnL={pnl_ml:.2f}"
            )
        else:
            exit_price = ev['next_high_price']
            pnl_ml = (exit_price - low_price) * POSITION_SIZE
            total_ml_pnl += pnl_ml
            color = Fore.GREEN if pnl_ml >= 0 else Fore.RED
            print(
                color
                + f"ML  {symbol} {tf_label} {date}: "
                  f"ENTRY={low_price:.2f}, STOP={stop_price:.2f}, TARGET={ml_target:.2f}, "
                  f"RESULT=EXIT@{exit_price:.2f}, PnL={pnl_ml:.2f}"
            )

    print(Style.BRIGHT + f"\nTotal EMP PnL ({symbol} {tf_label}, last 5 dips, 100sh): {total_emp_pnl:.2f}")
    print(Style.BRIGHT + f"Total ML  PnL ({symbol} {tf_label}, last 5 dips, 100sh): {total_ml_pnl:.2f}")


def check_symbol_timeframe(symbol, bar_size, duration, tf_label):
    print(f"\n======== {symbol} {tf_label} CHECK ========")
    raw = fetch_ib_history(symbol, bar_size, duration)
    if raw.empty or 'close' not in raw.columns:
        print("No data.")
        return

    df = add_indicators(raw.copy(), tf_label)
    events = build_dip_events(df, tf_label)
    if not events:
        print("No historical dips found.")
        return

    clf, reg, train_df = train_models(events)
    if clf is None:
        print("Not enough clean samples to train ML.")
        return

    hist_avg_dip = train_df['dip_pct'].mean()
    hist_avg_reb = train_df['rebound_pct'].mean()
    print(f"Historical avg dip: {hist_avg_dip:.2f}%, avg rebound: {hist_avg_reb:.2f}% on {len(train_df)} events.")

    print("\nLast 5 sizeable dips (unfiltered, most recent last):")
    last_events = events[-5:]
    for ev in last_events:
        feat_vec = np.array([[ev['dip_pct'], ev['rsi_14'], ev['vol_10'], ev['ret_1'], ev['dist_ma20']]])
        prob_good = float(clf.predict_proba(feat_vec)[0, 1])
        ml_reb = float(reg.predict(feat_vec)[0])
        ml_target = ev['low_price'] * (1 + ml_reb/100.0)
        print(
            f"Low {ev['low_date']} @ {ev['low_price']:.2f} | "
            f"Dip {ev['dip_pct']:.2f}% | Reb {ev['rebound_pct']:.2f}% | "
            f"ML good-prob {prob_good*100:.1f}% | ML reb {ml_reb:.2f}% -> tgt≈{ml_target:.2f}"
        )

    latest = find_latest_candidate_low(raw, tf_label)
    if latest is None:
        print("\nNo fresh local low candidate.")
        return

    key = (symbol, tf_label)
    dip_cut = DIP_THRESHOLDS.get(key, -1.5)
    if latest['dip_pct'] > dip_cut:
        print(f"\nLatest local low dip {latest['dip_pct']:.2f}% is smaller than threshold {dip_cut:.2f}% -> not flagged as sizeable dip.")
        return

    print(f"\nRule dip detected: low {latest['low_date']} @ {latest['low_price']:.2f}, dip {latest['dip_pct']:.2f}% (threshold {dip_cut:.2f}%).")

    feat_vec = np.array([[latest['dip_pct'], latest['rsi_14'], latest['vol_10'], latest['ret_1'], latest['dist_ma20']]])
    prob_good = float(clf.predict_proba(feat_vec)[0, 1])
    ml_reb = float(reg.predict(feat_vec)[0])

    exp_entry = EXPECTED.get(key)
    if exp_entry:
        emp_dip = exp_entry['avg_dip']
        emp_reb = exp_entry['avg_reb']
    else:
        emp_dip = hist_avg_dip
        emp_reb = hist_avg_reb

    print("\n--- Unfiltered empirical vs ML ---")
    print(f"ML dip quality prob (good dip): {prob_good*100:.1f}%")
    print(f"Empirical avg dip:     {emp_dip:.2f}%")
    print(f"Empirical avg rebound: {emp_reb:.2f}%")
    ml_target = latest['low_price'] * (1 + ml_reb/100.0)
    print(f"ML-predicted rebound:  {ml_reb:.2f}%  -> target ≈ {ml_target:.2f}")
    print(f"(Training set mean rebound (unfiltered): {hist_avg_reb:.2f}%.)")

    simulate_trades_for_dips(raw, events, symbol, tf_label, clf, reg, hist_avg_reb)


# Run checks
for sym in symbols:
    # Daily
    daily_df = fetch_ib_history(sym, '1 day', '5 Y')
    if not daily_df.empty:
        check_symbol_timeframe(sym, '1 day', '5 Y', '1D')
        check_daily_bear_ema_regimes(sym, daily_df)
    else:
        print(f"No daily data for {sym}.")

    # 4H
    check_symbol_timeframe(sym, '4 hours', '2 Y', '4H')

ib.disconnect()
