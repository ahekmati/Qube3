from ib_insync import *
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from colorama import init, Fore, Style
import json
import os
from datetime import datetime


init(autoreset=True)


IB_HOST = '127.0.0.1'
IB_PORT = 4001
IB_CLIENT_ID = 13
ACCOUNT_ID = 'U22816462'


# Trade these symbols live
symbols = ['TQQQ', 'SSO']
exchange = 'SMART'
currency = 'USD'


# Empirical averages (you can refine later with TQQQ/SSO-specific stats)
EXPECTED = {
    ('TQQQ', '1D'): {'avg_dip': -6.54, 'avg_reb': 6.94},
    ('TQQQ', '4H'): {'avg_dip': -3.66, 'avg_reb': 3.08},
    ('SSO',  '1D'): {'avg_dip': -4.03, 'avg_reb': 3.64},
    ('SSO',  '4H'): {'avg_dip': -2.57, 'avg_reb': 1.68},
}


# Sizeable-dip thresholds (rule-based)
DIP_THRESHOLDS = {
    ('TQQQ', '1D'): -2.0,
    ('TQQQ', '4H'): -1.5,
    ('SSO',  '1D'): -2.0,
    ('SSO',  '4H'): -1.5,
}


# EMA settings (used for indicators and daily bear regimes)
EMA_FILTERS = {
    '1D': {'ema_fast': 9,  'ema_slow': 12},
    '4H': {'ema_fast': 9,  'ema_slow': 26},
}


STOP_LOSS_PCT = 2.0  # 2% below entry
TRAIL_PCT   = 1.0    # 1% trail below target/highest
POSITION_SIZE = 1    # shares


STATE_DIR = "state_live"
os.makedirs(STATE_DIR, exist_ok=True)


ib = IB()
ib.connect(IB_HOST, IB_PORT, IB_CLIENT_ID)
ib.account = ACCOUNT_ID  # default account


def state_path(symbol):
    return os.path.join(STATE_DIR, f"{symbol.lower()}_state.json")


def load_state(symbol):
    path = state_path(symbol)
    if not os.path.exists(path):
        return {
            "status": "flat",             # flat | order_placed | long_active
            "entry_source": None,
            "planned_entry": None,
            "entry_order_id": None,
            "stop_order_id": None,
            "fill_price": None,
            "stop_price": None,
            "target_price": None,
            "highest_since_target": None,
            "last_update_date": None,
        }
    with open(path, "r") as f:
        return json.load(f)


def save_state(symbol, state):
    path = state_path(symbol)
    with open(path, "w") as f:
        json.dump(state, f, indent=2, default=str)


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
    closes = df['close'].values
    highs  = df['high'].values if 'high' in df.columns else closes
    lows   = df['low'].values  if 'low'  in df.columns else closes
    dates  = df.index.to_list()
    last_high_idx = None
    events = []

    for i in range(1, len(closes)-1):
        if closes[i] > closes[i-1] and closes[i] > closes[i+1]:
            last_high_idx = i

        if closes[i] < closes[i-1] and closes[i] < closes[i+1] and closes[i+1] > closes[i]:
            low_idx = i
            if last_high_idx is None or last_high_idx >= low_idx:
                continue

            high_price = closes[last_high_idx]
            low_price  = closes[low_idx]
            dip_pct    = (low_price - high_price) / high_price * 100.0

            if dip_pct > -1.0:
                continue

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
            rebound_pct     = (next_high_price - low_price) / low_price * 100.0

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
    df = pd.DataFrame(events)
    df = df.dropna(subset=['dip_pct', 'rebound_pct', 'rsi_14', 'vol_10', 'ret_1', 'dist_ma20'])
    if len(df) < 40:
        return None, None, None

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
    df = add_indicators(df.copy(), tf_label)
    closes = df['close'].values
    dates  = df.index.to_list()

    for i in range(len(closes)-2, 1, -1):
        if closes[i] < closes[i-1] and closes[i] < closes[i+1] and closes[i+1] > closes[i]:
            low_idx = i
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
            low_price  = closes[low_idx]
            dip_pct    = (low_price - high_price) / high_price * 100.0
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
    dates  = df.index.to_list()

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
            pred_low = cp * (1 + ml_dd/100.0)
            print(f"Empirical avg DD from cross: {avg_dd_emp:.2f}%")
            print(f"ML-predicted DD from cross:  {ml_dd:.2f}% -> predicted swing low ≈ {pred_low:.2f}")
        else:
            print("No ML model or cross bar missing; only empirical avg DD is available.")
    else:
        print("\nNo active 9<12 EMA regime right now.")


def decide_entry(symbol, daily_latest_dip, h4_latest_dip, bear_predicted_low, daily_ema_9_lt_12, last_daily_date):
    def dip_is_recent(dip):
        if dip is None:
            return False
        dip_date = pd.to_datetime(dip['low_date']).normalize()
        last_date = pd.to_datetime(last_daily_date).normalize()
        age = (last_date - dip_date).days
        return 0 <= age <= 3

    if daily_ema_9_lt_12 and bear_predicted_low is not None:
        return bear_predicted_low, "bear_regime_swing_low"

    daily_ok = dip_is_recent(daily_latest_dip)
    h4_ok   = dip_is_recent(h4_latest_dip)

    if daily_ok and h4_ok:
        entry = (daily_latest_dip['low_price'] + h4_latest_dip['low_price']) / 2.0
        return entry, "avg_dip"

    if daily_ok:
        return daily_latest_dip['low_price'], "daily_dip"

    if h4_ok:
        return h4_latest_dip['low_price'], "h4_dip"

    return None, None


def place_limit_entry(symbol, price):
    contract = Stock(symbol, exchange, currency)
    order = LimitOrder(
        action='BUY',
        totalQuantity=POSITION_SIZE,
        lmtPrice=round(price, 2),
        tif='GTC'
    )
    order.account = ACCOUNT_ID
    trade = ib.placeOrder(contract, order)
    ib.sleep(1.0)
    print(Fore.CYAN + f"{symbol} LIVE: Placed LIMIT BUY GTC @ {price:.2f}, orderId={trade.order.orderId}")
    return trade.order.orderId


def place_stop_order(symbol, price, qty):
    contract = Stock(symbol, exchange, currency)
    order = Order()
    order.action = 'SELL'
    order.orderType = 'STP'
    order.tif = 'GTC'
    order.auxPrice = round(price, 2)
    order.totalQuantity = qty
    order.account = ACCOUNT_ID
    trade = ib.placeOrder(contract, order)
    ib.sleep(1.0)
    print(Fore.CYAN + f"{symbol} LIVE: Placed STOP SELL GTC @ {price:.2f}, qty={qty}, orderId={trade.order.orderId}")
    return trade.order.orderId


def modify_stop_order(order_id, new_price):
    open_trades = ib.openTrades()
    for t in open_trades:
        if t.order.orderId == order_id:
            t.order.auxPrice = round(new_price, 2)
            ib.placeOrder(t.contract, t.order)
            ib.sleep(1.0)
            return True
    return False


def get_position_info(symbol):
    positions = ib.positions()
    for p in positions:
        if p.contract.symbol == symbol and p.contract.secType == 'STK' and p.account == ACCOUNT_ID:
            return p.position, p.avgCost
    return 0, None


def has_live_entry_order(symbol, planned_price, side='BUY'):
    """
    Check IBKR for a live open limit order matching our planned entry.
    """
    open_trades = ib.openTrades()
    for t in open_trades:
        o = t.order
        c = t.contract
        if (
            c.symbol == symbol and
            c.secType == 'STK' and
            o.account == ACCOUNT_ID and
            o.action == side and
            o.orderType == 'LMT' and
            o.tif == 'GTC' and
            abs(o.lmtPrice - planned_price) < 1e-4 and
            o.totalQuantity == POSITION_SIZE
        ):
            return True
    return False


def check_entry_filled(symbol, state, emp_reb_daily):
    """
    For status=order_placed: consider entry filled if we now hold a long position in this symbol.
    """
    pos_size, avg_cost = get_position_info(symbol)
    if pos_size <= 0:
        print(f"{symbol} LIVE: No position yet; entry order still working.")
        return state

    fill_price  = avg_cost
    stop_price  = fill_price * (1 - STOP_LOSS_PCT/100.0)
    target_price = fill_price * (1 + emp_reb_daily/100.0)

    if state.get("stop_order_id") is None:
        stop_order_id = place_stop_order(symbol, stop_price, pos_size)
        state["stop_order_id"] = stop_order_id

    state["status"] = "long_active"
    state["fill_price"] = fill_price
    state["stop_price"] = stop_price
    state["target_price"] = target_price
    state["highest_since_target"] = None

    print(
        Fore.GREEN
        + f"{symbol} LIVE: Entry FILLED. fill_price={fill_price:.2f}, "
          f"stop={stop_price:.2f}, target={target_price:.2f}"
    )
    return state


def manage_active_position(symbol, state, current_price):
    pos_size, avg_cost = get_position_info(symbol)
    if pos_size <= 0:
        print(f"{symbol} LIVE: No position found; resetting state to flat.")
        state["status"] = "flat"
        state["planned_entry"] = None
        state["entry_order_id"] = None
        state["stop_order_id"] = None
        state["fill_price"] = None
        state["stop_price"] = None
        state["target_price"] = None
        state["highest_since_target"] = None
        return state

    fill_price = state.get("fill_price", avg_cost)
    stop_price = state.get("stop_price")
    target_price = state.get("target_price")
    highest_since_target = state.get("highest_since_target")
    stop_order_id = state.get("stop_order_id")

    if stop_price is None or target_price is None:
        print(f"{symbol} LIVE: Missing stop/target in state; nothing to manage.")
        return state

    print(
        f"{symbol} LIVE STATUS: current={current_price:.2f}, fill/avg={fill_price:.2f}, "
        f"stop={stop_price:.2f}, target={target_price:.2f}"
    )

    if current_price < target_price:
        min_stop = fill_price * (1 - STOP_LOSS_PCT/100.0)
        new_stop = max(stop_price, min_stop)
        if new_stop > stop_price and stop_order_id is not None:
            if modify_stop_order(stop_order_id, new_stop):
                state["stop_price"] = new_stop
                print(Fore.CYAN + f"{symbol} LIVE: Raised stop (pre-target) to {new_stop:.2f}")
        return state

    if highest_since_target is None:
        highest_since_target = current_price
    else:
        highest_since_target = max(highest_since_target, current_price)
    new_stop = highest_since_target * (1 - TRAIL_PCT/100.0)

    if new_stop > stop_price and stop_order_id is not None:
        if modify_stop_order(stop_order_id, new_stop):
            state["stop_price"] = new_stop
            state["highest_since_target"] = highest_since_target
            print(
                Fore.GREEN
                + f"{symbol} LIVE: TARGET reached; highest={highest_since_target:.2f}, "
                  f"new trailing stop={new_stop:.2f}"
            )
    else:
        state["highest_since_target"] = highest_since_target

    return state


def check_symbol_timeframe(symbol, bar_size, duration, tf_label):
    print(f"\n======== {symbol} {tf_label} CHECK ========")
    raw = fetch_ib_history(symbol, bar_size, duration)
    if raw.empty or 'close' not in raw.columns:
        print("No data.")
        return None, None, None, None

    df = add_indicators(raw.copy(), tf_label)
    events = build_dip_events(df, tf_label)
    if not events:
        print("No historical dips found.")
        return df, None, None, None

    clf, reg, train_df = train_models(events)
    if clf is None:
        print("Not enough clean samples to train ML.")
        return df, events, None, None

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
        return df, events, hist_avg_reb, None

    key = (symbol, tf_label)
    dip_cut = DIP_THRESHOLDS.get(key, -1.5)
    if latest['dip_pct'] > dip_cut:
        print(f"\nLatest local low dip {latest['dip_pct']:.2f}% is smaller than threshold {dip_cut:.2f}% -> not flagged as sizeable dip.")
        return df, events, hist_avg_reb, None

    print(f"\nRule dip detected: low {latest['low_date']} @ {latest['low_price']:.2f}, dip {latest['dip_pct']:.2f}% (threshold {dip_cut:.2f}%).")

    return df, events, hist_avg_reb, latest


all_messages = []

for sym in symbols:
    state = load_state(sym)

    daily_raw = fetch_ib_history(sym, '1 day', '5 Y')
    if daily_raw.empty:
        print(f"No daily data for {sym}.")
        continue

    daily_df = add_indicators(daily_raw.copy(), '1D')

    check_daily_bear_ema_regimes(sym, daily_df)

    regimes = build_bear_ema_regimes_daily(daily_df)
    bear_predicted_low = None
    daily_ema_9_lt_12 = False
    if regimes:
        model_bear, reg_df_bear = train_bear_regime_ml(regimes, daily_df)
        if model_bear is not None:
            ef, es = 9, 12
            ema_fast = daily_df[f'ema_fast_{ef}']
            ema_slow = daily_df[f'ema_slow_{es}']
            if ema_fast.iloc[-1] < ema_slow.iloc[-1]:
                daily_ema_9_lt_12 = True
                in_regime = False
                start_idx = None
                for i in range(1, len(daily_df)):
                    if not in_regime:
                        if ema_fast.iloc[i-1] >= ema_slow.iloc[i-1] and ema_fast.iloc[i] < ema_slow.iloc[i]:
                            in_regime = True
                            start_idx = i
                    else:
                        if ema_fast.iloc[i-1] <= ema_slow.iloc[i-1] and ema_fast.iloc[i] > ema_slow.iloc[i]:
                            in_regime = False
                            start_idx = None
                if in_regime and start_idx is not None:
                    cross_date = daily_df.index[start_idx]
                    cross_price = daily_df['close'].iloc[start_idx]
                    if cross_date in daily_df.index:
                        row = daily_df.loc[cross_date]
                        feat_vec = np.array([[row.get('rsi_14', np.nan),
                                              row.get('vol_10', np.nan),
                                              row.get('ret_1', np.nan),
                                              row.get('dist_ma20', np.nan)]])
                        ml_dd = float(model_bear.predict(feat_vec)[0])
                        bear_predicted_low = cross_price * (1 + ml_dd/100.0)
                        print(f"{sym} DAILY bear-regime predicted swing low for entry ≈ {bear_predicted_low:.2f}")

    daily_df2, daily_events, daily_emp_reb, daily_latest_dip = check_symbol_timeframe(sym, '1 day', '5 Y', '1D')
    h4_df, h4_events, h4_emp_reb_dummy, h4_latest_dip = check_symbol_timeframe(sym, '4 hours', '2 Y', '4H')

    if daily_df2 is None:
        continue

    key_daily = (sym, '1D')
    if EXPECTED.get(key_daily):
        emp_reb_daily = EXPECTED[key_daily]['avg_reb']
    else:
        emp_reb_daily = daily_emp_reb if daily_emp_reb is not None else 0.0

    last_daily_date = daily_df2.index[-1]
    last_close      = daily_df2['close'].iloc[-1]

    if state["status"] == "flat":
        entry_price, entry_source = decide_entry(
            sym,
            daily_latest_dip,
            h4_latest_dip,
            bear_predicted_low,
            daily_ema_9_lt_12,
            last_daily_date
        )
        if entry_price is not None:
            # If there is already a live GTC BUY LMT at this price, do NOT place another
            if has_live_entry_order(sym, entry_price, 'BUY'):
                print(
                    Fore.YELLOW
                    + f"{sym} LIVE: Existing GTC BUY LMT @ {entry_price:.2f} detected; "
                      f"not placing a new order."
                )
                state["status"] = "order_placed"
                state["planned_entry"] = entry_price
                state["entry_source"] = entry_source
            else:
                order_id = place_limit_entry(sym, entry_price)
                state["status"] = "order_placed"
                state["planned_entry"] = entry_price
                state["entry_source"] = entry_source
                state["entry_order_id"] = order_id
                state["stop_order_id"] = None
                state["fill_price"] = None
                state["stop_price"] = None
                state["target_price"] = None
                state["highest_since_target"] = None
                msg = f"{sym}: New LIMIT BUY GTC placed @ {entry_price:.2f} (source={entry_source})"
                all_messages.append(msg)
        else:
            print(f"\n{sym} DAILY: No entry decided (no recent dips / no bear swing-low).")

    elif state["status"] == "order_placed":
        state = check_entry_filled(sym, state, emp_reb_daily)

    elif state["status"] == "long_active":
        state = manage_active_position(sym, state, last_close)

    state["last_update_date"] = datetime.now().isoformat()
    save_state(sym, state)

print("\n==== LIVE ORDER / POSITION SUMMARY ====")
if all_messages:
    for msg in all_messages:
        print(msg)
else:
    print("No new live orders or stop updates logged today.")

ib.disconnect()
