import pandas as pd
from datetime import datetime, timedelta, UTC
from ta.trend import EMAIndicator
from ta.volatility import AverageTrueRange
from coinbase.rest import RESTClient
import uuid
from decimal import Decimal, ROUND_FLOOR, getcontext

getcontext().prec = 28
client = RESTClient(key_file="coinbase_api_key.json")

PRODUCT_IDS = ["SOL-USDC", "ETH-USDC"]
GRANULARITY = "ONE_DAY"
CANDLE_COUNT = 240
STOP_OFFSET = 8.0
FIXED_RISK_USD = 100.0
ATR_WINDOW = 14
ATR_MULT = 2.0
STOP_ORDER_TAG = "ema_stop"
QUOTE_TICK = Decimal("0.01")
BASE_STEP  = Decimal("0.0001")
ONE_DAYSEC = 24 * 3600

def client_id(prefix):
    return f"{prefix}_{uuid.uuid4().hex[:12]}"

def align_to_day_epoch(ts_utc: datetime) -> int:
    d = ts_utc.date()
    aligned = datetime(d.year, d.month, d.day, tzinfo=UTC)
    return int(aligned.timestamp())

def last_completed_day_end() -> int:
    now_utc = datetime.now(UTC) - timedelta(minutes=1)
    return align_to_day_epoch(now_utc)

def fmt_price(x: Decimal) -> str:
    q = QUOTE_TICK.normalize()
    decimals = -q.as_tuple().exponent if q.as_tuple().exponent < 0 else 0
    return f"{x:.{decimals}f}"

def fmt_size(x: Decimal) -> str:
    s = BASE_STEP.normalize()
    decimals = -s.as_tuple().exponent if s.as_tuple().exponent < 0 else 0
    return f"{x:.{decimals}f}"

def round_price_to_tick(x: Decimal) -> Decimal:
    return (x / QUOTE_TICK).to_integral_value() * QUOTE_TICK

def floor_size_to_step(x: Decimal) -> Decimal:
    steps = (x / BASE_STEP).to_integral_value(rounding=ROUND_FLOOR)
    return steps * BASE_STEP

def maybe_load_increments(product_id):
    global QUOTE_TICK, BASE_STEP
    try:
        p = client.get_product(product_id=product_id)
        qi = p.get("quote_increment") if isinstance(p, dict) else getattr(p, "quote_increment", None)
        bi = p.get("base_increment") if isinstance(p, dict) else getattr(p, "base_increment", None)
        if qi:
            QUOTE_TICK = Decimal(str(qi))
        if bi:
            BASE_STEP = Decimal(str(bi))
    except Exception:
        pass

def get_accounts_list():
    accts = client.get_accounts()
    accounts_list = getattr(accts, "accounts", None)
    if accounts_list is None and isinstance(accts, dict):
        accounts_list = accts.get("accounts", [])
    return accounts_list or []

def get_open_base_size(product_id):
    base, _ = product_id.split("-")
    for a in get_accounts_list():
        curr = a.get("currency") if isinstance(a, dict) else getattr(a, "currency", None)
        if curr == base:
            avail = a.get("available_balance") if isinstance(a, dict) else getattr(a, "available_balance", None)
            hold = a.get("hold") if isinstance(a, dict) else getattr(a, "hold", None)
            avail_val = avail.get("value") if isinstance(avail, dict) else (
                avail if isinstance(avail, (int, float, str)) else 0.0)
            hold_val = hold.get("value") if isinstance(hold, dict) else (
                hold if isinstance(hold, (int, float, str)) else 0.0)
            try:
                avail_f = float(avail_val)
            except Exception:
                avail_f = 0.0
            try:
                hold_f = float(hold_val)
            except Exception:
                hold_f = 0.0
            return avail_f + hold_f
    return 0.0
def get_available_base_precise(product_id) -> Decimal:
    base, _ = product_id.split("-")
    for a in get_accounts_list():
        curr = a.get("currency") if isinstance(a, dict) else getattr(a, "currency", None)
        if curr == base:
            avail = a.get("available_balance") if isinstance(a, dict) else getattr(a, "available_balance", None)
            hold = a.get("hold") if isinstance(a, dict) else getattr(a, "hold", None)
            avail_val = avail.get("value") if isinstance(avail, dict) else (
                avail if isinstance(avail, (int, float, str)) else 0.0)
            hold_val = hold.get("value") if isinstance(hold, dict) else (
                hold if isinstance(hold, (int, float, str)) else 0.0)
            try:
                avail_d = Decimal(str(avail_val))
            except Exception:
                avail_d = Decimal("0")
            try:
                hold_d = Decimal(str(hold_val))
            except Exception:
                hold_d = Decimal("0")
            return avail_d + hold_d
    return Decimal("0")

def get_quote_available(product_id):
    _, quote = product_id.split("-")
    for a in get_accounts_list():
        curr = a.get("currency") if isinstance(a, dict) else getattr(a, "currency", None)
        if curr == quote:
            ab = a.get("available_balance") if isinstance(a, dict) else getattr(a, "available_balance", None)
            hold = a.get("hold") if isinstance(a, dict) else getattr(a, "hold", None)
            val = ab.get("value") if isinstance(ab, dict) else (ab if isinstance(ab, (int, float, str)) else 0.0)
            hold_val = hold.get("value") if isinstance(hold, dict) else (hold if isinstance(hold, (int, float, str)) else 0.0)
            try:
                return float(val) + float(hold_val)
            except Exception:
                return 0.0
    return 0.0

def print_balances(product_id):
    base, quote = product_id.split("-")
    base_avail = get_open_base_size(product_id)
    quote_avail = get_quote_available(product_id)
    print(f"[BAL] {product_id}: {base} (available+hold): {base_avail}, {quote} (available+hold): {quote_avail}")

def fetch_ohlcv(product_id, granularity, limit):
    end_unix = last_completed_day_end()
    resp = client.get_candles(
        product_id=product_id,
        start=str(end_unix - (limit - 1) * ONE_DAYSEC),
        end=str(end_unix),
        granularity=granularity
    )
    rows = None
    if isinstance(resp, dict) and "candles" in resp:
        rows = resp["candles"]
    if rows is None and hasattr(resp, "candles"):
        rows = getattr(resp, "candles")
    if rows is None and isinstance(resp, list):
        rows = resp
    if rows is None and hasattr(resp, "to_dict"):
        d = resp.to_dict()
        rows = d.get("candles")
    if rows is None and hasattr(resp, "dict"):
        d = resp.dict()
        rows = d.get("candles")
    if rows is None or not isinstance(rows, list) or len(rows) == 0:
        raise RuntimeError("No candles returned")
    def to_int_s(v):
        try:
            return int(v)
        except Exception:
            try:
                return int(float(v))
            except Exception:
                return None
    def to_float_s(v):
        try:
            return float(v)
        except Exception:
            return None
    def get_field(r, k):
        if isinstance(r, dict):
            return r.get(k)
        return getattr(r, k, None)
    clean = []
    for r in rows:
        start_raw = get_field(r, "start"); low_raw=get_field(r,"low"); high_raw=get_field(r,"high")
        open_raw = get_field(r,"open"); close_raw = get_field(r,"close"); vol_raw = get_field(r,"volume")
        if all(x is None for x in (start_raw, low_raw, high_raw, open_raw, close_raw, vol_raw)):
            try:
                start_raw=r[0]; low_raw=r[1]; high_raw=r[2]; open_raw=r[3]; close_raw=r[4]; vol_raw=r[5]
            except Exception:
                continue
        start_val = to_int_s(start_raw)
        if start_val is None:
            continue
        clean.append({
            "start": start_val,
            "open":  to_float_s(open_raw),
            "high":  to_float_s(high_raw),
            "low":   to_float_s(low_raw),
            "close": to_float_s(close_raw),
            "volume": to_float_s(vol_raw),
        })
    if not clean:
        raise RuntimeError("Empty normalized candles")
    df = pd.DataFrame(clean)
    df["timestamp"] = pd.to_datetime(df["start"].astype(int), unit="s", utc=True)
    df = df.sort_values("timestamp").reset_index(drop=True)
    for c in ["open","high","low","close"]:
        df[c] = df[c].astype(float)
    return df

def compute_indicators(df):
    df["ema9"]  = EMAIndicator(df["close"], window=9).ema_indicator()
    df["ema18"] = EMAIndicator(df["close"], window=18).ema_indicator()
    df["atr"]   = AverageTrueRange(high=df["high"], low=df["low"], close=df["close"], window=ATR_WINDOW).average_true_range()
    return df

def primary_cross_signal(df):
    prev = df.iloc[-2]; last = df.iloc[-1]
    return (last["ema9"] > last["ema18"]) and (prev["ema9"] <= prev["ema18"])

def reentry_signal(df):
    last = df.iloc[-1]
    if not (last["ema9"] > last["ema18"]):
        return False
    lookback = min(30, len(df) - 2)
    if lookback <= 0:
        return False
    recent = df.tail(lookback + 1)
    below = recent["close"] < recent["ema18"]
    dips = below.iloc[:-1][below.iloc[:-1]].index
    if len(dips) == 0:
        return False
    return last["close"] > last["ema18"]

def get_last_price(product_id):
    bb = client.get_best_bid_ask(product_ids=[product_id])
    pricebooks = bb.get("pricebooks", []) if isinstance(bb, dict) else getattr(bb, "pricebooks", [])
    if pricebooks:
        pb0 = pricebooks[0]
        bids = pb0.get("bids", []) if isinstance(pb0, dict) else getattr(pb0, "bids", [])
        asks = pb0.get("asks", []) if isinstance(pb0, dict) else getattr(pb0, "asks", [])
        if bids and asks:
            bid = Decimal(str(bids[0]["price"] if isinstance(bids[0], dict) else bids[0].price))
            ask = Decimal(str(asks[0]["price"] if isinstance(asks[0], dict) else asks[0].price))
            return float((bid + ask) / 2)
    product = client.get_product(product_id=product_id)
    p = None
    if isinstance(product, dict) and "price" in product:
        try:
            p = float(product["price"])
        except Exception:
            p = None
    else:
        pa = getattr(product, "price", None)
        if pa is not None:
            try:
                p = float(pa)
            except Exception:
                p = None
    if p is not None:
        return p
    trades = client.get_market_trades(product_id=product_id, limit=1)
    tl = trades.get("trades") if isinstance(trades, dict) else getattr(trades, "trades", [])
    if tl:
        pr = tl[0].get("price") if isinstance(tl[0], dict) else getattr(tl[0], "price", None)
        if pr is not None:
            return float(pr)
    raise RuntimeError("Cannot fetch price")
def get_position_avg_entry(product_id, max_fills=100):
    fills_list = []
    try:
        fills = client.get_fills(product_id=product_id, limit=max_fills)
        fills_list = fills.get("fills") if isinstance(fills, dict) else getattr(fills, "fills", [])
    except AttributeError:
        try:
            fills = client.fills(product_id=product_id, limit=max_fills)
            fills_list = fills.get("fills") if isinstance(fills, dict) else getattr(fills, "fills", [])
        except Exception as e:
            print(f"[WARN] fills error: {e}")
            return None
    except Exception as e:
        print(f"[WARN] get_fills error: {e}")
        return None
    if not fills_list:
        return None
    total_qty = Decimal("0"); total_cost = Decimal("0")
    for f in fills_list:
        side = f.get("side") if isinstance(f, dict) else getattr(f, "side", "")
        settled = f.get("settled") if isinstance(f, dict) else getattr(f, "settled", True)
        size = f.get("size") if isinstance(f, dict) else getattr(f, "size", None)
        price = f.get("price") if isinstance(f, dict) else getattr(f, "price", None)
        try:
            if str(side).upper() == "BUY" and size and price and (settled is True or str(settled).lower() == "true"):
                qty = Decimal(str(size)); px = Decimal(str(price))
                total_qty += qty; total_cost += qty * px
        except Exception:
            continue
    if total_qty <= 0:
        return None
    return float(total_cost / total_qty)

def place_market_buy_with_bracket(product_id, quote_size, tp_limit_price, sl_trigger_price):
    cid = client_id("buy")
    print(f"[LEVELS] TP(limit): {tp_limit_price} | SL(trigger): {sl_trigger_price}")
    resp = client.create_order(
        client_order_id=cid,
        product_id=product_id,
        side="BUY",
        order_configuration={
            "market_market_ioc": {
                "quote_size": str(round(quote_size, 2)),
                "reduce_only": False
            }
        },
        attached_order_configuration={
            "trigger_bracket_gtc": {
                "limit_price": str(round(tp_limit_price, 4)),
                "stop_trigger_price": str(round(sl_trigger_price, 4))
            }
        }
    )
    print(f"[TRADE] 🟢 Market buy with TP/SL bracket placed: {resp}")
    return resp

def list_open_orders(product_id):
    try:
        o = client.list_orders(product_id=product_id, order_status="OPEN")
        return o.get("orders") if isinstance(o, dict) else getattr(o, "orders", [])
    except Exception as e:
        print(f"[WARN] list_orders OPEN error: {e}")
        return []

def has_open_protective_order(product_id):
    for o in list_open_orders(product_id):
        side = o.get("side") if isinstance(o, dict) else getattr(o, "side", "")
        if side and side.upper() != "SELL":
            continue
        cfg = o.get("order_configuration") if isinstance(o, dict) else getattr(o, "order_configuration", {})
        if isinstance(cfg, dict):
            if "stop_limit_stop_limit_gtc" in cfg or "stop_limit_stop_limit_gtd" in cfg:
                return True
            if "trigger_bracket_gtc" in cfg:
                return True
        t = o.get("order_type") if isinstance(o, dict) else getattr(o, "order_type", "")
        if t and ("STOP" in str(t).upper() or "BRACKET" in str(t).upper()):
            return True
    return False

def has_open_take_profit(product_id, target_price_dec: Decimal, ticks_tolerance: int = 2) -> bool:
    tol = QUOTE_TICK * ticks_tolerance
    lo = target_price_dec - tol
    hi = target_price_dec + tol
    for o in list_open_orders(product_id):
        side = o.get("side") if isinstance(o, dict) else getattr(o, "side", "")
        if not side or side.upper() != "SELL":
            continue
        cfg = o.get("order_configuration") if isinstance(o, dict) else getattr(o, "order_configuration", {})
        limit_price = None
        if isinstance(cfg, dict):
            for k in ("limit_limit_gtc", "limit_limit_gtd"):
                if k in cfg:
                    lp = cfg[k].get("limit_price")
                    if lp:
                        try:
                            limit_price = Decimal(str(lp))
                        except Exception:
                            pass
        if limit_price is None:
            lp = o.get("price") if isinstance(o, dict) else getattr(o, "price", None)
            if lp:
                try:
                    limit_price = Decimal(str(lp))
                except Exception:
                    limit_price = None
        if limit_price is not None and lo <= limit_price <= hi:
            return True
    return False

def place_stop_limit_sell(product_id, base_size_dec: Decimal, stop_price_dec: Decimal, limit_price_dec: Decimal or None = None, tag=STOP_ORDER_TAG):
    stop_p = round_price_to_tick(stop_price_dec)
    limit_p = round_price_to_tick(stop_p - QUOTE_TICK) if limit_price_dec is None else round_price_to_tick(limit_price_dec)
    size_p = floor_size_to_step(base_size_dec)
    cid = client_id(tag)
    order_cfg = {
        "stop_limit_stop_limit_gtc": {
            "base_size": fmt_size(size_p),
            "limit_price": fmt_price(limit_p),
            "stop_price": fmt_price(stop_p),
            "stop_direction": "STOP_DIRECTION_STOP_DOWN"
        }
    }
    resp = client.create_order(
        client_order_id=cid,
        product_id=product_id,
        side="SELL",
        order_configuration=order_cfg
    )
    print(f"[STOP] 📉 Stop-limit sell placed @ stop {fmt_price(stop_p)} / limit {fmt_price(limit_p)} for {fmt_size(size_p)}: {resp}")
    return resp

def place_take_profit_limit(product_id, base_size_dec: Decimal, tp_price_dec: Decimal, tag="tp_limit"):
    tp_p = round_price_to_tick(tp_price_dec)
    size_p = floor_size_to_step(base_size_dec)
    cid = client_id(tag)
    order_cfg = {
        "limit_limit_gtc": {
            "base_size": fmt_size(size_p),
            "limit_price": fmt_price(tp_p)
        }
    }
    resp = client.create_order(
        client_order_id=cid,
        product_id=product_id,
        side="SELL",
        order_configuration=order_cfg
    )
    print(f"[TP] 🎯 Take-profit limit placed @ {fmt_price(tp_p)} for {fmt_size(size_p)}: {resp}")
    return resp

def enter_trade(product_id, df):
    last_close = Decimal(str(df["close"].iloc[-1]))
    ema18 = Decimal(str(df["ema18"].iloc[-1]))
    atr_val = df["atr"].iloc[-1]
    if pd.isna(atr_val):
        print("[ERROR] ❌ ATR not available; need more history.")
        return
    atr = Decimal(str(atr_val))
    sl_trigger = ema18 - Decimal(str(STOP_OFFSET))
    tp_limit = last_close + Decimal(str(ATR_MULT)) * atr
    buy_quote = FIXED_RISK_USD
    quote_avail = get_quote_available(product_id)
    fee_buffer = buy_quote * 0.002
    if quote_avail < buy_quote + fee_buffer:
        print(f"[ERROR] ❌ Insufficient quote balance. Need ~{buy_quote+fee_buffer:.2f}, available {quote_avail:.2f}.")
        return
    print(f"[TRADE] 🟢 Buying ~{buy_quote:.2f} USDC of {product_id.split('-')[0]} with TP {float(tp_limit):.4f} and SL trigger {float(sl_trigger):.4f}")
    order = place_market_buy_with_bracket(
        product_id=product_id,
        quote_size=buy_quote,
        tp_limit_price=float(tp_limit),
        sl_trigger_price=float(sl_trigger)
    )
    ok = False
    if isinstance(order, dict) and order.get("success"):
        ok = True
    elif hasattr(order, "success") and getattr(order, "success"):
        ok = True
    if not ok:
        err = order.get("error_response") if isinstance(order, dict) else getattr(order, "error_response", None)
        print(f"[ERROR] ❌ Buy+Bracket failed: {err}")
        return

def ensure_protection_when_open(product_id, df):
    base_pos = get_open_base_size(product_id)
    avg_entry = get_position_avg_entry(product_id)
    last_close_f = df["close"].iloc[-1]
    atr_f = df["atr"].iloc[-1]
    if pd.isna(atr_f):
        print(f"[LEVELS] ATR not available to compute TP yet.")
        tp_price_dec = None
    else:
        tp_now = last_close_f + ATR_MULT * atr_f
        print(f"[LEVELS] Current ATR TP(calc): {tp_now:.4f} (close {last_close_f:.4f} + {ATR_MULT} × ATR {atr_f:.4f})")
        tp_price_dec = Decimal(str(tp_now))
    sl_now = df["ema18"].iloc[-1] - STOP_OFFSET
    print(f"[LEVELS] Current SL(trigger): {sl_now:.4f} (EMA18 {df['ema18'].iloc[-1]:.4f} − {STOP_OFFSET})")
    if avg_entry is not None:
        print(f"[POS] Current {product_id.split('-')[0]} position: {base_pos} @ avg entry {avg_entry:.6f} USDC")
    else:
        print(f"[POS] Current {product_id.split('-')[0]} position: {base_pos} @ avg entry unknown (no fills found)")
    if not has_open_protective_order(product_id):
        ema18 = Decimal(str(df["ema18"].iloc[-1]))
        sl_trigger = ema18 - Decimal(str(STOP_OFFSET))
        raw_avail = get_available_base_precise(product_id)
        safe_size = raw_avail - Decimal("0.0000001")
        if safe_size > 0:
            safe_size = floor_size_to_step(safe_size)
            if safe_size > 0:
                print(f"[LEVELS] Adding protective SL(trigger): {float(sl_trigger):.4f} for {fmt_size(safe_size)} (from raw {float(raw_avail)})")
                place_stop_limit_sell(product_id, base_size_dec=safe_size, stop_price_dec=sl_trigger, limit_price_dec=None, tag=f"{STOP_ORDER_TAG}")
        else:
            print("[INFO] Available base size too small; skipping stop placement.")
    else:
        print("[INFO] Protective stop/bracket already present; no additional stop placed.")
    if tp_price_dec is not None:
        if not has_open_take_profit(product_id, tp_price_dec):
            raw_avail = get_available_base_precise(product_id)
            safe_size = raw_avail - Decimal("0.0000001")
            safe_size = floor_size_to_step(safe_size)
            if safe_size > 0:
                print(f"[TP] Ensuring TP: placing limit at {float(tp_price_dec):.4f} for {fmt_size(safe_size)}")
                place_take_profit_limit(product_id, base_size_dec=safe_size, tp_price_dec=tp_price_dec)
            else:
                print("[TP] Skipped placing TP: size too small after flooring.")
        else:
            print("[TP] Take-profit already present near current ATR target; no new TP placed.")

def main():
    for product_id in PRODUCT_IDS:
        maybe_load_increments(product_id)
        print(f"[RUN] {product_id} at {datetime.now(UTC).isoformat().replace('+00:00', 'Z')}")
        print_balances(product_id)
        df = fetch_ohlcv(product_id, GRANULARITY, CANDLE_COUNT)
        df = compute_indicators(df)
        if not df["atr"].isna().iloc[-1]:
            last_close = df["close"].iloc[-1]
            atr_now = df["atr"].iloc[-1]
            tp_now = last_close + ATR_MULT * atr_now
            print(f"[LEVELS] {product_id} ATR TP(calc): {tp_now:.4f} (close {last_close:.4f} + {ATR_MULT} × ATR {atr_now:.4f})")
        sl_now = df["ema18"].iloc[-1] - STOP_OFFSET
        print(f"[LEVELS] {product_id} SL(trigger): {sl_now:.4f} (EMA18 {df['ema18'].iloc[-1]:.4f} − {STOP_OFFSET})")
        base_pos = get_open_base_size(product_id)
        crossed = primary_cross_signal(df)
        reenter = reentry_signal(df)
        if base_pos > 0.0:
            print(f"[INFO] 📌 {product_id} position is currently OPEN.")
            ensure_protection_when_open(product_id, df)
            continue
        if crossed:
            print(f"[SIGNAL] {product_id} Primary EMA 9/18 bullish crossover.")
            user_input = input(f"🚨 {product_id} 9/18 bullish crossover detected. Enter trade with ATR TP? (yes/no): ").strip().lower()
            if user_input == "yes":
                enter_trade(product_id, df)
            else:
                print(f"[INFO] ❌ {product_id} Trade skipped by user.")
            continue
        if reenter:
            print(f"[SIGNAL] {product_id} Re-entry signal met.")
            user_input = input(f"🚨 {product_id} Re-entry signal detected. Enter trade with ATR TP? (yes/no): ").strip().lower()
            if user_input == "yes":
                enter_trade(product_id, df)
            else:
                print(f"[INFO] ❌ {product_id} Re-entry skipped by user.")
            continue
        print(f"[INFO] No entry or re-entry signal for {product_id}.")

if __name__ == "__main__":
    main()
