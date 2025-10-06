import pandas as pd
from datetime import datetime, timedelta, UTC
from ta.trend import EMAIndicator
from ta.volatility import AverageTrueRange
from coinbase.rest import RESTClient
import uuid
from decimal import Decimal, ROUND_FLOOR, getcontext

getcontext().prec = 28

client = RESTClient(key_file="coinbase_api_key.json")

PRODUCT_ID = "SOL-USDC"
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

def maybe_load_increments():
    global QUOTE_TICK, BASE_STEP
    try:
        p = client.get_product(product_id=PRODUCT_ID)
        qi = p.get("quote_increment") if isinstance(p, dict) else getattr(p, "quote_increment", None)
        bi = p.get("base_increment") if isinstance(p, dict) else getattr(p, "base_increment", None)
        if qi:
            QUOTE_TICK = Decimal(str(qi))
        if bi:
            BASE_STEP = Decimal(str(bi))
    except Exception:
        pass

# ========= BALANCES =========
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
            avail_val = avail.get("value") if isinstance(avail, dict) else (
                avail if isinstance(avail, (int, float, str)) else 0.0)
            hold = a.get("hold") if isinstance(a, dict) else getattr(a, "hold", None)
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
            avail_val = avail.get("value") if isinstance(avail, dict) else (
                avail if isinstance(avail, (int, float, str)) else 0.0)
            hold = a.get("hold") if isinstance(a, dict) else getattr(a, "hold", None)
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
            val = ab.get("value") if isinstance(ab, dict) else (ab if isinstance(ab, (int, float, str)) else 0.0)
            try:
                return float(val)
            except Exception:
                return 0.0
    return 0.0

def print_balances(product_id):
    base, quote = product_id.split("-")
    base_total = get_open_base_size(product_id)
    quote_avail = get_quote_available(product_id)
    print(f"[BAL] {base} (available+hold): {base_total}")
    print(f"[BAL] {quote} available: {quote_avail}")
    others = []
    for a in get_accounts_list():
        curr = a.get("currency") if isinstance(a, dict) else getattr(a, "currency", None)
        if curr in (base, quote):
            continue
        ab = a.get("available_balance") if isinstance(a, dict) else getattr(a, "available_balance", None)
        val = ab.get("value") if isinstance(ab, dict) else (ab if isinstance(ab, (int, float, str)) else None)
        hold = a.get("hold") if isinstance(a, dict) else getattr(a, "hold", None)
        hold_val = hold.get("value") if isinstance(hold, dict) else (hold if isinstance(hold, (int, float, str)) else 0.0)
        try:
            v = float(val) if val is not None else 0.0
        except Exception:
            v = 0.0
        try:
            h = float(hold_val) if hold_val is not None else 0.0
        except Exception:
            h = 0.0
        pos = v + h
        if pos > 0:
            others.append(f"{curr}:{pos}")
    if others:
        print(f"[BAL] Other nonzero: {', '.join(others)}")

# The rest of your script remains unchanged from your last provided version...
# [CANDLE FETCH], [INDICATOR], [SIGNAL], [ORDER] and all main trading logic remain unmodified.
# Paste your indicator, trading, and main loop code here as needed.

# ======== MAIN EXAMPLE ========
def main():
    maybe_load_increments()
    run_ts = datetime.now(UTC).isoformat().replace("+00:00", "Z")
    print(f"[RUN] 📅 {run_ts}")
    print_balances(PRODUCT_ID)
    # ...rest of your logic...

if __name__ == "__main__":
    main()
