from ib_insync import *
import math
from datetime import datetime
import numpy as np

ib = IB()
ib.connect('127.0.0.1', 4001, clientId=1)

ticker = input("Enter ticker symbol: ").strip().upper()
holding_days = int(input("How many days until expiry (e.g., 30)? "))

# Get underlying contract and last price
contract = Stock(ticker, 'SMART', 'USD')
ib.qualifyContracts(contract)
market_data = ib.reqMktData(contract)
ib.sleep(2)
underlying_price = None
if market_data.last and not math.isnan(market_data.last):
    underlying_price = float(market_data.last)
elif market_data.close and not math.isnan(market_data.close):
    underlying_price = float(market_data.close)
else:
    print(f"Cannot fetch a valid price for {ticker}. Exiting.")
    ib.disconnect()
    exit(1)

# Fetch available expiries from option param chain:
chains = ib.reqSecDefOptParams(ticker, '', 'STK', contract.conId)
chains = [c for c in chains if c.exchange.upper() == 'SMART']
if not chains:
    print(f"No option chain for {ticker}. Exiting.")
    ib.disconnect()
    exit(1)
chain = chains[0]
if not chain.expirations:
    print(f"No option expiries for {ticker} available.")
    ib.disconnect()
    exit(1)

today = datetime.now().date()
# Find nearest expiry by days
expiries = sorted(chain.expirations)
try:
    target_expiry = min(expiries, key=lambda x: abs((datetime.strptime(x, '%Y%m%d').date() - today).days - holding_days))
except Exception as e:
    print(f"Could not parse expiry: {e}")
    ib.disconnect()
    exit(1)
days_to_expiry = (datetime.strptime(target_expiry, '%Y%m%d').date() - today).days

# Retrieve all call contracts for this expiry
filters = Option(ticker, target_expiry, 0, 'C', 'SMART')
contracts_details = ib.reqContractDetails(filters)
real_contracts = [cd.contract for cd in contracts_details if cd.contract.lastTradeDateOrContractMonth == target_expiry and cd.contract.right == 'C']
if not real_contracts:
    print(f"No listed call options found for {ticker} on {target_expiry}")
    ib.disconnect()
    exit(1)

# For each contract, get Greeks and filter for 0.50 <= delta <= 0.70
filtered_options = []
print(f"\nScanning options for delta range 0.50 to 0.70 at expiry {target_expiry}...\n")
for c in real_contracts:
    ib.qualifyContracts(c)
    odata = ib.reqMktData(c, '', False, False, None)
    ib.sleep(2)
    # Greeks may be None if market is closed or no permission
    try:
        delta = odata.modelGreeks.delta if odata.modelGreeks else None
    except Exception:
        delta = None
    if delta is not None and 0.50 <= abs(delta) <= 0.70:
        filtered_options.append((c.strike, odata.bid, odata.ask, odata.impliedVolatility if odata.impliedVolatility else 0.0, delta))

if not filtered_options:
    print("No call options found with delta in the 0.50–0.70 range for this expiry. Try a different expiry or check during market hours.")
else:
    print(f"{'Strike':>8} {'Bid':>8} {'Ask':>8} {'IV':>8} {'Delta':>8}")
    for strike, bid, ask, iv, delta in sorted(filtered_options):
        print(f"{strike:8.2f} {bid:8} {ask:8} {iv:8.2%} {delta:8.2f}")

ib.disconnect()
