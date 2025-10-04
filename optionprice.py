from ib_insync import *
import pandas as pd
try:
    from colorama import Fore, Style, init as colorama_init
    colorama_init(autoreset=True)
    COLORAMA = True
except ImportError:
    COLORAMA = False

ib = IB()
ib.connect('127.0.0.1', 4001, 101)

symbol = 'AAPL'
target_strike = 170
target_expiry = '20251017'
right = 'C'

stock = Stock(symbol, 'SMART', 'USD')
ib.qualifyContracts(stock)
chains = ib.reqSecDefOptParams(symbol, '', 'STK', stock.conId)

# Only query the first valid exchange that matches
result = None
for chain in chains:
    if target_expiry in chain.expirations and target_strike in chain.strikes:
        contract = Option(symbol, target_expiry, target_strike, right, 'SMART')
        ib.qualifyContracts(contract)
        ticker = ib.reqMktData(contract)
        ib.sleep(1)
        bid = ticker.bid if ticker.bid not in (None, -1) else "N/A"
        ask = ticker.ask if ticker.ask not in (None, -1) else "N/A"
        last = ticker.last if ticker.last not in (None, -1) and not pd.isna(ticker.last) else "N/A"
        result = {
            "exchange": chain.exchange,
            "symbol": symbol,
            "expiry": target_expiry,
            "strike": target_strike,
            "right": right,
            "bid": bid,
            "ask": ask,
            "last": last
        }
        ib.cancelMktData(ticker)
        break  # Only the first match, stop searching exchanges

ib.disconnect()

print("\n" + "="*74)
print("Exchange   | Symbol   | Expiry     | Strike | Right |   Bid   |   Ask   |  Last  ")
print("="*74)
if result:
    line = f"{result['exchange']:<10} | {result['symbol']:<8} | {result['expiry']} | {result['strike']:<6} | {result['right']:<5} | {str(result['bid']):<7} | {str(result['ask']):<7} | {str(result['last']):<7}"
    if COLORAMA and (result['bid'] == "N/A" or result['ask'] == "N/A"):
        print(Fore.LIGHTBLACK_EX + line + Style.RESET_ALL)
    elif COLORAMA and (result['bid'] != "N/A" or result['ask'] != "N/A"):
        print(Fore.GREEN + line + Style.RESET_ALL)
    else:
        print(line)
else:
    print("No matching option found for parameters.")
print("="*74)
