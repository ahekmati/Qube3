from binance.client import Client

# Replace these with your Binance API credentials
api_key = 'tbk4AJ9Lk6uNSFIUJuEOgd8e2UXB2r1IfId0OQi9GlG1hPCM1nfDNRdjWsej9psB'
api_secret = 'eHVdI6rlsIn33MCwXH33jX8G0Xp27a8cMESVZm5oslpSCnB4tw5E6ympRpbFjxeg'

client = Client(api_key, api_secret)

# Pull account information (balances, etc.)
account_info = client.get_account()
print("Account Info:")
for balance in account_info['balances']:
    asset = balance['asset']
    free = float(balance['free'])
    locked = float(balance['locked'])
    if free > 0 or locked > 0:
        print(f"{asset}: Free={free}, Locked={locked}")

# Choose a symbol, e.g., BTCUSDT for recent trades
symbol = 'BTCUSDT'
trades = client.get_my_trades(symbol=symbol, limit=10)
print("\nRecent Trades for", symbol)
for trade in trades:
    print(f"Trade ID: {trade['id']}, Price: {trade['price']}, Qty: {trade['qty']}, Time: {trade['time']}, Side: {'Buy' if trade['isBuyer'] else 'Sell'}")

