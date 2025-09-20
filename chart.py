from ibapi.client import EClient
from ibapi.wrapper import EWrapper
from ibapi.contract import Contract
import threading
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time

# IB API setup
class IBapi(EWrapper, EClient):
    def __init__(self):
        EClient.__init__(self, self)
        self.data = []

    def historicalData(self, reqId, bar):
        self.data.append([bar.date, bar.close])
    def historicalDataEnd(self, reqId, start, end):
        self.disconnect()

def run_loop(app):
    app.run()

def smma(series, length):
    smma = [np.nan] * len(series)
    if len(series) < length:
        return smma
    smma[length-1] = np.mean(series[:length])
    for i in range(length, len(series)):
        smma[i] = (smma[i-1] * (length - 1) + series[i]) / length
    return smma

app = IBapi()
app.connect('127.0.0.1', 4001, 123)
api_thread = threading.Thread(target=run_loop, args=(app,), daemon=True)
api_thread.start()
time.sleep(1)

contract = Contract()
contract.symbol = 'AAPL'  # Change as needed
contract.secType = 'STK'
contract.exchange = 'SMART'
contract.currency = 'USD'

# Request daily historical data (2 years as example)
app.reqHistoricalData(1, contract, '', '2 Y', '1 day', 'TRADES', 0, 2, False, [])
time.sleep(4)  # Increase if you need more time for data download

data = pd.DataFrame(app.data, columns=['date', 'close'])
data['close'] = pd.to_numeric(data['close'])

# Calculate SMMAs
data['smma_13'] = smma(data['close'], 13)
data['smma_31'] = smma(data['close'], 31)

# Plotting
plt.figure(figsize=(14,7))
plt.plot(data['date'], data['close'], label='Daily Close')
plt.plot(data['date'], data['smma_13'], label='13 SMMA')
plt.plot(data['date'], data['smma_31'], label='31 SMMA')
plt.title('Daily Chart with 13 & 31 SMMAs')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
