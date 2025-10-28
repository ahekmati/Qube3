import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX
import numpy as np

# Download QQQ daily closes for the last 2 years
qqq = yf.download('QQQ', period='2y', auto_adjust=True)['Close'].dropna()

# Take log of prices to stabilize variance
log_prices = np.log(qqq)

# Fit SARIMAX/ARIMA(1,1,1) on log prices
model = SARIMAX(log_prices, order=(1,1,1))
result = model.fit(disp=False)
print(result.summary())

# Forecast next day's log-close price
forecast = result.forecast(steps=1)
tomorrow_log = forecast.values[0]
tomorrow_close = np.exp(tomorrow_log)

# Today's price for comparison, force float for safe format
today_log = float(log_prices.iloc[-1])
today_close = float(qqq.iloc[-1])

print(f"Today's close: {today_close:.2f}")
print(f"Tomorrow's forecasted close: {tomorrow_close:.2f}")

if tomorrow_close > today_close:
    print("Suggested SIGNAL: BUY (model expects price rise)")
elif tomorrow_close < today_close:
    print("Suggested SIGNAL: SELL/SHORT (model expects price fall)")
else:
    print("Suggested SIGNAL: HOLD (no projected change)")

# Plot the last 30 closes and forecast
plt.figure(figsize=(12,4))
plt.plot(qqq[-30:], label='Recent Close Prices', marker='o')
plt.scatter([qqq.index[-1]], [today_close], label='Today', color='blue', zorder=5)
plt.scatter([qqq.index[-1] + pd.Timedelta(days=1)], [tomorrow_close], label='Forecast', color='red', zorder=5)
plt.legend()
plt.title('QQQ Close Prices & Next-Day ARIMA Forecast')
plt.show()
