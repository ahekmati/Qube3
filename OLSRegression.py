import yfinance as yf
import pandas as pd
import statsmodels.api as sm

# Download daily VXX prices and QQQ prices
aapl = yf.download('VXX', period='3y', auto_adjust=True)['Close']
rates = yf.download('QQQ', period='3y')['Close'] / 100

# Compute daily returns and daily yield changes
aapl_ret = aapl.pct_change()
rate_chg = rates.diff()

# Create named DataFrame after ensuring non-null, aligned data
data = pd.concat([aapl_ret, rate_chg], axis=1)
data.columns = ['aapl_ret', 'rate_chg']
data = data.dropna()

print("Columns:", data.columns)
print(data.head())

# Now regression will work as expected
X = sm.add_constant(data['rate_chg'])
y = data['aapl_ret']
model = sm.OLS(y, X)
results = model.fit()
print(results.summary())



#>OLS Regression Results between AAPL returns and QQQ yield changes