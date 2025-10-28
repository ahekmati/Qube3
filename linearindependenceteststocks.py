import yfinance as yf
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

tickers = ["ARKK","IBIT","NUGT","QTUM","SOXL", "SSO", "SVXY" ,"TSLL","NVDL","TQQQ"]
names = {
    "ARKK":"ARK Innovation ETF",
    "IBIT":"iShares Blockchain and Tech ETF",
    "NUGT":"Direxion Daily Gold Miners Bull 3X Shares",
    "QTUM":"QTUM Token",
    "SOXL":"Direxion Daily Semiconductor Bull 3X Shares",
    "SSO":"ProShares Ultra S&P500",
    "SVXY":"ProShares Short VIX Short-Term Futures ETF",
    "TSLL":"TSL Limited",
    "NVDL":"NVIDIA Corporation",
    "TQQQ":"ProShares UltraPro QQQ"
}

data = yf.download(tickers, period='1y', auto_adjust=True)['Close'].dropna()
returns = data.pct_change().dropna()

argt = returns['TQQQ'].values
results = []

for ticker in tickers:
    if ticker == 'TQQQ' or ticker not in returns:
        continue
    stock = returns[ticker].values

    if len(stock) == 0 or len(argt) == 0:
        print(f"Skipping {ticker}: no valid return data.")
        continue

    # Beta and alpha via linear regression (stock = alpha + beta * ARGT)
    X = argt.reshape(-1, 1)
    y = stock
    model = LinearRegression().fit(X, y)
    beta = model.coef_[0]
    alpha = model.intercept_

    # Pearson correlation
    correlation = np.corrcoef(stock, argt)[0, 1]

    # Scalar projection
    scalar_projection = np.dot(stock, argt) / np.linalg.norm(argt)

    results.append({
        'name': names.get(ticker, ticker),
        'beta': beta,
        'alpha': alpha,
        'correlation': correlation,
        'scalar_proj': scalar_projection
    })

df = pd.DataFrame(results)
if not df.empty:
    print(df.set_index('name').round(4))
else:
    print("No valid results to display. All eligible tickers were missing data.")

"""
- This script checks for valid data and only displays results if at least one ticker is analyzed.
- Otherwise, it prints a clear message.
"""
