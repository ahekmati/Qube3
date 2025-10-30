import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA

# -----------------------------
# Parameters
# -----------------------------
ticker = "QQQ"
period = "6mo"
interval = "1d"
fast_ema = 7
slow_ema = 12
sma_period = 40

# -----------------------------
# Download Data
# -----------------------------
df = yf.download(ticker, period=period, interval=interval)
print("[*] Downloaded Data")
print("Columns in DataFrame:", df.columns)

# Flatten MultiIndex if needed
if isinstance(df.columns, pd.MultiIndex):
    df.columns = ['_'.join(col).strip() for col in df.columns.values]

# Ensure column names are lowercase
df.columns = [c.lower() for c in df.columns]

# Check for 'close' column
if 'close' not in df.columns:
    raise KeyError("No 'close' column found in the data.")

# -----------------------------
# Compute returns
# -----------------------------
df['returns'] = df['close'].pct_change()

# Drop NA for PCA
returns_dropna = df[['returns']].dropna()

# -----------------------------
# PCA (example, only 1 feature here)
# -----------------------------
pca = PCA(n_components=1)
pca.fit(returns_dropna)
print("PCA Explained Variance Ratio:", pca.explained_variance_ratio_)

# -----------------------------
# Moving averages
# -----------------------------
df['ema7'] = df['close'].ewm(span=fast_ema, adjust=False).mean()
df['ema12'] = df['close'].ewm(span=slow_ema, adjust=False).mean()
df['sma40'] = df['close'].rolling(sma_period).mean()

# -----------------------------
# Generate signals
# -----------------------------
df['signal'] = 0

for i in range(1, len(df)):
    # 12 EMA < 40 SMA
    condition1 = df.loc[df.index[i], 'ema12'] < df.loc[df.index[i], 'sma40']
    # 7 EMA crosses above 12 EMA
    condition2 = (df.loc[df.index[i-1], 'ema7'] <= df.loc[df.index[i-1], 'ema12']) and \
                 (df.loc[df.index[i], 'ema7'] > df.loc[df.index[i], 'ema12'])
    
    if condition1 and condition2:
        df.loc[df.index[i], 'signal'] = 1  # Buy
        print(f"{df.index[i]} BUY @ Close={df.loc[df.index[i], 'close']}")
    # Optional: add sell condition
    elif df.loc[df.index[i], 'ema7'] < df.loc[df.index[i], 'ema12']:
        df.loc[df.index[i], 'signal'] = -1  # Sell
        print(f"{df.index[i]} SELL @ Close={df.loc[df.index[i], 'close']}")

print("Done.")
