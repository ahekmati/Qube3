import pandas as pd

def smma(series, period):
    """Calculate Smoothed Moving Average (SMMA) using exponential weighting."""
    smma_series = series.ewm(alpha=1/period, adjust=False).mean()
    return smma_series

def average_smma_above_days(df):
    """Returns average period (days) where SMMA(9) > SMMA(18) in the DataFrame."""
    df['SMMA9'] = smma(df['close'], 12)
    df['SMMA18'] = smma(df['close'], 40)
    above = df['SMMA9'] > df['SMMA18']
    periods = []
    count = 0
    for val in above:
        if val:
            count += 1
        else:
            if count > 0:
                periods.append(count)
                count = 0
    if count > 0:
        periods.append(count)
    if periods:
        return sum(periods)/len(periods)
    return 0

# --- Pull data from IBKR and process ---
from ib_insync import *
ib = IB()
ib.connect('127.0.0.1', 4001, clientId=23)  # Port changed to 4001
contract = Stock('QQQ', 'SMART', 'USD')
bars = ib.reqHistoricalData(contract, endDateTime='', durationStr='4 Y',
                            barSizeSetting='1 day', whatToShow='TRADES', useRTH=True)
df = util.df(bars)

average_days = average_smma_above_days(df)
print(f'Average number of days SMMA(9) > SMMA(18): {average_days:.2f}')
