import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from lifelines import KaplanMeierFitter
from fpdf import FPDF
from ib_insync import *

def smma(series, period):
    return series.ewm(alpha=1/period, adjust=False).mean()

def wma(series, window):
    weights = np.arange(1, window+1)
    return series.rolling(window).apply(
        lambda prices: np.dot(prices, weights)/weights.sum(), raw=True)

def streaks(bool_series):
    streak_lengths = []
    count = 0
    for val in bool_series:
        if val:
            count += 1
        else:
            if count > 0:
                streak_lengths.append(count)
                count = 0
    if count > 0:
        streak_lengths.append(count)
    return streak_lengths

def mfe_mae(df, regime):
    mfe, mae = [], []
    current_run = []
    for idx in range(len(df)):
        if regime.iloc[idx]:
            current_run.append(df['close'].iloc[idx])
        else:
            if current_run:
                entry = current_run[0]
                mfe.append(max(current_run)-entry)
                mae.append(min(current_run)-entry)
                current_run = []
    if current_run:
        entry = current_run[0]
        mfe.append(max(current_run)-entry)
        mae.append(min(current_run)-entry)
    return mfe, mae

def atr_trailing_exit(df, regime, atr_n=14, multiple=2):
    df['ATR'] = df['close'].rolling(atr_n).std()
    streak_days = []
    in_trade = False
    high = None
    for idx in range(1, len(df)):
        if regime.iloc[idx] and not regime.iloc[idx-1]:
            in_trade = True
            entry_idx = idx
            high = df['close'].iloc[idx]
        elif in_trade:
            high = max(high, df['close'].iloc[idx])
            stop = high - multiple * df['ATR'].iloc[idx]
            if df['close'].iloc[idx] < stop:
                streak_days.append(idx - entry_idx)
                in_trade = False
    return streak_days

def print_stats(label, streaks_list, mfe, mae, atr_streaks):
    print(f"\n--- {label} regime ---")
    print(f"Total days: {sum(streaks_list)}")
    print(f"Mean streak: {np.mean(streaks_list):.2f}, Median streak: {np.median(streaks_list):.2f}")
    print(f"75th percentile: {np.percentile(streaks_list, 75):.2f}, Max streak: {np.max(streaks_list) if streaks_list else 0}")
    kmf = KaplanMeierFitter()
    if streaks_list:
        kmf.fit(streaks_list)
        print("Survival probabilities (chance streak > t days):")
        for t in [10, 20, 30, 50, 75, 100]:
            if t <= max(streaks_list):
                print(f"Day {t}: survival probability = {kmf.survival_function_at_times(t).values[0]:.2f}")
    print(f"\nMaximum Favorable Excursion (mean/75th): {np.mean(mfe):.2f}/{np.percentile(mfe, 75):.2f}")
    print(f"Maximum Adverse Excursion (mean/10th): {np.mean(mae):.2f}/{np.percentile(mae, 10):.2f}")
    print(f"ATR trailing stop exit days (mean/median): {np.mean(atr_streaks):.2f}/{np.median(atr_streaks):.2f}")

class PDFReport(FPDF):
    def header(self):
        self.set_font('Arial', 'B', 16)
        self.cell(0, 12, f'{self.ticker} 9SMMA/58WMA Analytics Report', ln=1, align='C')
        self.set_font('Arial', '', 12)
        self.cell(0, 10, "", ln=1)
    def section_title(self, txt):
        self.ln(5)
        self.set_font('Arial', 'B', 14)
        self.cell(0, 10, txt, ln=1)
    def section_body(self, txt):
        self.set_font('Arial', '', 11)
        self.multi_cell(0, 8, txt)
        self.ln(2)

def main_analysis(df, ticker):
    df['SMMA9'] = smma(df['close'], 9)
    df['WMA58'] = wma(df['close'], 58)
    above = df['SMMA9'] > df['WMA58']
    below = df['SMMA9'] < df['WMA58']

    regimes = {'Bullish (9SMMA>58WMA)': above, 'Bearish (9SMMA<58WMA)': below}
    pdf = PDFReport()
    pdf.ticker = ticker
    pdf.add_page()
    for label, regime in regimes.items():
        streaks_list = streaks(regime)
        mfe, mae = mfe_mae(df, regime)
        atr_streaks = atr_trailing_exit(df, regime)
        print_stats(label, streaks_list, mfe, mae, atr_streaks)
        summary = (
            f"Total days: {sum(streaks_list)}\n"
            f"Mean streak: {np.mean(streaks_list):.2f}, Median: {np.median(streaks_list):.2f}\n"
            f"75th percentile: {np.percentile(streaks_list, 75):.2f}, Max: {np.max(streaks_list) if streaks_list else 0}\n"
            f"MFE (mean/75th): {np.mean(mfe):.2f}/{np.percentile(mfe, 75):.2f}\n"
            f"MAE (mean/10th): {np.mean(mae):.2f}/{np.percentile(mae, 10):.2f}\n"
            f"ATR stop exit (mean/median): {np.mean(atr_streaks):.2f}/{np.median(atr_streaks):.2f}\n"
        )
        pdf.section_title(label + " Regime Statistics")
        pdf.section_body(summary)
        plt.figure(figsize=(4.5,2.5))
        sns.histplot(streaks_list, bins=20)
        plt.title(f"{label} Streak Lengths")
        plt.xlabel('Consecutive Days')
        plt.ylabel('Frequency')
        plt.tight_layout()
        plt.savefig('hist.png')
        pdf.image('hist.png', w=160)
        plt.close()
        kmf = KaplanMeierFitter()
        if streaks_list:
            kmf.fit(streaks_list)
            plt.figure(figsize=(4.5,2.5))
            kmf.plot_survival_function()
            plt.title(f"{label} Streak Survival Probability")
            plt.xlabel('Days')
            plt.ylabel('Probability')
            plt.tight_layout()
            plt.savefig('surv.png')
            pdf.image('surv.png', w=160)
            plt.close()
        plt.figure(figsize=(4.5,2.5))
        sns.histplot(mfe, bins=20, color='green', label='MFE', alpha=0.7)
        sns.histplot(mae, bins=20, color='red', label='MAE', alpha=0.7)
        plt.title(f"{label} MFE (up) & MAE (down)")
        plt.xlabel('Max Move from Entry')
        plt.ylabel('Frequency')
        plt.legend()
        plt.tight_layout()
        plt.savefig('mfe_mae.png')
        pdf.image('mfe_mae.png', w=160)
        plt.close()
    pdf.output(f"{ticker}_ma_regime_report.pdf")
    print(f"\nPDF report saved as: {ticker}_ma_regime_report.pdf")

def valid_duration(durationStr):
    units = {"S", "D", "W", "M", "Y"}
    # Split and check format like '4 Y'
    parts = durationStr.strip().upper().split()
    return len(parts) == 2 and parts[0].isdigit() and parts[1] in units

def valid_bar_size(barSizeSetting):
    allowed = {"day", "hour", "min"}
    lval = barSizeSetting.strip().lower()
    return any(x in lval for x in allowed)

def get_ibkr_data():
    ticker = input("Enter the ticker to study: ").upper()
    secType = input("Security type (STK or FUT): ").upper()
    exchange = input("Exchange [default SMART]: ").upper() or "SMART"
    currency = input("Currency [default USD]: ").upper() or "USD"
    expiry = ""
    if secType == "FUT":
        expiry = input("Future expiry (yyyymm): ")
        contract = Future(ticker, expiry, exchange, currency)
    elif secType == "STK":
        contract = Stock(ticker, exchange, currency)
    else:
        print("Invalid security type.")
        return None, ticker

    durationStr = input("History duration (example '4 Y' for 4 years) [default '4 Y']: ").strip()
    if not valid_duration(durationStr):
        print("Using default duration: 4 Y (for four years)")
        durationStr = "4 Y"

    barSizeSetting = input("Bar size (example '1 day') [default '1 day']: ").strip()
    if not valid_bar_size(barSizeSetting):
        print("Using default bar size: 1 day")
        barSizeSetting = "1 day"

    ib = IB()
    ib.connect('127.0.0.1', 4001, clientId=1)
    try:
        bars = ib.reqHistoricalData(
            contract,
            endDateTime='',
            durationStr=durationStr,
            barSizeSetting=barSizeSetting,
            whatToShow='TRADES',
            useRTH=True
        )
        df = util.df(bars) if bars else None
    except Exception as e:
        print(f"IBKR Error: {e}")
        df = None
    ib.disconnect()
    if df is None or df.empty:
        print(f"No data returned for {ticker}")
        return None, ticker
    return df, ticker

if __name__ == "__main__":
    df, ticker = get_ibkr_data()
    if df is not None and not df.empty:
        main_analysis(df, ticker)
