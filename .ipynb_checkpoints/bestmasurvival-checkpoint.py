import numpy as np
import pandas as pd
from ta.trend import SMAIndicator, EMAIndicator, WMAIndicator
from ib_insync import *
import itertools

def smma(series, period):
    return series.ewm(alpha=1/period, adjust=False).mean()

def get_ma(series, period, ma_type):
    if ma_type == 'SMA':
        return SMAIndicator(series, window=period).sma_indicator()
    elif ma_type == 'EMA':
        return EMAIndicator(series, window=period).ema_indicator()
    elif ma_type == 'WMA':
        return WMAIndicator(series, window=period).wma()
    elif ma_type == 'SMMA':
        return smma(series, period)
    else:
        raise ValueError('Unknown MA type')

def get_trade_results(df, fast_type, slow_type, fast_period, slow_period, take_profit=None):
    df = df.copy()
    df['fast'] = get_ma(df['close'], fast_period, fast_type)
    df['slow'] = get_ma(df['close'], slow_period, slow_type)
    signals = (df['fast'] > df['slow']).astype(int)
    signals = signals.diff().fillna(0)
    trade_results = []
    capital = 1.0
    position = 0
    entry = 0
    for i in range(1, len(signals)):
        if signals.iloc[i] == 1 and position == 0:
            position = 1
            entry = df['close'].iloc[i]
        elif signals.iloc[i] == -1 and position == 1:
            exit_price = df['close'].iloc[i]
            gain = exit_price / entry
            trade_results.append(gain)
            capital *= gain
            position = 0
        # Take-profit logic: close early if desired gain hit
        if position == 1 and take_profit is not None:
            if df['close'].iloc[i] / entry >= 1 + take_profit:
                gain = df['close'].iloc[i] / entry
                trade_results.append(gain)
                capital *= gain
                position = 0
    # Close last open trade if any
    if position == 1:
        gain = df['close'].iloc[-1] / entry
        trade_results.append(gain)
        capital *= gain
    return capital, trade_results

def streak_survival_probability(trade_results):
    # Survival probability as product of win rate and longest win streak (example metric)
    wins = [r > 1.0 for r in trade_results]
    win_count = sum(wins)
    if len(trade_results) == 0:
        return 0.0
    win_prob = win_count / len(trade_results)
    # Find longest win streak
    max_streak, streak = 0, 0
    for w in wins:
        if w:
            streak += 1
            if streak > max_streak:
                max_streak = streak
        else:
            streak = 0
    # Use product as a ranking metric (you can adjust)
    survival_score = win_prob * max_streak
    return survival_score

def run_for_timeframe(ib, contract, durationStr, barSizeSetting, timeframe_name):
    bars = ib.reqHistoricalData(contract, endDateTime='', durationStr=durationStr,
                                barSizeSetting=barSizeSetting, whatToShow='TRADES', useRTH=True)
    df = util.df(bars)
    if df.empty:
        print(f"No data found for {contract.symbol} in timeframe: {timeframe_name}.")
        return None

    ma_types = ['SMA', 'EMA', 'SMMA', 'WMA']
    results = []
    streak_stats = []

    best_result = -np.inf
    best_combo = None

    for fast_type, slow_type in itertools.product(ma_types, repeat=2):
        for fast_period in range(5, 21, 2):
            for slow_period in range(18, 61, 5):
                if fast_period >= slow_period:
                    continue
                capital, trade_results = get_trade_results(df, fast_type, slow_type, fast_period, slow_period)
                streak_score = streak_survival_probability(trade_results)
                streak_stats.append((fast_type, fast_period, slow_type, slow_period, capital, streak_score))
                results.append((fast_type, fast_period, slow_type, slow_period, capital))
                if capital > best_result:
                    best_result = capital
                    best_combo = (fast_type, fast_period, slow_type, slow_period)

    print(f"\nBest MA combo for {contract.symbol} ({timeframe_name}): {best_combo} with final capital: {best_result:.2f}x")
    print(f"\nTop 5 combinations for {contract.symbol} ({timeframe_name}):")
    sorted_results = sorted(results, key=lambda x: -x[4])[:5]
    for row in sorted_results:
        print(f"Fast {row[0]}({row[1]}), Slow {row[2]}({row[3]}): {row[4]:.2f}x")

    # Top 2 by streak survival “probability”
    top2_streak = sorted(streak_stats, key=lambda x: -x[5])[:2]
    print(f"\nTop 2 MA combos by streak survival probability for {contract.symbol} ({timeframe_name}):")
    for combo in top2_streak:
        print(f"Fast {combo[0]}({combo[1]}), Slow {combo[2]}({combo[3]}) | Capital: {combo[4]:.2f}x | StreakSurvivalScore: {combo[5]:.3f}")

    # Test take-profit for those combos (ensure periods are integers)
    TAKE_PROFIT = 0.06  # 6% take-profit example
    print(f"\nCapital if taking 6% profit per trade (take-profit) for top 2 streak-survival combos:")
    for combo in top2_streak:
        capital_tp, _ = get_trade_results(
            df,
            combo[0],                  # fast_type
            combo[2],                  # slow_type
            int(combo[1]),             # fast_period, ensure int
            int(combo[3]),             # slow_period, ensure int
            take_profit=TAKE_PROFIT
        )
        print(f"Fast {combo[0]}({combo[1]}), Slow {combo[2]}({combo[3]}): {capital_tp:.2f}x")

def main():
    ticker = input("Enter the stock ticker you want to find the best MA for: ").upper()
    ib = IB()
    ib.connect('127.0.0.1', 4001, clientId=111)
    contract = Stock(ticker, 'SMART', 'USD')

    timeframes = [
        ('4 Y', '1 day', 'Daily'),
        ('2 Y', '8 hours', '8-Hour'),
        ('1 Y', '4 hours', '4-Hour'),
        ('180 D', '1 hour', '1-Hour')
    ]

    for durationStr, barSizeSetting, tf_name in timeframes:
        run_for_timeframe(ib, contract, durationStr, barSizeSetting, tf_name)

    ib.disconnect()

if __name__ == "__main__":
    main()
