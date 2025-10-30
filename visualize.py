import pandas as pd
import numpy as np
import yfinance as yf
import ta  # pip install ta

# --- Download and Prepare Data ---
df = yf.download(
    'QQQ',
    start='2024-01-01',
    end='2025-10-01',
    interval='4h',
    auto_adjust=False
)
df.dropna(inplace=True)
df.reset_index(inplace=True)

# --- FLATTEN MultiIndex columns if present ---
if isinstance(df.columns, pd.MultiIndex):
    df.columns = [col[0] for col in df.columns]

print("Flat columns:", df.columns)
print("Sample data:\n", df.head())

# --- Standardize column names ---
df.rename(columns=lambda x: x.title(), inplace=True)
# Columns now: ['Datetime', 'Adj Close', 'Close', 'High', 'Low', 'Open', 'Volume']

# --- ATR for stop loss ---
df['Atr14'] = ta.volatility.AverageTrueRange(df['High'], df['Low'], df['Close'], window=14).average_true_range()

# --- Parameter Grid ---
z_long_range = [-0.5]
z_short_range = [0.7]
z_window = 10
ema_fast_range = [28]
ema_slow_range = [70]
ma_fast_confirm_range = [0, 9]  # 0 disables, 9 enables MA crossover
take_profit_range = [0.07]
max_hold_range = [20]
stop_loss_type_range = ['percent', 'atr', 'none']
stop_loss_val_range = [0.07, 1.5]

results = []

def calc_drawdown(equity_curve):
    roll_max = equity_curve.cummax()
    drawdown = equity_curve - roll_max
    return drawdown.min()

# --- Main Backtest Loop ---
for use_confirm in [False, True]:
    for stop_loss_type in stop_loss_type_range:
        for stop_loss_val in stop_loss_val_range:
            equity = []
            trades = []
            position = None

            for bar in range(z_window + 1, len(df)-1):
                row = df.iloc[bar]
                ema_fast = df['Close'].ewm(span=28, adjust=False).mean().iloc[bar]
                ema_slow = df['Close'].ewm(span=70, adjust=False).mean().iloc[bar]
                zscore = (df['Close'].iloc[bar] - ema_fast) / df['Close'].rolling(z_window).std().iloc[bar]
                close = df['Close'].iloc[bar]
                dt = df['Datetime'].iloc[bar]
                atr = df['Atr14'].iloc[bar]

                bullish = df['Close'].iloc[bar] > df['Open'].iloc[bar]
                bearish = df['Close'].iloc[bar] < df['Open'].iloc[bar]

                # Fast MA confirmation
                ma_confirm = True
                if use_confirm and ma_fast_confirm_range[1] > 0:
                    fast_confirm = df['Close'].ewm(span=ma_fast_confirm_range[1], adjust=False).mean().iloc[bar]
                    if zscore < 0 and close < fast_confirm:
                        ma_confirm = False
                    elif zscore > 0 and close > fast_confirm:
                        ma_confirm = False

                # --- Long entry ---
                if not position and zscore < z_long_range[0] and ema_fast > ema_slow:
                    if not use_confirm or (bullish and ma_confirm):
                        position = {'type': 'long', 'entry_bar': bar, 'entry_time': dt, 'entry_price': close,
                                    'stop': None}
                        if stop_loss_type == 'percent':
                            position['stop'] = close * (1 - stop_loss_val)
                        elif stop_loss_type == 'atr':
                            position['stop'] = close - stop_loss_val * atr

                # --- Short entry ---
                elif not position and zscore > z_short_range[0] and ema_fast < ema_slow:
                    if not use_confirm or (bearish and ma_confirm):
                        position = {'type': 'short', 'entry_bar': bar, 'entry_time': dt, 'entry_price': close,
                                    'stop': None}
                        if stop_loss_type == 'percent':
                            position['stop'] = close * (1 + stop_loss_val)
                        elif stop_loss_type == 'atr':
                            position['stop'] = close + stop_loss_val * atr

                # --- Position management ---
                if position:
                    exit_reason = None
                    for hold in range(1, max_hold_range[0]+1):
                        idx = position['entry_bar'] + hold
                        if idx >= len(df):
                            break
                        cur_close = df['Close'].iloc[idx]
                        cur_time = df['Datetime'].iloc[idx]

                        if position['type'] == 'long':
                            if cur_close >= position['entry_price'] * (1 + take_profit_range[0]):
                                pnl = cur_close - position['entry_price']
                                exit_reason = 'TP'
                                trades.append({'side':'long','entry_time':position['entry_time'],'exit_time':cur_time,'entry':position['entry_price'],'exit':cur_close,'pnl':pnl,'reason':'TP'})
                                equity.append(pnl)
                                position = None
                                break
                            if stop_loss_type != 'none' and cur_close <= position['stop']:
                                pnl = cur_close - position['entry_price']
                                exit_reason = 'SL'
                                trades.append({'side':'long','entry_time':position['entry_time'],'exit_time':cur_time,'entry':position['entry_price'],'exit':cur_close,'pnl':pnl,'reason':'SL'})
                                equity.append(pnl)
                                position = None
                                break
                            cur_ema_fast = df['Close'].ewm(span=28, adjust=False).mean().iloc[idx]
                            cur_ema_slow = df['Close'].ewm(span=70, adjust=False).mean().iloc[idx]
                            if cur_ema_fast < cur_ema_slow:
                                pnl = cur_close - position['entry_price']
                                exit_reason = 'X'
                                trades.append({'side':'long','entry_time':position['entry_time'],'exit_time':cur_time,'entry':position['entry_price'],'exit':cur_close,'pnl':pnl,'reason':'EMA'})
                                equity.append(pnl)
                                position = None
                                break
                        elif position['type'] == 'short':
                            if cur_close <= position['entry_price'] * (1 - take_profit_range[0]):
                                pnl = position['entry_price'] - cur_close
                                exit_reason = 'TP'
                                trades.append({'side':'short','entry_time':position['entry_time'],'exit_time':cur_time,'entry':position['entry_price'],'exit':cur_close,'pnl':pnl,'reason':'TP'})
                                equity.append(pnl)
                                position = None
                                break
                            if stop_loss_type != 'none' and cur_close >= position['stop']:
                                pnl = position['entry_price'] - cur_close
                                exit_reason = 'SL'
                                trades.append({'side':'short','entry_time':position['entry_time'],'exit_time':cur_time,'entry':position['entry_price'],'exit':cur_close,'pnl':pnl,'reason':'SL'})
                                equity.append(pnl)
                                position = None
                                break
                            cur_ema_fast = df['Close'].ewm(span=28, adjust=False).mean().iloc[idx]
                            cur_ema_slow = df['Close'].ewm(span=70, adjust=False).mean().iloc[idx]
                            if cur_ema_fast > cur_ema_slow:
                                pnl = position['entry_price'] - cur_close
                                exit_reason = 'X'
                                trades.append({'side':'short','entry_time':position['entry_time'],'exit_time':cur_time,'entry':position['entry_price'],'exit':cur_close,'pnl':pnl,'reason':'EMA'})
                                equity.append(pnl)
                                position = None
                                break
                    if position:
                        cur_close = df['Close'].iloc[min(position['entry_bar'] + max_hold_range[0], len(df)-1)]
                        cur_time = df['Datetime'].iloc[min(position['entry_bar'] + max_hold_range[0], len(df)-1)]
                        pnl = (cur_close - position['entry_price']) if position['type']=='long' else (position['entry_price'] - cur_close)
                        trades.append({'side':position['type'],'entry_time':position['entry_time'],'exit_time':cur_time,'entry':position['entry_price'],'exit':cur_close,'pnl':pnl,'reason':'TIME'})
                        equity.append(pnl)
                        position = None

            equity_curve = pd.Series(np.cumsum(equity))
            max_dd = calc_drawdown(equity_curve) if len(equity_curve) else 0
            win_rate = np.mean([t['pnl'] > 0 for t in trades]) if trades else 0
            results.append({'use_confirm':use_confirm,
                            'stop_loss_type':stop_loss_type,
                            'stop_loss_value':stop_loss_val,
                            'PnL':equity_curve.iloc[-1] if len(equity_curve) else 0,
                            'win_rate':win_rate,
                            'num_trades':len(trades),
                            'max_dd':max_dd,
                            'trades':trades})

results_df = pd.DataFrame(results)
print("Top Settings by PnL:")
print(results_df.sort_values('PnL', ascending=False)[['use_confirm', 'stop_loss_type', 'stop_loss_value', 'PnL', 'win_rate', 'num_trades', 'max_dd']].head(8))

print("\nSample Trades for Best Result:")
best = results_df.sort_values('PnL', ascending=False).iloc[0]
for t in best['trades'][:10]:
    print(f"{t['side'].upper()} Entry: {t['entry_time']} @ {t['entry']:.2f} | Exit: {t['exit_time']} @ {t['exit']:.2f} | PnL: {t['pnl']:.2f} | Reason: {t['reason']}")
