import pandas as pd
import numpy as np
import yfinance as yf
import ta

# --- Download and Prepare Data ---
df = yf.download(
    'QQQ',
    start='2024-01-01',
    end='2025-10-27',
    interval='4h',
    auto_adjust=False
)
df.dropna(inplace=True)
df.reset_index(inplace=True)

if isinstance(df.columns, pd.MultiIndex):
    df.columns = [col[0] for col in df.columns]
df.rename(columns=lambda x: x.title(), inplace=True)
df['Atr14'] = ta.volatility.AverageTrueRange(df['High'], df['Low'], df['Close'], window=14).average_true_range()

# --- Parameters (edit as needed) ---
z_long_thresh = -0.5
z_short_thresh = 0.7
z_window = 10
ema_fast = 28
ema_slow = 70
take_profit = 0.07
max_hold = 20
stop_loss_type = 'percent'
stop_loss_val = 0.07

def calc_drawdown(equity_curve):
    roll_max = equity_curve.cummax()
    drawdown = equity_curve - roll_max
    return drawdown.min()

# --- Backtest Logic: Only one position at a time ---
equity = []
trades = []
position = None

for bar in range(z_window + 1, len(df)-1):
    row = df.iloc[bar]
    ema_fast_val = df['Close'].ewm(span=ema_fast, adjust=False).mean().iloc[bar]
    ema_slow_val = df['Close'].ewm(span=ema_slow, adjust=False).mean().iloc[bar]
    zscore = (df['Close'].iloc[bar] - ema_fast_val) / df['Close'].rolling(z_window).std().iloc[bar]
    close = df['Close'].iloc[bar]
    dt = df['Datetime'].iloc[bar]
    atr = df['Atr14'].iloc[bar]

    bullish = df['Close'].iloc[bar] > df['Open'].iloc[bar]
    bearish = df['Close'].iloc[bar] < df['Open'].iloc[bar]

    # --- Only enter when no open position ---
    if position is None:
        # Long entry
        if zscore < z_long_thresh and ema_fast_val > ema_slow_val and bullish:
            position = {'type': 'long', 'entry_bar': bar, 'entry_time': dt, 'entry_price': close, 'stop': None}
            if stop_loss_type == 'percent':
                position['stop'] = close * (1 - stop_loss_val)
            elif stop_loss_type == 'atr':
                position['stop'] = close - stop_loss_val * atr
        # Short entry
        elif zscore > z_short_thresh and ema_fast_val < ema_slow_val and bearish:
            position = {'type': 'short', 'entry_bar': bar, 'entry_time': dt, 'entry_price': close, 'stop': None}
            if stop_loss_type == 'percent':
                position['stop'] = close * (1 + stop_loss_val)
            elif stop_loss_type == 'atr':
                position['stop'] = close + stop_loss_val * atr
    # --- Manage open position ---
    if position:
        is_closed = False
        for hold in range(1, max_hold+1):
            idx = position['entry_bar'] + hold
            if idx >= len(df):
                break
            cur_close = df['Close'].iloc[idx]
            cur_time = df['Datetime'].iloc[idx]
            if position['type'] == 'long':
                # Take profit
                if cur_close >= position['entry_price'] * (1 + take_profit):
                    pnl = cur_close - position['entry_price']
                    trades.append({'side':'long','entry_time':position['entry_time'],'exit_time':cur_time,'entry':position['entry_price'],'exit':cur_close,'pnl':pnl,'reason':'TP'})
                    equity.append(pnl)
                    position = None
                    is_closed = True
                    break
                # Stop loss
                if stop_loss_type != 'none' and cur_close <= position['stop']:
                    pnl = cur_close - position['entry_price']
                    trades.append({'side':'long','entry_time':position['entry_time'],'exit_time':cur_time,'entry':position['entry_price'],'exit':cur_close,'pnl':pnl,'reason':'SL'})
                    equity.append(pnl)
                    position = None
                    is_closed = True
                    break
                cur_ema_fast = df['Close'].ewm(span=ema_fast, adjust=False).mean().iloc[idx]
                cur_ema_slow = df['Close'].ewm(span=ema_slow, adjust=False).mean().iloc[idx]
                if cur_ema_fast < cur_ema_slow:
                    pnl = cur_close - position['entry_price']
                    trades.append({'side':'long','entry_time':position['entry_time'],'exit_time':cur_time,'entry':position['entry_price'],'exit':cur_close,'pnl':pnl,'reason':'EMA'})
                    equity.append(pnl)
                    position = None
                    is_closed = True
                    break
            elif position['type'] == 'short':
                # Take profit
                if cur_close <= position['entry_price'] * (1 - take_profit):
                    pnl = position['entry_price'] - cur_close
                    trades.append({'side':'short','entry_time':position['entry_time'],'exit_time':cur_time,'entry':position['entry_price'],'exit':cur_close,'pnl':pnl,'reason':'TP'})
                    equity.append(pnl)
                    position = None
                    is_closed = True
                    break
                # Stop loss
                if stop_loss_type != 'none' and cur_close >= position['stop']:
                    pnl = position['entry_price'] - cur_close
                    trades.append({'side':'short','entry_time':position['entry_time'],'exit_time':cur_time,'entry':position['entry_price'],'exit':cur_close,'pnl':pnl,'reason':'SL'})
                    equity.append(pnl)
                    position = None
                    is_closed = True
                    break
                cur_ema_fast = df['Close'].ewm(span=ema_fast, adjust=False).mean().iloc[idx]
                cur_ema_slow = df['Close'].ewm(span=ema_slow, adjust=False).mean().iloc[idx]
                if cur_ema_fast > cur_ema_slow:
                    pnl = position['entry_price'] - cur_close
                    trades.append({'side':'short','entry_time':position['entry_time'],'exit_time':cur_time,'entry':position['entry_price'],'exit':cur_close,'pnl':pnl,'reason':'EMA'})
                    equity.append(pnl)
                    position = None
                    is_closed = True
                    break
        if not is_closed and position:
            cur_close = df['Close'].iloc[min(position['entry_bar'] + max_hold, len(df)-1)]
            cur_time = df['Datetime'].iloc[min(position['entry_bar'] + max_hold, len(df)-1)]
            pnl = (cur_close - position['entry_price']) if position['type']=='long' else (position['entry_price'] - cur_close)
            trades.append({'side':position['type'],'entry_time':position['entry_time'],'exit_time':cur_time,'entry':position['entry_price'],'exit':cur_close,'pnl':pnl,'reason':'TIME'})
            equity.append(pnl)
            position = None

# --- Results ---
equity_curve = pd.Series(np.cumsum(equity))
win_rate = np.mean([t['pnl'] > 0 for t in trades]) if trades else 0
max_dd = calc_drawdown(equity_curve) if len(equity_curve) else 0
print(f"Total PnL: {equity_curve.iloc[-1] if len(equity_curve) else 0:.2f}, Win Rate: {win_rate:.2%}, Num Trades: {len(trades)}, Max DD: {max_dd:.2f}")
print("\nSample Trades:")
for t in trades[:10]:
    print(f"{t['side'].upper()} Entry: {t['entry_time']} @ {t['entry']:.2f} | Exit: {t['exit_time']} @ {t['exit']:.2f} | PnL: {t['pnl']:.2f} | Reason: {t['reason']}")

