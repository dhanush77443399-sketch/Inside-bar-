"""Inside-bar strategy: backtest + live scan, with Telegram alerts."""
import os, sys
import pandas as pd
import requests
import yfinance as yf

# ---------------- CONFIG ----------------
MIN_MOVE_PCT = 1.5      # mother candle (close-open)/open >= 1.5%
MIN_BODY_RATIO = 0.80   # body >= 80% of mother candle's full range
RR = 2.0                # 1:2 risk-reward
MAX_HOLD_DAYS = 30      # force-exit after N days (None = hold until SL/TP)
BACKTEST_PERIOD = "5y"
# ----------------------------------------


def load(symbol, period="5y"):
    df = yf.download(symbol, period=period, interval="1d",
                     auto_adjust=False, progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df[["Open", "High", "Low", "Close"]].dropna()


def mother_ok(c, side):
    body = abs(c.Close - c.Open)
    rng = c.High - c.Low
    if rng <= 0:
        return False
    color_ok = c.Close > c.Open if side == "BUY" else c.Close < c.Open
    return (color_ok
            and body / c.Open * 100 >= MIN_MOVE_PCT
            and body / rng >= MIN_BODY_RATIO)


def signal_at(df, i):
    """Check if candle i is a signal candle (mother = i-2, inside = i-1)."""
    if i < 2:
        return None
    m, ib, s = df.iloc[i - 2], df.iloc[i - 1], df.iloc[i]
    inside = ib.High <= m.High and ib.Low >= m.Low
    if not inside:
        return None
    if mother_ok(m, "BUY") and s.Close > s.Open and s.Close > max(m.High, ib.High):
        entry, sl = s.Close, m.Low
        return dict(side="BUY", entry=entry, sl=sl, tp=entry + RR * (entry - sl))
    if mother_ok(m, "SELL") and s.Close < s.Open and s.Close < min(m.Low, ib.Low):
        entry, sl = s.Close, m.High
        return dict(side="SELL", entry=entry, sl=sl, tp=entry - RR * (sl - entry))
    return None


def simulate(df, i, sig):
    """Walk forward from day after signal. SL checked first if both hit same day."""
    risk = abs(sig["entry"] - sig["sl"])
    if risk <= 0:
        return None
    last = len(df) - 1
    end = last if MAX_HOLD_DAYS is None else min(last, i + MAX_HOLD_DAYS)
    for j in range(i + 1, end + 1):
        d = df.iloc[j]
        if sig["side"] == "BUY":
            if d.Low <= sig["sl"]: return j, -1.0, "SL"
            if d.High >= sig["tp"]: return j, RR, "TP"
        else:
            if d.High >= sig["sl"]: return j, -1.0, "SL"
            if d.Low <= sig["tp"]: return j, RR, "TP"
    if end == last and (MAX_HOLD_DAYS is None or last < i + MAX_HOLD_DAYS):
        return None  # still open
    exit_px = df.iloc[end].Close
    r = ((exit_px - sig["entry"]) if sig["side"] == "BUY"
         else (sig["entry"] - exit_px)) / risk
    return end, r, "TIME"


def backtest(symbol):
    df = load(symbol, BACKTEST_PERIOD)
    trades, i = [], 2
    while i < len(df):
        sig = signal_at(df, i)
        if sig:
            res = simulate(df, i, sig)
            if res:
                j, r, why = res
                trades.append(dict(symbol=symbol, date=df.index[i].date(),
                                   side=sig["side"], entry=round(sig["entry"], 2),
                                   sl=round(sig["sl"], 2), tp=round(sig["tp"], 2),
                                   exit_date=df.index[j].date(), result=why, R=round(r, 2)))
                i = j  # one trade at a time per symbol
        i += 1
    return trades


def summary(t):
    if t.empty:
        return "No trades found."
    wins = (t.R > 0).sum()
    eq = t.R.cumsum()
    dd = (eq.cummax() - eq).max()
    return (f"Trades: {len(t)}\nWin rate: {wins/len(t)*100:.1f}%\n"
            f"Total R: {t.R.sum():.1f}\nAvg R/trade: {t.R.mean():.2f}\n"
            f"Max drawdown: {dd:.1f}R\nBuy: {(t.side=='BUY').sum()} | Sell: {(t.side=='SELL').sum()}")


def telegram(msg):
    tok, chat = os.getenv("TELEGRAM_TOKEN"), os.getenv("TELEGRAM_CHAT_ID")
    if not tok or not chat:
        print("[Telegram not configured]\n" + msg); return
    for k in range(0, len(msg), 4000):
        requests.post(f"https://api.telegram.org/bot{tok}/sendMessage",
                      data={"chat_id": chat, "text": msg[k:k+4000]}, timeout=30)


def symbols():
    with open("symbols.txt") as f:
        return [x.strip() for x in f if x.strip() and not x.startswith("#")]


def run_backtest():
    all_t = []
    for s in symbols():
        try: all_t += backtest(s)
        except Exception as e: print(s, "failed:", e)
    t = pd.DataFrame(all_t)
    t.to_csv("trades.csv", index=False)
    telegram(f"📊 Inside Bar Backtest ({BACKTEST_PERIOD}, 1D, RR 1:{RR:g})\n\n" + summary(t))


def run_scan():
    out = []
    for s in symbols():
        try:
            df = load(s, "3mo")
            sig = signal_at(df, len(df) - 1)
            if sig:
                out.append(f"{'🟢' if sig['side']=='BUY' else '🔴'} {sig['side']} {s}\n"
                           f"Entry {sig['entry']:.2f} | SL {sig['sl']:.2f} | TP {sig['tp']:.2f}")
        except Exception as e: print(s, "failed:", e)
    if out:
        telegram("🔔 Inside Bar signals (daily close)\n\n" + "\n\n".join(out))
    else:
        print("No signals today")


if __name__ == "__main__":
    {"backtest": run_backtest, "scan": run_scan}[sys.argv[1] if len(sys.argv) > 1 else "backtest"]()
