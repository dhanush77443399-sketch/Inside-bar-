"""Inside-bar strategy: backtest + live scan, with Telegram alerts."""
import os, sys, io, time
import pandas as pd
import requests
import yfinance as yf

# ---------------- CONFIG ----------------
MIN_MOVE_PCT = 1.5      # mother candle (close-open)/open >= 1.5%
MIN_BODY_RATIO = 0.80   # body >= 80% of mother candle's full range
RR = 2.0                # 1:2 risk-reward
MAX_HOLD_DAYS = 30      # force-exit after N days (None = hold until SL/TP)
# Days to backtest: set from GitHub "Run workflow" box (DAYS env) or `python strategy.py backtest 90`
BACKTEST_DAYS = int(os.getenv("DAYS") or (sys.argv[2] if len(sys.argv) > 2 else 60))
BACKTEST_PERIOD = "2y" if BACKTEST_DAYS <= 500 else ("5y" if BACKTEST_DAYS <= 1700 else "10y")
ENTRY_TIME = "15:30 IST"   # daily candle closes at NSE close, entry is taken at that close
MAX_TRADES_IN_TELEGRAM = 60  # most recent N trades listed in chat (full list in trades.csv)
UNIVERSE = "NIFTY200"   # "NIFTY200" = fetch official list, anything else = symbols.txt
NIFTY200_URLS = [
    "https://nsearchives.nseindia.com/content/indices/ind_nifty200list.csv",
    "https://www.niftyindices.com/IndexConstituent/ind_nifty200list.csv",
]
# ----------------------------------------


def load(symbol, period="5y"):
    df = pd.DataFrame()
    for _ in range(3):  # retry on flaky Yahoo responses
        df = yf.download(symbol, period=period, interval="1d",
                         auto_adjust=False, progress=False)
        if not df.empty:
            break
        time.sleep(2)
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
        return last, 0.0, "OPEN"  # still open, excluded from stats
    exit_px = df.iloc[end].Close
    r = ((exit_px - sig["entry"]) if sig["side"] == "BUY"
         else (sig["entry"] - exit_px)) / risk
    return end, r, "TIME"


def backtest(symbol):
    df = load(symbol, BACKTEST_PERIOD)
    trades, i = [], 2
    cutoff = df.index[-1] - pd.Timedelta(days=BACKTEST_DAYS)
    while i < len(df):
        if df.index[i] < cutoff:
            i += 1
            continue
        sig = signal_at(df, i)
        if sig:
            res = simulate(df, i, sig)
            if res:
                j, r, why = res
                trades.append(dict(symbol=symbol.replace(".NS", ""), date=df.index[i].date(),
                                   time=ENTRY_TIME, side=sig["side"], entry=round(sig["entry"], 2),
                                   sl=round(sig["sl"], 2), tp=round(sig["tp"], 2),
                                   exit_date=df.index[j].date(), result=why, R=round(r, 2)))
                i = j  # one trade at a time per symbol
        i += 1
    return trades


def summary(t):
    if t.empty:
        return "No trades found."
    n_open = (t.result == "OPEN").sum()
    t = t[t.result != "OPEN"]
    if t.empty:
        return f"No closed trades yet. Open trades: {n_open}"
    wins = (t.R > 0).sum()
    eq = t.R.cumsum()
    dd = (eq.cummax() - eq).max()
    return (f"Trades: {len(t)}\nWin rate: {wins/len(t)*100:.1f}%\n"
            f"Total R: {t.R.sum():.1f}\nAvg R/trade: {t.R.mean():.2f}\n"
            f"Max drawdown: {dd:.1f}R\nStill open: {n_open}\n"
            + "\n".join(f"{k}: {len(g)} trades, win {((g.R>0).mean()*100):.0f}%, {g.R.sum():.1f}R"
                        for k, g in t.groupby("side")))


def telegram(msg):
    tok, chat = os.getenv("TELEGRAM_TOKEN"), os.getenv("TELEGRAM_CHAT_ID")
    if not tok or not chat:
        print("[Telegram not configured]\n" + msg); return
    chunks, cur = [], ""
    for line in msg.split("\n"):
        if len(cur) + len(line) + 1 > 3800:
            chunks.append(cur); cur = ""
        cur += line + "\n"
    chunks.append(cur)
    for c in chunks:
        requests.post(f"https://api.telegram.org/bot{tok}/sendMessage",
                      data={"chat_id": chat, "text": c}, timeout=30)


def nifty200():
    hdr = {"User-Agent": "Mozilla/5.0"}
    for url in NIFTY200_URLS:
        try:
            r = requests.get(url, headers=hdr, timeout=30)
            r.raise_for_status()
            syms = pd.read_csv(io.StringIO(r.text))["Symbol"].str.strip().tolist()
            if len(syms) >= 150:
                return [x + ".NS" for x in syms]
        except Exception as e:
            print("Nifty200 fetch failed:", url, e)
    return None


def symbols():
    if UNIVERSE == "NIFTY200":
        got = nifty200()
        if got:
            print(f"Loaded {len(got)} Nifty 200 symbols")
            return got
        print("Falling back to symbols.txt")
    with open("symbols.txt") as f:
        return [x.strip() for x in f if x.strip() and not x.startswith("#")]


def trade_lines(t):
    icon = {"TP": "✅ TP", "SL": "❌ SL", "TIME": "⏱ TIME", "OPEN": "🔵 OPEN"}
    t = t.sort_values("date", ascending=False).head(MAX_TRADES_IN_TELEGRAM)
    out = []
    for r in t.itertuples():
        out.append(f"{'🟢' if r.side=='BUY' else '🔴'} {r.symbol} | {r.side}\n"
                   f"📅 {r.date} {r.time}\n"
                   f"Entry {r.entry} | SL {r.sl} | TP {r.tp}\n"
                   f"{icon[r.result]} ({r.exit_date}) {r.R:+.1f}R\n")
    return out


def run_backtest():
    all_t, syms = [], symbols()
    for n, s in enumerate(syms, 1):
        try: all_t += backtest(s)
        except Exception as e: print(s, "failed:", e)
        if n % 25 == 0: print(f"{n}/{len(syms)} done")
    t = pd.DataFrame(all_t)
    t.to_csv("trades.csv", index=False)
    telegram(f"📊 Inside Bar Backtest - {len(syms)} stocks (last {BACKTEST_DAYS} days, 1D, RR 1:{RR:g})\n\n" + summary(t))
    if not t.empty:
        telegram(f"📋 Trades (latest {min(len(t), MAX_TRADES_IN_TELEGRAM)} of {len(t)})\n\n"
                 + "\n".join(trade_lines(t)))


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
