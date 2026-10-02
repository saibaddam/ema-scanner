#!/usr/bin/env python3
"""
S&P 500 Daily/Weekly EMA Touch Scanner  ->  ntfy phone alerts
Runs on GitHub Actions every 5 minutes. Each run does ONE job and exits:

  ~8:00 ET (first run of the day) : build today's EMA levels for all S&P 500 stocks
  9:31 - 4:00 PM ET                : check live prices, alert on new EMA touches
  after 4:10 PM ET                 : send the close report (REJECTED / BOUNCED), save CSV

State between runs lives in data/levels.json and data/state.json, which the
workflow commits back to the repo.

Live EMA formula (same as the Pine script):
    live_ema = prev_ema + 2/(len+1) * (price - prev_ema)

Local test:   NTFY_TOPIC=your-topic python scanner.py --test
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime, timedelta
from io import StringIO
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
import requests
import yfinance as yf

# ============================ SETTINGS ============================
EMA_LENGTHS = [9, 21, 50]
TIMEFRAMES = ["D", "W"]          # "D" = daily, "W" = weekly
TOUCH_PCT = 0.10                 # within 0.10% of the EMA counts as a touch
MIN_PRICE = 0                    # e.g. 100 to only scan stocks above $100
EXTRA_TICKERS = []               # e.g. ["PLTR", "SNDK"]
EXCLUDE_TICKERS = []
MAX_SINGLE_ALERTS = 5            # more new touches than this in one run -> one combined alert
# ==================================================================

NTFY_TOPIC = os.environ.get("NTFY_TOPIC", "").strip()
NTFY_SERVER = os.environ.get("NTFY_SERVER", "https://ntfy.sh").strip()

ET = ZoneInfo("America/New_York")
HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
LEVELS_FILE = DATA / "levels.json"
STATE_FILE = DATA / "state.json"
RESULTS_DIR = DATA / "results"
TF_NAME = {"D": "Daily", "W": "Weekly"}


# ---------------------------------------------------------------- helpers
def log(msg):
    print(f"[{datetime.now(ET):%Y-%m-%d %H:%M:%S} ET] {msg}", flush=True)


def notify(title, body, tags="chart_with_upwards_trend", priority="default", click=None):
    if not NTFY_TOPIC:
        log(f"(no NTFY_TOPIC set) {title}: {body}")
        return
    headers = {"Title": title, "Tags": tags, "Priority": priority}
    if click:
        headers["Click"] = click
    try:
        r = requests.post(f"{NTFY_SERVER}/{NTFY_TOPIC}", data=body[:4000].encode("utf-8"),
                          headers=headers, timeout=10)
        if r.status_code >= 300:
            log(f"ntfy error {r.status_code}: {r.text[:200]}")
    except Exception as e:
        log(f"ntfy send failed: {e}")


def tv_link(ticker):
    return f"https://www.tradingview.com/chart/?symbol={ticker.replace('-', '.')}"


def load_json(path, default):
    try:
        return json.loads(path.read_text())
    except Exception:
        return default


def save_json(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(obj, indent=1, sort_keys=True)
    if not path.exists() or path.read_text() != text:   # only touch the file if it changed
        path.write_text(text)


# ---------------------------------------------------------------- tickers
def get_sp500():
    cache = DATA / "sp500_tickers.csv"
    try:
        html = requests.get("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies",
                            headers={"User-Agent": "Mozilla/5.0"}, timeout=20).text
        table = pd.read_html(StringIO(html))[0]
        syms = sorted(table["Symbol"].astype(str).str.replace(".", "-", regex=False).unique())
        cache.parent.mkdir(parents=True, exist_ok=True)
        pd.Series(syms, name="ticker").to_csv(cache, index=False)
        return syms
    except Exception as e:
        if cache.exists():
            log(f"Could not refresh S&P 500 list ({e}); using cached list.")
            return pd.read_csv(cache)["ticker"].tolist()
        sys.exit(f"Could not download the S&P 500 list: {e}")


# ---------------------------------------------------------------- data
def _clean(d):
    d = d.dropna(subset=["Open", "High", "Low", "Close"])
    idx = pd.to_datetime(d.index)
    if idx.tz is not None:
        idx = idx.tz_convert(ET).tz_localize(None)
    d = d.copy()
    d.index = idx.normalize()
    return d


def split_by_ticker(df, tickers):
    out = {}
    if df is None or df.empty:
        return out
    if isinstance(df.columns, pd.MultiIndex):
        have = set(df.columns.get_level_values(0))
        for t in tickers:
            if t in have:
                d = _clean(df[t])
                if not d.empty:
                    out[t] = d
    elif len(tickers) == 1:
        d = _clean(df)
        if not d.empty:
            out[tickers[0]] = d
    return out


def download_daily(tickers, period):
    data = {}
    for i in range(0, len(tickers), 100):
        chunk = tickers[i:i + 100]
        for attempt in range(3):
            try:
                df = yf.download(chunk, period=period, interval="1d", group_by="ticker",
                                 auto_adjust=False, threads=True, progress=False)
                data.update(split_by_ticker(df, chunk))
                break
            except Exception as e:
                log(f"Download error (try {attempt + 1}/3): {e}")
                time.sleep(5)
    return data


def build_levels(today):
    """EMAs as of the last COMPLETED daily and weekly candles, plus this week's prior days."""
    tickers = sorted((set(get_sp500()) | set(EXTRA_TICKERS)) - set(EXCLUDE_TICKERS))
    log(f"Building levels for {len(tickers)} tickers...")
    history = download_daily(tickers, "5y")
    monday = today - timedelta(days=today.weekday())
    out = {}
    for t, d in history.items():
        d = d[d.index.date < today]
        if len(d) < 60 or (MIN_PRICE and d["Close"].iloc[-1] < MIN_PRICE):
            continue
        ema = {}
        for n in EMA_LENGTHS:
            ema[f"D{n}"] = round(float(d["Close"].ewm(span=n, adjust=False).mean().iloc[-1]), 4)
        wk = (d[d.index.date < monday].resample("W-FRI")
              .agg({"Open": "first", "High": "max", "Low": "min", "Close": "last"}).dropna())
        if len(wk) >= 60:
            for n in EMA_LENGTHS:
                ema[f"W{n}"] = round(float(wk["Close"].ewm(span=n, adjust=False).mean().iloc[-1]), 4)
        wp = d[d.index.date >= monday]
        week = None if wp.empty else {"open": float(wp["Open"].iloc[0]),
                                      "high": float(wp["High"].max()),
                                      "low": float(wp["Low"].min())}
        out[t] = {"ema": ema, "week": week}
    log(f"Levels ready for {len(out)} stocks.")
    return {"date": today.isoformat(), "levels": out}


def live_bars(tickers, today):
    bars = {}
    for t, d in download_daily(tickers, "5d").items():
        row = d[d.index.date == today]
        if row.empty:
            continue
        r = row.iloc[-1]
        bars[t] = {"open": float(r["Open"]), "high": float(r["High"]),
                   "low": float(r["Low"]), "close": float(r["Close"])}
    return bars


# ---------------------------------------------------------------- logic
def candles(info, bar):
    out = {"D": bar}
    wk = info.get("week")
    out["W"] = dict(bar) if not wk else {
        "open": wk["open"], "high": max(wk["high"], bar["high"]),
        "low": min(wk["low"], bar["low"]), "close": bar["close"]}
    return out


def evaluate(info, bar):
    tol = TOUCH_PCT / 100
    rows = []
    for tf, c in candles(info, bar).items():
        if tf not in TIMEFRAMES:
            continue
        for n in EMA_LENGTHS:
            prev = info["ema"].get(f"{tf}{n}")
            if prev is None:
                continue
            k = 2 / (n + 1)
            ema = prev + k * (c["close"] - prev)
            ema_open = prev + k * (c["open"] - prev)
            side = "below" if c["open"] < ema_open else "above"
            touched = (c["high"] >= ema * (1 - tol)) if side == "below" \
                else (c["low"] <= ema * (1 + tol))
            rows.append({"tf": tf, "n": n, "ema": ema, "side": side, "touched": touched, **c})
    return rows


def status_text(r):
    if r["side"] == "below":
        return "back below - rejecting so far" if r["close"] < r["ema"] \
            else "trading above - breaking through"
    return "holding above - bouncing so far" if r["close"] > r["ema"] \
        else "trading below - breaking down"


def short_line(t, r):
    want = "REJ?" if r["side"] == "below" else "BNC?"
    return f"{t} {r['tf']}{r['n']} {want} EMA {r['ema']:.2f} px {r['close']:.2f}"


def dispatch(new):
    if not new:
        return
    new.sort(key=lambda x: (x[1]["tf"] != "W", x[0]))
    for t, r in new:
        log("TOUCH " + short_line(t, r))
    if len(new) <= MAX_SINGLE_ALERTS:
        for t, r in new:
            below = r["side"] == "below"
            dist = (r["close"] - r["ema"]) / r["ema"] * 100
            period = "Day" if r["tf"] == "D" else "Week"
            body = (f"Testing from {r['side']} -> watch for {'REJECTION' if below else 'BOUNCE'}\n"
                    f"EMA {r['ema']:.2f} | Price {r['close']:.2f} ({dist:+.2f}%)\n"
                    f"{period} high {r['high']:.2f}  low {r['low']:.2f}\n"
                    f"Now: {status_text(r)}")
            notify(f"{t} {TF_NAME[r['tf']]} {r['n']} EMA touch", body,
                   tags="red_circle" if below else "green_circle",
                   priority="high" if r["tf"] == "W" else "default", click=tv_link(t))
    else:
        lines = [short_line(t, r) for t, r in new]
        body = "\n".join(lines[:30])
        if len(lines) > 30:
            body += f"\n...and {len(lines) - 30} more (see the Actions log)"
        notify(f"{len(new)} new EMA touches", body, tags="bell")


def close_report(levels, bars, today):
    is_friday = today.weekday() == 4
    results, broke = [], 0
    for t, bar in bars.items():
        for r in evaluate(levels[t], bar):
            if not r["touched"] or (r["tf"] == "W" and not is_friday):
                continue
            if r["side"] == "below":
                verdict = "REJECTED" if r["close"] < r["ema"] else "BROKE ABOVE"
            else:
                verdict = "BOUNCED" if r["close"] > r["ema"] else "BROKE BELOW"
            broke += verdict.startswith("BROKE")
            results.append({"date": today.isoformat(), "ticker": t, "timeframe": TF_NAME[r["tf"]],
                            "ema_len": r["n"], "ema": round(r["ema"], 2), "open": r["open"],
                            "high": r["high"], "low": r["low"], "close": r["close"],
                            "result": verdict})
    if results:
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(results).to_csv(RESULTS_DIR / f"{today.isoformat()}.csv", index=False)

    held = sorted([x for x in results if x["result"] in ("REJECTED", "BOUNCED")],
                  key=lambda x: (x["timeframe"] != "Weekly", x["result"], x["ticker"]))
    lines = [f"{x['ticker']} {x['timeframe'][0]}{x['ema_len']} {x['result']} "
             f"@ {x['ema']:.2f} (close {x['close']:.2f})" for x in held]
    body = "\n".join(lines[:40]) if lines else "No confirmed rejections or bounces today."
    if len(lines) > 40:
        body += f"\n...and {len(lines) - 40} more (see data/results CSV)"
    body += f"\n\n{broke} touches broke through instead."
    if not is_friday:
        body += "\nWeekly results come on Friday."
    title = (f"Close report {today:%b %d}: "
             f"{sum(x['result'] == 'REJECTED' for x in held)} rejected, "
             f"{sum(x['result'] == 'BOUNCED' for x in held)} bounced")
    notify(title, body, tags="checkered_flag")
    log(title)


# ---------------------------------------------------------------- one run
def run():
    now = datetime.now(ET)
    today = now.date()
    iso = today.isoformat()
    at = lambda h, m: now.replace(hour=h, minute=m, second=0, microsecond=0)

    if today.weekday() >= 5 or now < at(8, 0):
        log("Outside the trading window. Nothing to do.")
        return

    state = load_json(STATE_FILE, {})
    if state.get("holiday") == iso:
        log("Market holiday today.")
        return
    if state.get("close_report") == iso:
        log("Close report already sent today.")
        return

    monday = (today - timedelta(days=today.weekday())).isoformat()
    sent = {k for k in state.get("sent", []) if k.rsplit("|", 1)[-1] >= monday}

    levels = load_json(LEVELS_FILE, {})
    if levels.get("date") != iso:
        levels = build_levels(today)
        if not levels["levels"]:
            log("Level build returned no data (Yahoo problem?). Will retry next run.")
            return
        save_json(LEVELS_FILE, levels)
        notify("EMA scanner ready", f"Watching {len(levels['levels'])} stocks today: "
               f"{'/'.join(TF_NAME[x] for x in TIMEFRAMES)} "
               f"{'/'.join(map(str, EMA_LENGTHS))} EMAs.", tags="rocket", priority="low")

    if now < at(9, 31) or at(16, 0) <= now < at(16, 10):
        log("Waiting (pre-open or for final closing prices).")
        state["sent"] = sorted(sent)
        save_json(STATE_FILE, state)
        return

    lv = levels["levels"]
    bars = live_bars(list(lv), today)
    if not bars:
        if now > at(9, 50):
            state["holiday"] = iso
            log("No prices for today - treating as a market holiday.")
        else:
            log("No prices yet.")
        state["sent"] = sorted(sent)
        save_json(STATE_FILE, state)
        return

    if now < at(16, 0):
        new = []
        for t, bar in bars.items():
            for r in evaluate(lv[t], bar):
                if not r["touched"]:
                    continue
                period = iso if r["tf"] == "D" else monday
                key = f"{t}|{r['tf']}{r['n']}|{period}"
                if key not in sent:
                    sent.add(key)
                    new.append((t, r))
        dispatch(new)
        log(f"Scan: {len(bars)} prices, {len(new)} new touches.")
    else:
        close_report(lv, bars, today)
        state["close_report"] = iso

    state["sent"] = sorted(sent)
    save_json(STATE_FILE, state)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--test", action="store_true", help="send a test notification and exit")
    if ap.parse_args().test:
        notify("EMA scanner test", "If you see this, ntfy is working.",
               tags="white_check_mark", click=tv_link("AMD"))
        print("Test sent." if NTFY_TOPIC else "Set NTFY_TOPIC first.")
    else:
        run()
