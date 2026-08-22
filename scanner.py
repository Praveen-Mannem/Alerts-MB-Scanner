"""
Mother & Baby Inside Bar Scanner (free yfinance data source)
==============================================================================
Scans a watchlist for fresh BUY/SELL breakout signals across 15m / 1H / 1D
timeframes and pushes new signals to Telegram. Designed to run unattended
(e.g. GitHub Actions) every ~15 minutes during market hours.

Uses Yahoo Finance (yfinance) for NSE data via the ".NS" suffix — free, no
API key or subscription needed. Two trade-offs vs. a paid broker feed:
  - Quotes/candles run roughly 15-20 min delayed, not tick-real-time.
  - Intraday history (15m/1H) is only available for the trailing ~60 days,
    which is irrelevant here since we only need the last few hundred bars.

State (which signals have already been alerted) is kept in state.json so the
same breakout isn't re-sent every run.

ENV VARS REQUIRED:
    TELEGRAM_BOT_TOKEN  - from @BotFather
    TELEGRAM_CHAT_ID    - chat id to send alerts to
"""

import os
import json
import time
import csv
from datetime import datetime

import requests
import pandas as pd
import yfinance as yf

TG_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN", "")
TG_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID", "")

STATE_FILE = os.path.join(os.path.dirname(__file__), "state.json")
WATCHLIST_FILE = os.path.join(os.path.dirname(__file__), "watchlist.csv")

# yfinance interval strings and how much history to request for each
TIMEFRAMES = {
    "1H": {"interval": "60m", "period": "180d"},
    "1D": {"interval": "1d", "period": "2y"},
    "1W": {"interval": "1wk", "period": "5y"},
}

# ---- Pattern engine parameters (mirrors Pine script defaults) ----
FILTER_DOJI = True
MIN_MOTHER_BODY_PCT = 10.0
FILTER_EMA = True
FILTER_WICKS = True
REQUIRE_OPPOSITE_COLORS = False


# ------------------------------------------------------------------
# Data fetch (yfinance)
# ------------------------------------------------------------------
def fetch_candles(nse_symbol, tf_key):
    """Returns a DataFrame [open, high, low, close, volume, timestamp] ascending by time."""
    cfg = TIMEFRAMES[tf_key]
    ticker = f"{nse_symbol}.NS"

    raw = yf.download(
        ticker,
        interval=cfg["interval"],
        period=cfg["period"],
        auto_adjust=False,
        progress=False,
        multi_level_index=False,
    )
    if raw is None or raw.empty:
        return pd.DataFrame()

    raw = raw.reset_index()
    time_col = "Datetime" if "Datetime" in raw.columns else "Date"
    df = pd.DataFrame({
        "open": raw["Open"],
        "high": raw["High"],
        "low": raw["Low"],
        "close": raw["Close"],
        "volume": raw["Volume"],
        "timestamp": raw[time_col].astype(str),
    })
    return df.reset_index(drop=True)


# ------------------------------------------------------------------
# Mother & Baby pattern engine (direct port of the Pine state machine)
# ------------------------------------------------------------------
def compute_signal(df):
    """
    Walks the candle series bar-by-bar, replicating the Pine script's
    inside-bar tracking + breakout + wick-rejection + EMA trend filter logic.
    Returns the status as of the LAST closed bar: "BUY", "SELL", "INSIDE", or "NONE".
    """
    if len(df) < 60:
        return "NONE", {}

    df = df.copy()
    df["ema21"] = df["close"].ewm(span=21, adjust=False).mean()
    df["ema50"] = df["close"].ewm(span=50, adjust=False).mean()

    in_ib = False
    m_high = m_low = None
    status = "NONE"
    detail = {}

    for i in range(1, len(df)):
        prev = df.iloc[i - 1]
        cur = df.iloc[i]
        in_ib_before = in_ib

        m_range = prev["high"] - prev["low"]
        m_body = abs(prev["close"] - prev["open"])
        m_body_ratio = (m_body / m_range * 100) if m_range > 0 else 0.0
        mother_valid = (not FILTER_DOJI) or (m_body_ratio >= MIN_MOTHER_BODY_PCT)
        baby_inside = (cur["high"] <= prev["high"]) and (cur["low"] >= prev["low"])
        mother_red = prev["close"] < prev["open"]
        mother_green = prev["close"] > prev["open"]
        baby_red = cur["close"] < cur["open"]
        baby_green = cur["close"] > cur["open"]
        opp_color = (mother_red and baby_green) or (mother_green and baby_red)
        color_ok = (not REQUIRE_OPPOSITE_COLORS) or opp_color
        is_mb_pattern = mother_valid and baby_inside and color_ok

        if is_mb_pattern and not in_ib:
            m_high, m_low = prev["high"], prev["low"]
            in_ib = True
        elif in_ib and (cur["high"] <= m_high) and (cur["low"] >= m_low):
            in_ib = True
        else:
            if not is_mb_pattern and m_high is not None and (
                cur["close"] > m_high or cur["close"] < m_low or cur["high"] > m_high or cur["low"] < m_low
            ):
                in_ib = False

        bo_rng = cur["high"] - cur["low"]
        valid_buy_wick = (not FILTER_WICKS) or (bo_rng > 0 and ((cur["close"] - cur["low"]) / bo_rng) >= 0.5)
        valid_sell_wick = (not FILTER_WICKS) or (bo_rng > 0 and ((cur["high"] - cur["close"]) / bo_rng) >= 0.5)

        raw_buy = in_ib_before and m_high is not None and (cur["close"] > m_high) and (cur["close"] > cur["open"]) and valid_buy_wick
        raw_sell = in_ib_before and m_low is not None and (cur["close"] < m_low) and (cur["close"] < cur["open"]) and valid_sell_wick

        ema_bull = cur["ema21"] > cur["ema50"]
        ema_bear = cur["ema21"] < cur["ema50"]

        buy_sig = raw_buy and (not FILTER_EMA or ema_bull)
        sell_sig = raw_sell and (not FILTER_EMA or ema_bear)

        if buy_sig:
            status = "BUY"
            in_ib = False
            detail = {"entry": float(cur["close"]), "opp_color": bool(opp_color), "time": str(cur.get("timestamp"))}
        elif sell_sig:
            status = "SELL"
            in_ib = False
            detail = {"entry": float(cur["close"]), "opp_color": bool(opp_color), "time": str(cur.get("timestamp"))}
        elif is_mb_pattern:
            status = "INSIDE"
        else:
            if status not in ("BUY", "SELL") and not in_ib:
                status = "NONE"

    return status, detail


# ------------------------------------------------------------------
# Telegram
# ------------------------------------------------------------------
def send_telegram(text):
    if not TG_TOKEN or not TG_CHAT_ID:
        print("Telegram not configured, skipping send:\n", text)
        return
    url = f"https://api.telegram.org/bot{TG_TOKEN}/sendMessage"
    resp = requests.post(url, data={"chat_id": TG_CHAT_ID, "text": text, "parse_mode": "HTML"}, timeout=15)
    if resp.status_code != 200:
        print("Telegram send failed:", resp.text)


# ------------------------------------------------------------------
# State (avoid duplicate alerts)
# ------------------------------------------------------------------
def load_state():
    if os.path.exists(STATE_FILE):
        with open(STATE_FILE) as f:
            return json.load(f)
    return {}


def save_state(state):
    with open(STATE_FILE, "w") as f:
        json.dump(state, f, indent=2)


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------
def load_watchlist():
    with open(WATCHLIST_FILE) as f:
        return [row["symbol"].strip() for row in csv.DictReader(f) if row.get("symbol", "").strip()]


def main():
    watchlist = load_watchlist()
    state = load_state()
    alerts = []

    for symbol in watchlist:
        for tf_key in TIMEFRAMES:
            try:
                df = fetch_candles(symbol, tf_key)
                status, detail = compute_signal(df)
            except Exception as e:
                print(f"[{symbol} {tf_key}] error: {e}")
                continue

            key = f"{symbol}:{tf_key}"
            last_alerted = state.get(key, {})

            if status in ("BUY", "SELL") and last_alerted.get("time") != detail.get("time"):
                conf = " \U0001F525 HIGH CONFIDENCE" if detail.get("opp_color") else ""
                msg = (
                    f"\u26a1 <b>{status} \u2014 {symbol}</b> ({tf_key}){conf}\n"
                    f"Entry: {detail.get('entry')}\n"
                    f"Bar time: {detail.get('time')}"
                )
                alerts.append(msg)
                state[key] = {"status": status, "time": detail.get("time")}

            time.sleep(0.3)  # be polite to Yahoo's endpoint

    if alerts:
        send_telegram("\n\n".join(alerts))
        print(f"Sent {len(alerts)} alert(s).")
    else:
        print("No new signals this run.")

    save_state(state)


if __name__ == "__main__":
    main()
