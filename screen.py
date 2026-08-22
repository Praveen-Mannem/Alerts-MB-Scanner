"""
NSE Universe Screener
======================================================================
Filters a maintained universe of liquid NSE stocks (nse_universe.csv) by
recent price and average volume via yfinance, and writes whatever
qualifies to watchlist.csv, which scanner.py then scans regularly.

WHY A STATIC UNIVERSE FILE INSTEAD OF SCRAPING NSE DIRECTLY:
nseindia.com actively blocks traffic from cloud/CI IP ranges (GitHub
Actions, AWS, Azure, etc.) — this is a long-standing, IP-range-based block
that doesn't respond to better headers or retries. yfinance, by contrast,
works reliably from GitHub Actions (confirmed in production use here).
So we keep the "which stocks exist" list as a maintained local file and
only use yfinance for the "does it pass the price/volume filter" part.

nse_universe.csv is a maintained list of liquid NSE stocks across sectors
— not a live mirror of an official index. Edit it directly to add/remove
names as you like; it only needs occasional updates (e.g. a new listing
you want tracked), not daily refreshing.

Filters applied:
  - Last close price between MIN_PRICE and MAX_PRICE (inclusive)
  - 20-day average volume >= MIN_AVG_VOLUME (a liquidity floor)
"""

import os
import time
import pandas as pd
import yfinance as yf

UNIVERSE_FILE = os.path.join(os.path.dirname(__file__), "nse_universe.csv")
WATCHLIST_FILE = os.path.join(os.path.dirname(__file__), "watchlist.csv")

MIN_PRICE = 100.0
MAX_PRICE = 1000.0
MIN_AVG_VOLUME = 500_000  # 20-day average daily volume floor — adjust as needed

BATCH_SIZE = 100  # symbols per yfinance batch call
LOOKBACK_PERIOD = "1mo"


def load_universe():
    with open(UNIVERSE_FILE) as f:
        return [line.strip() for line in f if line.strip()]


def chunked(seq, size):
    for i in range(0, len(seq), size):
        yield seq[i:i + size]


def screen_batch(symbols):
    """Returns list of symbols passing the price/volume filter for this batch."""
    tickers = " ".join(f"{s}.NS" for s in symbols)
    try:
        data = yf.download(
            tickers,
            period=LOOKBACK_PERIOD,
            interval="1d",
            group_by="ticker",
            auto_adjust=False,
            progress=False,
            threads=True,
        )
    except Exception as e:
        print(f"Batch download failed ({len(symbols)} symbols): {e}")
        return []

    passed = []
    for s in symbols:
        col = f"{s}.NS"
        try:
            if len(symbols) == 1:
                sub = data
            else:
                sub = data[col]
            sub = sub.dropna(subset=["Close", "Volume"])
            if sub.empty:
                continue
            last_close = float(sub["Close"].iloc[-1])
            avg_vol = float(sub["Volume"].mean())
            if MIN_PRICE <= last_close <= MAX_PRICE and avg_vol >= MIN_AVG_VOLUME:
                passed.append(s)
        except Exception:
            continue
    return passed


def main():
    symbols = load_universe()
    print(f"Loaded {len(symbols)} symbols from nse_universe.csv. "
          f"Screening in batches of {BATCH_SIZE}...")

    qualifying = []
    for batch_num, batch in enumerate(chunked(symbols, BATCH_SIZE), start=1):
        print(f"Batch {batch_num}: {len(batch)} symbols")
        qualifying.extend(screen_batch(batch))
        time.sleep(1)  # be polite between batches

    qualifying = sorted(set(qualifying))
    print(f"{len(qualifying)} symbols passed the filter "
          f"(price {MIN_PRICE}-{MAX_PRICE}, avg volume >= {MIN_AVG_VOLUME:,}).")

    if not qualifying:
        print("WARNING: 0 symbols passed. Leaving watchlist.csv unchanged "
              "rather than wiping it out — check MIN_PRICE/MAX_PRICE/MIN_AVG_VOLUME.")
        return

    with open(WATCHLIST_FILE, "w") as f:
        f.write("symbol\n")
        for s in qualifying:
            f.write(f"{s}\n")

    print(f"Wrote {len(qualifying)} symbols to {WATCHLIST_FILE}.")


if __name__ == "__main__":
    main()
