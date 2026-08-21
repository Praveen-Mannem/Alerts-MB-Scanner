"""
NSE Universe Screener
======================================================================
Pulls NSE's full official equity list, checks each stock's recent price
and average volume via yfinance, and filters down to stocks that fit:
  - Last close price between MIN_PRICE and MAX_PRICE (inclusive)
  - 20-day average volume >= MIN_AVG_VOLUME (a liquidity floor)

Writes the filtered symbol list to watchlist.csv, which scanner.py then
scans every 15 minutes. Meant to run once a day (before market open),
since price/volume screening doesn't need to be intraday-fresh.

No API key needed — NSE's list is a public CSV, and yfinance is free.
"""

import io
import time
import requests
import pandas as pd
import yfinance as yf

NSE_LIST_URL = "https://nsearchives.nseindia.com/content/equity/EQUITY_L.csv"
WATCHLIST_FILE = "watchlist.csv"

MIN_PRICE = 100.0
MAX_PRICE = 1000.0
MIN_AVG_VOLUME = 500_000  # 20-day average daily volume floor — adjust as needed

BATCH_SIZE = 150  # symbols per yfinance batch call
LOOKBACK_PERIOD = "1mo"


def fetch_nse_symbol_list():
    """
    NSE's site blocks generic requests without browser-like headers and an
    initial cookie-setting visit, so we mimic that handshake.
    """
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
        ),
        "Accept": "text/csv,application/csv,*/*",
    }
    session = requests.Session()
    session.headers.update(headers)
    session.get("https://www.nseindia.com", timeout=15)  # sets cookies
    resp = session.get(NSE_LIST_URL, timeout=20)
    resp.raise_for_status()

    df = pd.read_csv(io.StringIO(resp.text))
    df.columns = [c.strip() for c in df.columns]
    # Keep only the main equity board (excludes SME/illiquid series)
    if "SERIES" in df.columns:
        df = df[df["SERIES"].str.strip() == "EQ"]
    symbols = df["SYMBOL"].str.strip().tolist()
    return symbols


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
    print("Fetching NSE equity symbol list...")
    symbols = fetch_nse_symbol_list()
    print(f"Got {len(symbols)} symbols on the EQ board. Screening in batches of {BATCH_SIZE}...")

    qualifying = []
    for batch_num, batch in enumerate(chunked(symbols, BATCH_SIZE), start=1):
        print(f"Batch {batch_num}: {len(batch)} symbols")
        qualifying.extend(screen_batch(batch))
        time.sleep(1)  # be polite between batches

    qualifying = sorted(set(qualifying))
    print(f"{len(qualifying)} symbols passed the filter "
          f"(price {MIN_PRICE}-{MAX_PRICE}, avg volume >= {MIN_AVG_VOLUME:,}).")

    with open(WATCHLIST_FILE, "w") as f:
        f.write("symbol\n")
        for s in qualifying:
            f.write(f"{s}\n")

    print(f"Wrote {len(qualifying)} symbols to {WATCHLIST_FILE}.")


if __name__ == "__main__":
    main()
