#!/usr/bin/env python3

from pathlib import Path
import pandas as pd
import yfinance as yf

# ──────────────────────────── USER SETTINGS ────────────────────────────────
INPUT_CSV       = Path("data/assets/business_relationship_graph.csv")          # ORIGINAL FILE
INFO_CSV        = Path("data/assets/american_industry.csv")                  # FUNDAMENTALS
YF_SHARE_CSV    = Path("data/assets/yahoo_finance_shareholders.csv")  # HOLDERS
STOCK_PRICE_CSV = Path("data/assets/stock_prices.csv")                          # STOCK PRICES
# ────────────────────────────────────────────────────────────────────────────

# 1. ── Load data ───────────────────────────────────────────────────────────
df = pd.read_csv(INPUT_CSV)
tickers = sorted(df["Source"].dropna().unique())

print(f"Found {len(tickers)} unique tickers in ‘Source’.")

# 2. ── Yahoo Finance company industries ─────────────────────────────────
fields = [
    "symbol", "shortName", "currentPrice",
    "sector", "marketCap",
]

records = []
for tk in tickers:
    try:
        info = yf.Ticker(tk).info     # network request
        row = {f: info.get(f) for f in fields}
        row["symbol"] = tk            # guarantee symbol
        records.append(row)
    except Exception as e:
        records.append({"symbol": tk, "error": str(e)})
        print(f"  • {tk}: lookup failed → {e}")

info_df = pd.DataFrame(records)
info_df.to_csv(INFO_CSV, index=False)


#3. ── Yahoo Finance major holders ─────────────────────────────────
print("\nFetching institutional holders from Yahoo Finance …")

holder_frames = []
for tk in tickers:
    try:
        ih = yf.Ticker(tk).institutional_holders  # DataFrame or None
        if ih is None or ih.empty:
            print(f"  • {tk}: no institutional‑holder data")
            continue

        ih = ih.copy()                     # don’t alter raw frame
        ih.insert(0, "Ticker", tk)         # add ticker col first

        # Harmonise column names
        ih.rename(
            columns={
                "Holder": "Stockholder",
                "Shares": "SharesHeld",
                "Date Reported": "DateReported",
                "pctHeld": "pctHeld",
            },
            inplace=True,
        )

        # Use DateReported both as StartDate & EndDate (single snapshot)
        ih["DateReported"] = ih["DateReported"]

        holder_frames.append(ih[
            ["Ticker", "Stockholder", "SharesHeld", "pctHeld", "DateReported"]
        ])

    except Exception as e:
        print(f"{tk}: holder fetch failed → {e}")

if holder_frames:
    holders_df = (
        pd.concat(holder_frames, ignore_index=True)
    )
    holders_df.to_csv(YF_SHARE_CSV, index=False)
    print(f"Wrote {YF_SHARE_CSV}  ({len(holders_df)} rows)")
else:
    print("No institutional‑holder data returned for any ticker.")

# 4. ── Yahoo Finance stock prices ─────────────────────────────────
print("\nFetching stock prices from Yahoo Finance …")
price_frames = []

#get prices for the 2023-2024 period
for tk in tickers:
    try:
        stock = yf.Ticker(tk)
        history = stock.history(start="2023-01-01", end="2024-12-31")
        history = history.reset_index()
        history["Date"] = history["Date"].dt.strftime("%Y-%m-%d")
        history["Ticker"] = tk
        price_frames.append(history)
    except Exception as e:
        print(f"{tk}: price fetch failed → {e}")

if price_frames:
    prices_df = pd.concat(price_frames, ignore_index=True)
    prices_df.to_csv(STOCK_PRICE_CSV, index=False)
    print(f"Wrote {STOCK_PRICE_CSV}  ({len(prices_df)} rows)")
else:
    print("No stock price data returned for any ticker.")


