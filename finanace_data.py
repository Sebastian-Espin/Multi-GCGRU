import yfinance as yf
import requests, pandas as pd, itertools

# Liste d'actions américaines
tickers = ["AAPL", "MSFT", "TSLA", "GOOGL", "AMZN", "NVDA"]

# Dates spécifiques que tu veux observer
dates = ["2024-12-31", "2025-01-31", "2025-03-31"]

# 1. Extraire les infos fondamentales : secteur et industrie
sector_industry_data = []

for ticker in tickers:
    try:
        info = yf.Ticker(ticker).info
        sector_industry_data.append({
            "ticker": ticker,
            "sector": info.get("sector", "Unknown"),
            "industry": info.get("industryDisp", "Unknown"),
            "company_name": info.get("shortName", "Unknown")
        })
    except Exception as e:
        print(f"Erreur pour {ticker} : {e}")

sector_df = pd.DataFrame(sector_industry_data)
print("=== Table des secteurs et industries ===")
print(sector_df)

# 2. Extraire les données financières aux dates précises
financial_data = []

for ticker in tickers:
    stock = yf.Ticker(ticker)
    history = stock.history(start=min(dates), end="2025-04-01")  # prendre une marge
    history = history.reset_index()
    history["Date"] = history["Date"].dt.strftime("%Y-%m-%d")

    for d in dates:
        row = history[history["Date"] == d]
        if not row.empty:
            row_data = row.iloc[0]
            financial_data.append({
                "ticker": ticker,
                "date": d,
                "close": row_data["Close"],
                "open": row_data["Open"],
                "high": row_data["High"],
                "low": row_data["Low"],
                "volume": row_data["Volume"]
            })
        else:
            financial_data.append({
                "ticker": ticker,
                "date": d,
                "close": None,
                "open": None,
                "high": None,
                "low": None,
                "volume": None
            })

financial_df = pd.DataFrame(financial_data)
print("\n=== Table des données financières ===")
print(financial_df)
print(yf.Ticker("AAPL").major_holders)

# APIKEY = "wdludPKPrLkrURy9iluWxSQnD35L0Akm"
# tickers = ["AAPL","MSFT","TSLA","NVDA","AMZN"]
# rows = []

# for t in tickers:
#     url = f"https://financialmodelingprep.com/api/v3/institutional-holder/{t}?apikey={APIKEY}"
#     data = requests.get(url, timeout=30).json()
#     for rec in data:          # each rec: holder, shares, dateReported, change
#         rows.append({"ticker": t,
#                      "holder_name": rec["holder"],
#                      "hold_amount": rec["shares"],
#                      "report_date": rec["dateReported"]})
# share_df = pd.DataFrame(rows)
# share_df.to_csv("shareholder.csv", index=False)

print(yf.Ticker("NVDA").institutional_holders)