# src/ingest/yahoo_scraper.py
import yfinance as yf
import pandas as pd
from datetime import datetime

TICKERS = ["SPY", "AAPL", "BTC-USD"]

def fetch_yahoo_data(period="7d", interval="1h"):
    all_data = []
    for ticker in TICKERS:
        df = yf.download(ticker, period=period, interval=interval)
        df["ticker"] = ticker
        df["source"] = "yahoo_finance"
        df.reset_index(inplace=True)
        all_data.append(df)
    
    combined = pd.concat(all_data, ignore_index=True)
    combined.columns = [c.lower() for c in combined.columns]
    
    # compute price direction — this is your target label
    combined["price_change_pct"] = combined.groupby("ticker")["close"].pct_change() * 100
    combined["direction"] = combined["price_change_pct"].apply(
        lambda x: "UP" if x > 0 else "DOWN"
    )
    
    combined.to_csv("data/raw/yahoo_finance.csv", index=False)
    print(f"Yahoo: saved {len(combined)} rows")
    return combined

if __name__ == "__main__":
    fetch_yahoo_data()