import pandas as pd
import numpy as np

def build_timeseries(window="1D"):  # changed to DAILY to get more rows
    
    # ── load sentiment ──────────────────────────────────────────
    sentiment_df = pd.read_csv("data/labeled/sentiment_labeled.csv")
    sentiment_df["timestamp"] = pd.to_datetime(
        sentiment_df["timestamp"], utc=True, errors="coerce"
    )
    sentiment_df.dropna(subset=["timestamp"], inplace=True)
    
    # strip timezone for merging
    sentiment_df["timestamp"] = sentiment_df["timestamp"].dt.tz_localize(None)
    sentiment_df["date"] = sentiment_df["timestamp"].dt.date
    sentiment_df.set_index("timestamp", inplace=True)
    
    print(f"Sentiment rows: {len(sentiment_df)}")
    print(f"Sentiment range: {sentiment_df.index.min()} to {sentiment_df.index.max()}")

    # ── load price ───────────────────────────────────────────────
    price_df = pd.read_csv("data/raw/yahoo_finance.csv")
    
    # find datetime column
    dt_col = None
    for c in price_df.columns:
        if "date" in c.lower() or "time" in c.lower():
            dt_col = c
            break
    
    print(f"Price datetime column found: '{dt_col}'")
    print(f"Sample price timestamps: {price_df[dt_col].head(3).values}")
    
    price_df[dt_col] = pd.to_datetime(price_df[dt_col], utc=True, errors="coerce")
    price_df[dt_col] = price_df[dt_col].dt.tz_localize(None)
    price_df.dropna(subset=[dt_col], inplace=True)
    price_df.set_index(dt_col, inplace=True)
    
    # filter SPY only
    if "ticker" in price_df.columns:
        price_df = price_df[price_df["ticker"] == "SPY"]
    
    print(f"Price rows after filtering SPY: {len(price_df)}")
    print(f"Price range: {price_df.index.min()} to {price_df.index.max()}")
    print(f"Price columns: {price_df.columns.tolist()}")

    # ── aggregate sentiment DAILY ────────────────────────────────
    sentiment_daily = sentiment_df.resample("1D").agg(
        positive_count=("sentiment", lambda x: (x == "POSITIVE").sum()),
        negative_count=("sentiment", lambda x: (x == "NEGATIVE").sum()),
        neutral_count =("sentiment", lambda x: (x == "NEUTRAL").sum()),
    )
    sentiment_daily["total"] = sentiment_daily.sum(axis=1)
    sentiment_daily["sentiment_ratio"] = (
        (sentiment_daily["positive_count"] - sentiment_daily["negative_count"])
        / sentiment_daily["total"].replace(0, np.nan)
    ).fillna(0)
    
    print(f"\nSentiment daily buckets: {len(sentiment_daily)}")

    # ── aggregate price DAILY ────────────────────────────────────
    price_daily = price_df.resample("1D").agg(
        open  =("open",   "first"),
        high  =("high",   "max"),
        low   =("low",    "min"),
        close =("close",  "last"),
        volume=("volume", "sum"),
    ).dropna()
    
    price_daily["price_change_pct"] = price_daily["close"].pct_change() * 100
    price_daily["direction"] = price_daily["price_change_pct"].apply(
        lambda x: "UP" if x > 0 else "DOWN"
    )
    
    print(f"Price daily buckets: {len(price_daily)}")
    print(f"Price daily range: {price_daily.index.min()} to {price_daily.index.max()}")

    # ── join on DATE index ───────────────────────────────────────
    # normalize both indexes to date only so they match perfectly
    sentiment_daily.index = pd.to_datetime(sentiment_daily.index).normalize()
    price_daily.index = pd.to_datetime(price_daily.index).normalize()
    
    print(f"\nSentiment index sample: {sentiment_daily.index[:3].tolist()}")
    print(f"Price index sample: {price_daily.index[:3].tolist()}")

    # outer join first to see what we have
    combined = price_daily.join(sentiment_daily, how="left")
    
    # fill missing sentiment with 0 (weekend/no news days)
    sentiment_cols = ["positive_count","negative_count","neutral_count","total","sentiment_ratio"]
    combined[sentiment_cols] = combined[sentiment_cols].fillna(0)
    
    # drop rows with no price data
    combined.dropna(subset=["close"], inplace=True)
    
    print(f"\nFinal dataset rows: {len(combined)}")
    print(f"Columns: {combined.columns.tolist()}")
    print(combined.head(5))
    
    combined.to_csv("data/processed/timeseries_dataset.csv")
    print(f"\nSaved to data/processed/timeseries_dataset.csv")
    return combined

if __name__ == "__main__":
    build_timeseries()