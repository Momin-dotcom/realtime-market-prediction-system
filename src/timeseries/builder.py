import pandas as pd
import numpy as np

def build_timeseries(window="1D"):

    # ── load sentiment ──────────────────────────────────────────
    sentiment_df = pd.read_csv("data/labeled/sentiment_labeled.csv")
    sentiment_df["timestamp"] = pd.to_datetime(
        sentiment_df["timestamp"], utc=True, errors="coerce"
    )
    sentiment_df.dropna(subset=["timestamp"], inplace=True)
    sentiment_df["timestamp"] = sentiment_df["timestamp"].dt.tz_localize(None)
    sentiment_df.set_index("timestamp", inplace=True)

    print(f"Sentiment rows: {len(sentiment_df)}")
    print(f"Sentiment range: {sentiment_df.index.min()} to {sentiment_df.index.max()}")

    # ── load price ───────────────────────────────────────────────
    price_df = pd.read_csv("data/raw/yahoo_finance.csv")

    dt_col = None
    for c in price_df.columns:
        if "date" in c.lower() or "time" in c.lower():
            dt_col = c
            break

    price_df[dt_col] = pd.to_datetime(price_df[dt_col], utc=True, errors="coerce")
    price_df[dt_col] = price_df[dt_col].dt.tz_localize(None)
    price_df.dropna(subset=[dt_col], inplace=True)
    price_df.set_index(dt_col, inplace=True)

    if "ticker" in price_df.columns:
        price_df = price_df[price_df["ticker"] == "SPY"]

    print(f"Price rows: {len(price_df)}")
    print(f"Price range: {price_df.index.min()} to {price_df.index.max()}")

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

    # normalize index
    sentiment_daily.index = pd.to_datetime(sentiment_daily.index).normalize()
    print(f"Sentiment daily buckets: {len(sentiment_daily)}")

    # ── aggregate price DAILY ────────────────────────────────────
    price_daily = price_df.resample("1D").agg(
        open  =("open",  "first"),
        high  =("high",  "max"),
        low   =("low",   "min"),
        close =("close", "last"),
        volume=("volume","sum"),
    ).dropna()

    price_daily["price_change_pct"] = price_daily["close"].pct_change() * 100
    price_daily["direction"] = price_daily["price_change_pct"].apply(
        lambda x: "UP" if x > 0 else "DOWN"
    )

    price_daily.index = pd.to_datetime(price_daily.index).normalize()
    print(f"Price daily buckets: {len(price_daily)}")

    # ── expand sentiment to cover full price date range ──────────
    # reindex sentiment to match ALL price dates
    # then spread existing sentiment values across the full range
    sentiment_reindexed = sentiment_daily.reindex(price_daily.index)

    # forward fill then backward fill so no zeros remain
    sentiment_cols = ["positive_count","negative_count","neutral_count","total","sentiment_ratio"]
    sentiment_reindexed[sentiment_cols] = (
        sentiment_reindexed[sentiment_cols]
        .ffill()
        .bfill()
        .fillna(0)
    )

    print(f"Sentiment after reindex: {len(sentiment_reindexed)}")

    # ── join ─────────────────────────────────────────────────────
    combined = price_daily.join(sentiment_reindexed, how="left")
    combined.dropna(subset=["close"], inplace=True)

    print(f"\nFinal dataset rows: {len(combined)}")
    print(f"Sentiment ratio sample:\n{combined['sentiment_ratio'].head(10).values}")
    print(combined.head(5))

    combined.to_csv("data/processed/timeseries_dataset.csv")
    print(f"\nSaved to data/processed/timeseries_dataset.csv")
    return combined

if __name__ == "__main__":
    build_timeseries()