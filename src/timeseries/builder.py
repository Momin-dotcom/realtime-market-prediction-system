# src/timeseries/builder.py
import pandas as pd
import numpy as np

def build_timeseries(window="1h"):
    # load sentiment data
    sentiment_df = pd.read_csv("data/labeled/sentiment_labeled.csv")
    sentiment_df["timestamp"] = pd.to_datetime(sentiment_df["timestamp"], utc=True)
    sentiment_df.set_index("timestamp", inplace=True)
    
    # load price data
    price_df = pd.read_csv("data/raw/yahoo_finance.csv")
    price_df["datetime"] = pd.to_datetime(price_df["datetime"], utc=True)
    price_df = price_df[price_df["ticker"] == "SPY"]  # use SPY as benchmark
    price_df.set_index("datetime", inplace=True)
    
    # aggregate sentiment per time window
    sentiment_agg = sentiment_df.resample(window).apply({
        "sentiment": [
            ("positive_count", lambda x: (x == "POSITIVE").sum()),
            ("negative_count", lambda x: (x == "NEGATIVE").sum()),
            ("neutral_count",  lambda x: (x == "NEUTRAL").sum()),
        ]
    })
    sentiment_agg.columns = ["positive_count", "negative_count", "neutral_count"]
    sentiment_agg["total"] = sentiment_agg.sum(axis=1)
    sentiment_agg["sentiment_ratio"] = (
        (sentiment_agg["positive_count"] - sentiment_agg["negative_count"])
        / sentiment_agg["total"].replace(0, np.nan)
    ).fillna(0)
    
    # resample price to same window
    price_agg = price_df["close"].resample(window).last()
    price_change = price_df["price_change_pct"].resample(window).last()
    direction = price_df["direction"].resample(window).last()
    
    # join everything
    final = sentiment_agg.join(price_agg, how="inner")
    final = final.join(price_change, how="inner")
    final = final.join(direction, how="inner")
    final.dropna(inplace=True)
    
    final.to_csv("data/processed/timeseries_dataset.csv")
    print(f"Time-series: {len(final)} rows, {final.columns.tolist()}")
    return final

if __name__ == "__main__":
    build_timeseries()