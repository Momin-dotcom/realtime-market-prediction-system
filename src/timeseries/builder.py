import pandas as pd
import numpy as np

def add_technical_indicators(df):
    # Moving averages
    df["MA_7"]  = df["close"].rolling(window=7).mean()
    df["MA_21"] = df["close"].rolling(window=21).mean()
    
    # RSI
    delta = df["close"].diff()
    gain = delta.clip(lower=0).rolling(window=14).mean()
    loss = -delta.clip(upper=0).rolling(window=14).mean()
    rs = gain / loss.replace(0, np.nan)
    df["RSI"] = 100 - (100 / (1 + rs))
    
    # MACD
    ema12 = df["close"].ewm(span=12, adjust=False).mean()
    ema26 = df["close"].ewm(span=26, adjust=False).mean()
    df["MACD"] = ema12 - ema26
    df["MACD_signal"] = df["MACD"].ewm(span=9, adjust=False).mean()
    
    # Bollinger Bands
    df["BB_mid"]   = df["close"].rolling(window=20).mean()
    df["BB_upper"] = df["BB_mid"] + 2 * df["close"].rolling(window=20).std()
    df["BB_lower"] = df["BB_mid"] - 2 * df["close"].rolling(window=20).std()
    
    return df

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
    print(f"Sentiment distribution:\n{sentiment_df['sentiment'].value_counts()}")

    # ── load price ───────────────────────────────────────────────
    price_df = pd.read_csv("data/raw/yahoo_finance.csv")
    dt_col = next(c for c in price_df.columns if "date" in c.lower() or "time" in c.lower())
    price_df[dt_col] = pd.to_datetime(price_df[dt_col], utc=True, errors="coerce")
    price_df[dt_col] = price_df[dt_col].dt.tz_localize(None)
    price_df.dropna(subset=[dt_col], inplace=True)
    price_df.set_index(dt_col, inplace=True)
    if "ticker" in price_df.columns:
        price_df = price_df[price_df["ticker"] == "SPY"]

    print(f"Price rows: {len(price_df)}")

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
    sentiment_daily.index = pd.to_datetime(sentiment_daily.index).normalize()

    # ── aggregate price DAILY ────────────────────────────────────
    price_daily = price_df.resample("1D").agg(
        open  =("open",  "first"),
        high  =("high",  "max"),
        low   =("low",   "min"),
        close =("close", "last"),
        volume=("volume","sum"),
    ).dropna()
    price_daily.index = pd.to_datetime(price_daily.index).normalize()

    # fix first row NaN — fill price_change_pct
    price_daily["price_change_pct"] = price_daily["close"].pct_change() * 100
    price_daily["price_change_pct"].fillna(0, inplace=True)
    price_daily["direction"] = price_daily["price_change_pct"].apply(
        lambda x: "UP" if x > 0 else "DOWN"
    )

    # ── add technical indicators ─────────────────────────────────
    price_daily = add_technical_indicators(price_daily)

    # ── reindex sentiment to price date range ────────────────────
    sentiment_reindexed = sentiment_daily.reindex(price_daily.index)
    sentiment_cols = ["positive_count","negative_count","neutral_count","total","sentiment_ratio"]
    
    # only ffill/bfill where we actually have sentiment data
    # if no sentiment at all, leave as 0 (neutral)
    has_sentiment = sentiment_reindexed["total"].notna().any()
    if has_sentiment:
        sentiment_reindexed[sentiment_cols] = (
            sentiment_reindexed[sentiment_cols]
            .ffill()
            .bfill()
            .fillna(0)
        )
    else:
        sentiment_reindexed[sentiment_cols] = 0

    # ── join ─────────────────────────────────────────────────────
    combined = price_daily.join(sentiment_reindexed, how="left")
    combined.dropna(subset=["close", "MA_21", "RSI"], inplace=True)

    # ── normalize features ───────────────────────────────────────
    from sklearn.preprocessing import MinMaxScaler
    scale_cols = ["open","high","low","close","volume",
                  "price_change_pct","MA_7","MA_21","RSI","MACD","MACD_signal",
                  "BB_mid","BB_upper","BB_lower","sentiment_ratio"]
    
    scaler = MinMaxScaler()
    combined[[f"{c}_scaled" for c in scale_cols]] = scaler.fit_transform(
        combined[scale_cols].fillna(0)
    )

    print(f"\nFinal dataset rows: {len(combined)}")
    print(f"Columns: {combined.columns.tolist()}")
    print(f"\nSentiment distribution in final dataset:")
    print(f"  Positive days: {(combined['positive_count'] > 0).sum()}")
    print(f"  Negative days: {(combined['negative_count'] > 0).sum()}")
    print(f"  Sentiment ratio range: {combined['sentiment_ratio'].min():.2f} to {combined['sentiment_ratio'].max():.2f}")

    combined.to_csv("data/processed/timeseries_dataset.csv")
    print(f"\nSaved to data/processed/timeseries_dataset.csv")
    return combined

if __name__ == "__main__":
    build_timeseries()