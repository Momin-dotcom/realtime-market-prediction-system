# src/sentiment/labeler.py
import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

analyzer = SentimentIntensityAnalyzer()

def get_sentiment(text: str) -> str:
    if not isinstance(text, str) or text.strip() == "":
        return "NEUTRAL"
    score = analyzer.polarity_scores(text)["compound"]
    if score >= 0.05:
        return "POSITIVE"
    elif score <= -0.05:
        return "NEGATIVE"
    else:
        return "NEUTRAL"

def label_all_sources():
    dfs = []
    files = {
        "data/raw/rss_articles.csv": "rss",
        "data/raw/reddit_posts.csv": "reddit",
        "data/raw/twitter_posts.csv": "twitter",
    }
    
    for path, source in files.items():
        try:
            df = pd.read_csv(path)
            df["sentiment"] = df["text"].apply(get_sentiment)
            dfs.append(df)
            print(f"{source}: labeled {len(df)} rows")
        except FileNotFoundError:
            print(f"Warning: {path} not found, skipping")
    
    combined = pd.concat(dfs, ignore_index=True)
    combined["timestamp"] = pd.to_datetime(combined["timestamp"], utc=True, errors="coerce")
    combined.dropna(subset=["timestamp"], inplace=True)
    combined.sort_values("timestamp", inplace=True)
    
    combined.to_csv("data/labeled/sentiment_labeled.csv", index=False)
    print(f"Labeled: total {len(combined)} rows saved")
    return combined

if __name__ == "__main__":
    label_all_sources()