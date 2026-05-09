import feedparser
import pandas as pd
from datetime import datetime

RSS_FEEDS = [
    "https://feeds.finance.yahoo.com/rss/2.0/headline?s=SPY&region=US&lang=en-US",
    "https://feeds.finance.yahoo.com/rss/2.0/headline?s=AAPL&region=US&lang=en-US",
    "https://feeds.finance.yahoo.com/rss/2.0/headline?s=MSFT&region=US&lang=en-US",
    "https://feeds.finance.yahoo.com/rss/2.0/headline?s=TSLA&region=US&lang=en-US",
    "https://feeds.finance.yahoo.com/rss/2.0/headline?s=BTC-USD&region=US&lang=en-US",
    "https://feeds.reuters.com/reuters/businessNews",
    "https://feeds.reuters.com/reuters/technologyNews",
    "https://feeds.marketwatch.com/marketwatch/topstories/",
    "https://feeds.marketwatch.com/marketwatch/marketpulse/",
    "https://rss.cnn.com/rss/money_news_international.rss",
]

def fetch_rss_articles():
    articles = []
    for url in RSS_FEEDS:
        print(f"Fetching: {url}")
        try:
            feed = feedparser.parse(url)
            for entry in feed.entries:
                articles.append({
                    "timestamp": entry.get("published", datetime.now().isoformat()),
                    "source": "rss",
                    "title": entry.get("title", ""),
                    "text": entry.get("summary", entry.get("title", "")),
                    "url": entry.get("link", "")
                })
            print(f"  Got {len(feed.entries)} articles")
        except Exception as e:
            print(f"  Error: {e}")

    df = pd.DataFrame(articles)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    df.dropna(subset=["timestamp"], inplace=True)
    df.drop_duplicates(subset=["url"], inplace=True)
    df.sort_values("timestamp", ascending=False, inplace=True)
    df.to_csv("data/raw/rss_articles.csv", index=False)
    print(f"RSS: saved {len(df)} articles total")
    return df

if __name__ == "__main__":
    fetch_rss_articles()