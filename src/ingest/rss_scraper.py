# src/ingest/rss_scraper.py
import feedparser
import pandas as pd
from datetime import datetime

RSS_FEEDS = [
    "https://feeds.reuters.com/reuters/businessNews",
    "https://feeds.reuters.com/reuters/technologyNews",
    "https://feeds.reuters.com/reuters/UKdomesticNews",
]

def fetch_rss_articles():
    articles = []
    for url in RSS_FEEDS:
        feed = feedparser.parse(url)
        for entry in feed.entries:
            articles.append({
                "timestamp": entry.get("published", datetime.now().isoformat()),
                "source": "reuters_rss",
                "title": entry.get("title", ""),
                "text": entry.get("summary", entry.get("title", "")),
                "url": entry.get("link", "")
            })
    
    df = pd.DataFrame(articles)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    df.dropna(subset=["timestamp"], inplace=True)
    df.drop_duplicates(subset=["url"], inplace=True)
    
    df.to_csv("data/raw/rss_articles.csv", index=False)
    print(f"RSS: saved {len(df)} articles")
    return df

if __name__ == "__main__":
    fetch_rss_articles()