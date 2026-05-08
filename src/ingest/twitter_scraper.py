# src/ingest/twitter_scraper.py
import tweepy
import pandas as pd
from datetime import datetime

def fetch_tweets():
    client = tweepy.Client(bearer_token="YOUR_BEARER_TOKEN")
    
    queries = ["$SPY finance", "$BTC crypto market", "stock market today"]
    tweets_data = []
    
    for query in queries:
        try:
            response = client.search_recent_tweets(
                query=f"{query} -is:retweet lang:en",
                max_results=50,
                tweet_fields=["created_at", "text", "public_metrics"]
            )
            if response.data:
                for tweet in response.data:
                    tweets_data.append({
                        "timestamp": tweet.created_at.isoformat(),
                        "source": "twitter",
                        "title": "",
                        "text": tweet.text,
                        "score": tweet.public_metrics.get("like_count", 0)
                    })
        except Exception as e:
            print(f"Twitter error for query '{query}': {e}")
    
    df = pd.DataFrame(tweets_data)
    df.to_csv("data/raw/twitter_posts.csv", index=False)
    print(f"Twitter: saved {len(df)} tweets")
    return df

if __name__ == "__main__":
    fetch_tweets()