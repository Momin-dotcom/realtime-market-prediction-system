# src/ingest/reddit_scraper.py
import praw
import pandas as pd
from datetime import datetime

def fetch_reddit_posts():
    reddit = praw.Reddit(
        client_id="YOUR_CLIENT_ID",
        client_secret="YOUR_CLIENT_SECRET",
        user_agent="market_sentiment_bot/1.0"
    )
    
    subreddits = ["wallstreetbets", "investing", "stocks", "finance"]
    posts = []
    
    for sub_name in subreddits:
        subreddit = reddit.subreddit(sub_name)
        for post in subreddit.hot(limit=50):  # top 50 hot posts
            posts.append({
                "timestamp": datetime.utcfromtimestamp(post.created_utc).isoformat(),
                "source": f"reddit_{sub_name}",
                "title": post.title,
                "text": post.title + " " + (post.selftext[:500] if post.selftext else ""),
                "score": post.score,
                "num_comments": post.num_comments
            })
    
    df = pd.DataFrame(posts)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df.drop_duplicates(subset=["title"], inplace=True)
    
    df.to_csv("data/raw/reddit_posts.csv", index=False)
    print(f"Reddit: saved {len(df)} posts")
    return df

if __name__ == "__main__":
    fetch_reddit_posts()