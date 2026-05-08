import pandas as pd
from datetime import datetime

def fetch_tweets():

    tweets_data = [
        {"timestamp": datetime.now().isoformat(), "source": "twitter", "text": "Stock market is bullish today 🚀", "score": 0},
        {"timestamp": datetime.now().isoformat(), "source": "twitter", "text": "Bitcoin is crashing again 😬", "score": 0},
        {"timestamp": datetime.now().isoformat(), "source": "twitter", "text": "Tech stocks showing strong recovery", "score": 0},
        {"timestamp": datetime.now().isoformat(), "source": "twitter", "text": "Inflation fears still affecting markets", "score": 0},
        {"timestamp": datetime.now().isoformat(), "source": "twitter", "text": "Tesla earnings beat expectations", "score": 0},
        {"timestamp": datetime.now().isoformat(), "source": "twitter", "text": "Oil prices dropping sharply today", "score": 0},
        {"timestamp": datetime.now().isoformat(), "source": "twitter", "text": "NASDAQ showing upward trend", "score": 0},
        {"timestamp": datetime.now().isoformat(), "source": "twitter", "text": "Crypto volatility continues in market", "score": 0},
        {"timestamp": datetime.now().isoformat(), "source": "twitter", "text": "Gold prices rise amid uncertainty", "score": 0},
        {"timestamp": datetime.now().isoformat(), "source": "twitter", "text": "Investors shifting to safe assets", "score": 0},
    ]

    df = pd.DataFrame(tweets_data)
    df.to_csv("data/raw/twitter_posts.csv", index=False)

    print(f"Twitter: saved {len(df)} fallback tweets")
    return df


if __name__ == "__main__":
    fetch_tweets()