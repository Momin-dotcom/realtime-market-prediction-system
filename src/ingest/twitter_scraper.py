import pandas as pd
from datetime import datetime

def fetch_tweets():

    tweets_data = [
        {
            "timestamp": datetime.now().isoformat(),
            "source": "twitter",
            "text": "Stock market is bullish today after strong earnings reports 🚀"
        },
        {
            "timestamp": datetime.now().isoformat(),
            "source": "twitter",
            "text": "Bitcoin is crashing again as investors panic sell 😬"
        },
        {
            "timestamp": datetime.now().isoformat(),
            "source": "twitter",
            "text": "Tech stocks are showing strong recovery this week"
        },
        {
            "timestamp": datetime.now().isoformat(),
            "source": "twitter",
            "text": "Inflation fears continue to pressure global markets"
        },
        {
            "timestamp": datetime.now().isoformat(),
            "source": "twitter",
            "text": "Tesla earnings beat expectations and shares jump higher"
        },
        {
            "timestamp": datetime.now().isoformat(),
            "source": "twitter",
            "text": "Oil prices are falling sharply due to weak demand outlook"
        },
        {
            "timestamp": datetime.now().isoformat(),
            "source": "twitter",
            "text": "NASDAQ closes higher for the third straight session"
        },
        {
            "timestamp": datetime.now().isoformat(),
            "source": "twitter",
            "text": "Crypto market volatility continues as traders remain cautious"
        },
        {
            "timestamp": datetime.now().isoformat(),
            "source": "twitter",
            "text": "Gold prices rise amid uncertainty in financial markets"
        },
        {
            "timestamp": datetime.now().isoformat(),
            "source": "twitter",
            "text": "Investors are shifting money into safer assets"
        },
        {
            "timestamp": datetime.now().isoformat(),
            "source": "twitter",
            "text": "AI-related companies continue dominating Wall Street discussions"
        },
        {
            "timestamp": datetime.now().isoformat(),
            "source": "twitter",
            "text": "Banking sector stocks fall after weak quarterly guidance"
        },
        {
            "timestamp": datetime.now().isoformat(),
            "source": "twitter",
            "text": "Federal Reserve interest rate decision expected tomorrow"
        },
        {
            "timestamp": datetime.now().isoformat(),
            "source": "twitter",
            "text": "Retail investors are buying the dip aggressively today"
        },
        {
            "timestamp": datetime.now().isoformat(),
            "source": "twitter",
            "text": "Energy stocks rebound as crude oil stabilizes"
        },
        {
            "timestamp": datetime.now().isoformat(),
            "source": "twitter",
            "text": "Market sentiment remains neutral ahead of economic reports"
        },
        {
            "timestamp": datetime.now().isoformat(),
            "source": "twitter",
            "text": "Apple stock hits another all-time high 📈"
        },
        {
            "timestamp": datetime.now().isoformat(),
            "source": "twitter",
            "text": "Ethereum traders expect another breakout soon"
        },
        {
            "timestamp": datetime.now().isoformat(),
            "source": "twitter",
            "text": "Concerns about recession are increasing among analysts"
        },
        {
            "timestamp": datetime.now().isoformat(),
            "source": "twitter",
            "text": "Global markets recover after positive economic data release"
        }
    ]

    df = pd.DataFrame(tweets_data)

    df.to_csv("data/raw/twitter_posts.csv", index=False)

    print(f"Twitter: saved {len(df)} tweets")

    return df


if __name__ == "__main__":
    fetch_tweets()