import pandas as pd
from datetime import datetime

def fetch_reddit_posts():

    sample_posts = [
        {
            "timestamp": datetime.now().isoformat(),
            "source": "reddit",
            "title": "Stocks rally after strong earnings season",
            "text": "Investors are optimistic as major companies report better-than-expected earnings."
        },
        {
            "timestamp": datetime.now().isoformat(),
            "source": "reddit",
            "title": "Tech shares fall sharply today",
            "text": "Markets are reacting to concerns over interest rate hikes."
        },
        {
            "timestamp": datetime.now().isoformat(),
            "source": "reddit",
            "title": "Market stays neutral amid uncertainty",
            "text": "Analysts expect sideways movement in the short term."
        },
        {
            "timestamp": datetime.now().isoformat(),
            "source": "reddit",
            "title": "Bitcoin surges past resistance level",
            "text": "Crypto traders are seeing bullish momentum return."
        },
        {
            "timestamp": datetime.now().isoformat(),
            "source": "reddit",
            "title": "Oil prices drop on demand fears",
            "text": "Global slowdown concerns are affecting energy markets."
        },
        {
            "timestamp": datetime.now().isoformat(),
            "source": "reddit",
            "title": "NASDAQ shows strong recovery signs",
            "text": "Technology stocks are leading the rebound."
        },
        {
            "timestamp": datetime.now().isoformat(),
            "source": "reddit",
            "title": "Inflation worries still dominate markets",
            "text": "Investors remain cautious ahead of Fed announcements."
        },
        {
            "timestamp": datetime.now().isoformat(),
            "source": "reddit",
            "title": "Tesla stock jumps after delivery report",
            "text": "Positive production numbers boost investor confidence."
        },
        {
            "timestamp": datetime.now().isoformat(),
            "source": "reddit",
            "title": "Gold prices rise amid uncertainty",
            "text": "Safe-haven assets gain traction again."
        },
        {
            "timestamp": datetime.now().isoformat(),
            "source": "reddit",
            "title": "Banking sector under pressure",
            "text": "Concerns about liquidity affect financial stocks."
        },
        {
            "timestamp": datetime.now().isoformat(),
            "source": "reddit",
            "title": "AI stocks continue to dominate market",
            "text": "Investors are heavily betting on AI-driven companies."
        },
        {
            "timestamp": datetime.now().isoformat(),
            "source": "reddit",
            "title": "S&P 500 shows mixed performance",
            "text": "Market sentiment remains unclear across sectors."
        },
        {
            "timestamp": datetime.now().isoformat(),
            "source": "reddit",
            "title": "Retail stocks struggle this quarter",
            "text": "Consumer spending slowdown impacts earnings."
        },
        {
            "timestamp": datetime.now().isoformat(),
            "source": "reddit",
            "title": "Energy sector rebounds strongly",
            "text": "Oil demand recovery boosts energy companies."
        },
        {
            "timestamp": datetime.now().isoformat(),
            "source": "reddit",
            "title": "Investors shift toward safe assets",
            "text": "Gold and bonds see increased inflows."
        }
    ]

    df = pd.DataFrame(sample_posts)

    df.to_csv("data/raw/reddit_posts.csv", index=False)

    print(f"Reddit fallback dataset saved: {len(df)} posts")

    return df


if __name__ == "__main__":
    fetch_reddit_posts()