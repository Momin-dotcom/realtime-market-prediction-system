import pandas as pd
import numpy as np
import os

# Create sample data that mimics what Member 1 will provide
np.random.seed(42)
dates = pd.date_range(start='2023-01-01', end='2024-01-01', freq='D')

df = pd.DataFrame({
    'date': dates,
    'sentiment_score': np.random.choice([-1, 0, 1], size=len(dates)),  # -1=negative, 0=neutral, 1=positive
    'sentiment_label': np.random.choice(['negative', 'neutral', 'positive'], size=len(dates)),
    'price_change': np.random.randn(len(dates)),
    'volume': np.random.randint(1000000, 5000000, size=len(dates)),
    'market_direction': np.random.choice([0, 1], size=len(dates))  # 0=down, 1=up
})

os.makedirs('data/raw', exist_ok=True)
df.to_csv('data/raw/sample_market_data.csv', index=False)
print("Sample data created!")
print(df.head())
print(f"Shape: {df.shape}")