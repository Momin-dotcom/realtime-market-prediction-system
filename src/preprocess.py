import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import os
import pickle

def load_data(filepath='data/raw/sample_market_data.csv'):
    df = pd.read_csv(filepath)
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date').reset_index(drop=True)
    print(f"Data loaded: {df.shape}")
    return df

def create_features(df):
    # Use sentiment score + price change as features
    df['sentiment_numeric'] = df['sentiment_score']
    
    # Rolling averages (3-day and 7-day)
    df['sentiment_3day'] = df['sentiment_numeric'].rolling(3).mean().fillna(0)
    df['sentiment_7day'] = df['sentiment_numeric'].rolling(7).mean().fillna(0)
    df['price_change_3day'] = df['price_change'].rolling(3).mean().fillna(0)
    
    df = df.dropna().reset_index(drop=True)
    print(f"Features created. Shape: {df.shape}")
    return df

def create_sequences(df, sequence_length=10):
    """
    Sliding window: use past 10 days to predict next day
    """
    feature_cols = ['sentiment_numeric', 'sentiment_3day', 'sentiment_7day', 
                    'price_change', 'price_change_3day', 'volume']
    target_col = 'market_direction'
    
    scaler = MinMaxScaler()
    features_scaled = scaler.fit_transform(df[feature_cols])
    
    X, y = [], []
    for i in range(sequence_length, len(df)):
        X.append(features_scaled[i-sequence_length:i])
        y.append(df[target_col].iloc[i])
    
    X = np.array(X)
    y = np.array(y)
    print(f"Sequences created — X shape: {X.shape}, y shape: {y.shape}")
    return X, y, scaler

def split_data(X, y, train_ratio=0.7, val_ratio=0.15):
    n = len(X)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))
    
    X_train, y_train = X[:train_end], y[:train_end]
    X_val, y_val = X[train_end:val_end], y[train_end:val_end]
    X_test, y_test = X[val_end:], y[val_end:]
    
    print(f"Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")
    return X_train, y_train, X_val, y_val, X_test, y_test

def save_processed_data(X_train, y_train, X_val, y_val, X_test, y_test, scaler):
    os.makedirs('data/processed', exist_ok=True)
    np.save('data/processed/X_train.npy', X_train)
    np.save('data/processed/y_train.npy', y_train)
    np.save('data/processed/X_val.npy', X_val)
    np.save('data/processed/y_val.npy', y_val)
    np.save('data/processed/X_test.npy', X_test)
    np.save('data/processed/y_test.npy', y_test)
    
    with open('data/processed/scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    print("All processed data saved!")

if __name__ == "__main__":
    df = load_data()
    df = create_features(df)
    X, y, scaler = create_sequences(df, sequence_length=10)
    X_train, y_train, X_val, y_val, X_test, y_test = split_data(X, y)
    save_processed_data(X_train, y_train, X_val, y_val, X_test, y_test, scaler)