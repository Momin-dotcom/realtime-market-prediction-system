from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import numpy as np
import keras
import os
import yfinance as yf
import pandas as pd
import feedparser
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, "src", "modelss")

lstm_model = keras.models.load_model(os.path.join(MODELS_DIR, "best_lstm.keras"))
gru_model = keras.models.load_model(os.path.join(MODELS_DIR, "best_gru.keras"))
rnn_model = keras.models.load_model(os.path.join(MODELS_DIR, "best_rnn.keras"))

analyzer = SentimentIntensityAnalyzer()

class StockRequest(BaseModel):
    symbol: str

def get_sentiment(symbol: str) -> float:
    try:
        url = f"https://feeds.finance.yahoo.com/rss/2.0/headline?s={symbol}&region=US&lang=en-US"
        feed = feedparser.parse(url)

        if not feed.entries:
            return 0.5

        scores = []
        for entry in feed.entries[:10]:
            text = entry.get("summary", entry.get("title", ""))
            if text:
                score = analyzer.polarity_scores(text)['compound']
                scores.append(score)

        if not scores:
            return 0.5

        avg_score = float(np.mean(scores))
        return (avg_score + 1) / 2
    except:
        return 0.5

def get_stock_features(symbol: str):
    df = yf.download(symbol, period="60d", interval="1d", progress=False, auto_adjust=True)

    if df is None or len(df) < 30:
        raise HTTPException(status_code=400, detail=f"Not enough data for {symbol}")

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df = df.copy()
    df['price_change_pct'] = df['Close'].pct_change()
    df['MA_7'] = df['Close'].rolling(7).mean()
    df['MA_21'] = df['Close'].rolling(21).mean()

    delta = df['Close'].diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = -delta.clip(upper=0).rolling(14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))

    ema12 = df['Close'].ewm(span=12).mean()
    ema26 = df['Close'].ewm(span=26).mean()
    df['MACD'] = ema12 - ema26
    df['MACD_signal'] = df['MACD'].ewm(span=9).mean()

    df['BB_mid'] = df['Close'].rolling(20).mean()
    df['BB_upper'] = df['BB_mid'] + 2 * df['Close'].rolling(20).std()
    df['BB_lower'] = df['BB_mid'] - 2 * df['Close'].rolling(20).std()

    df['sentiment_ratio'] = get_sentiment(symbol)

    df = df.dropna().tail(10).copy()

    if len(df) < 10:
        raise HTTPException(status_code=400, detail="Not enough data after indicators")

    def scale(series):
        mn, mx = series.min(), series.max()
        if mx == mn:
            return pd.Series(np.zeros(len(series)), index=series.index)
        return (series - mn) / (mx - mn)

    features = np.column_stack([
        scale(df['Open']).values,
        scale(df['High']).values,
        scale(df['Low']).values,
        scale(df['Close']).values,
        scale(df['Volume']).values,
        scale(df['price_change_pct']).values,
        scale(df['MA_7']).values,
        scale(df['MA_21']).values,
        scale(df['RSI']).values,
        scale(df['MACD']).values,
        scale(df['MACD_signal']).values,
        scale(df['BB_mid']).values,
        scale(df['BB_upper']).values,
        scale(df['BB_lower']).values,
        df['sentiment_ratio'].values,
    ])

    return features.reshape(1, 10, 15).astype(np.float32)

@app.post("/predict")
def predict(req: StockRequest):
    try:
        x = get_stock_features(req.symbol.upper())
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    lstm_pred = float(lstm_model.predict(x, verbose=0)[0][0])
    gru_pred = float(gru_model.predict(x, verbose=0)[0][0])
    rnn_pred = float(rnn_model.predict(x, verbose=0)[0][0])
    avg_pred = (lstm_pred + gru_pred + rnn_pred) / 3

    return {
        "symbol": req.symbol.upper(),
        "direction": "UP 📈" if avg_pred > 0.5 else "DOWN 📉",
        "confidence": round(avg_pred * 100, 2),
        "lstm_prediction": round(lstm_pred, 4),
        "gru_prediction": round(gru_pred, 4),
        "rnn_prediction": round(rnn_pred, 4),
    }

@app.get("/")
def root():
    return {"status": "Market Prediction API is running"}