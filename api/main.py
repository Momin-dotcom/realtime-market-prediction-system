from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import numpy as np
import keras
import os
import yfinance as yf
import pandas as pd

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Models load karo
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, "src", "modelss")

lstm_model = keras.models.load_model(os.path.join(MODELS_DIR, "best_lstm.keras"))
gru_model = keras.models.load_model(os.path.join(MODELS_DIR, "best_gru.keras"))
rnn_model = keras.models.load_model(os.path.join(MODELS_DIR, "best_rnn.keras"))

class StockRequest(BaseModel):
    symbol: str  # e.g. "AAPL", "GOOGL"

def get_stock_features(symbol: str):
    df = yf.download(symbol, period="30d", interval="1d", progress=False, auto_adjust=True)
    
    if df is None or len(df) < 10:
        raise HTTPException(status_code=400, detail=f"Not enough data for {symbol}")
    
    # Multi-level columns fix
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    
    df = df.tail(10).copy()
    
    def norm(series):
        mn, mx = series.min(), series.max()
        if mx == mn:
            return pd.Series(np.zeros(len(series)), index=series.index)
        return (series - mn) / (mx - mn)
    
    features = np.column_stack([
        norm(df['Open']).values,
        norm(df['High']).values,
        norm(df['Low']).values,
        norm(df['Close']).values,
        norm(df['Volume']).values,
        norm(df['Volume']).values,
        norm(df['High'] - df['Low']).values,
        norm(df['Close'] - df['Open']).values,
        norm(df['Close'].rolling(3).mean().bfill()).values,
        norm(df['Close'].rolling(5).mean().bfill()).values,
        norm(df['Close'].rolling(7).mean().bfill()).values,
        norm(df['Close'].diff().fillna(0)).values,
        norm(df['Volume'].diff().fillna(0)).values,
        norm(df['High'] - df['Close']).values,
        np.ones(10),
    ])
    
    return features.reshape(1, 10, 15).astype(np.float32)
    
    features = np.column_stack([
        norm(df['Open']).values,       # 0
        norm(df['High']).values,       # 1
        norm(df['Low']).values,        # 2
        norm(df['Close']).values,      # 3
        norm(df['Volume']).values,     # 4
        norm(df['Volume']).values,     # 5
        norm(df['High'] - df['Low']).values,  # 6 - range
        norm(df['Close'] - df['Open']).values, # 7 - candle body
        norm(df['Close'].rolling(3).mean().fillna(method='bfill')).values,  # 8 - MA3
        norm(df['Close'].rolling(5).mean().fillna(method='bfill')).values,  # 9 - MA5
        norm(df['Close'].rolling(7).mean().fillna(method='bfill')).values,  # 10 - MA7
        norm(df['Close'].diff().fillna(0)).values,  # 11 - momentum
        norm(df['Volume'].diff().fillna(0)).values, # 12 - volume change
        norm(df['High'] - df['Close']).values,      # 13 - upper shadow
        np.ones(10),                                 # 14 - bias
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