from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import numpy as np
from tensorflow import keras
import os

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

class PredictRequest(BaseModel):
    features: list[list[float]]  # 10 timesteps x 15 features

@app.post("/predict")
def predict(req: PredictRequest):
    # Input prepare karo - shape (1, 10, 15)
    x = np.array(req.features).reshape(1, 10, 15)
    
    # Teeno models se prediction lo
    lstm_pred = float(lstm_model.predict(x)[0][0])
    gru_pred = float(gru_model.predict(x)[0][0])
    rnn_pred = float(rnn_model.predict(x)[0][0])
    
    # Average nikalo
    avg_pred = (lstm_pred + gru_pred + rnn_pred) / 3
    
    return {
        "direction": "up" if avg_pred > 0.5 else "down",
        "confidence": round(avg_pred, 4),
        "lstm_prediction": round(lstm_pred, 4),
        "gru_prediction": round(gru_pred, 4),
        "rnn_prediction": round(rnn_pred, 4),
    }

@app.get("/")
def root():
    return {"status": "Market Prediction API is running"}