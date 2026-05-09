from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class PredictRequest(BaseModel):
    text: str

@app.post("/predict")
def predict(req: PredictRequest):
    return {
        "direction": "up",
        "confidence": 0.72,
        "sentiment": "positive"
    }

@app.get("/")
def root():
    return {"status": "Market Prediction API is running"}