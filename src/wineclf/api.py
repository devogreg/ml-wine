# src/wineclf/api.py
from __future__ import annotations

import os
from typing import Any, Dict, Optional

import mlflow
import mlflow.sklearn
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel

MODEL_NAME = os.getenv("MODEL_NAME", "wineclf_rf")
MODEL_STAGE = os.getenv("MODEL_STAGE")
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI")

if MLFLOW_TRACKING_URI:
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

app = FastAPI(title="Wine Quality Classifier API")

class PredictRequest(BaseModel):
    features: Dict[str, float]


class PredictResponse(BaseModel):
    prediction: int
    probability: Optional[float]
    raw_output: Dict[str, Any]

def load_model():
    if MODEL_STAGE:
        model_uri = f"models:/{MODEL_NAME}/{MODEL_STAGE}"
    else:
        model_uri = f"models:/{MODEL_NAME}/latest"
    return mlflow.sklearn.load_model(model_uri)

MODEL = None

@app.on_event("startup")
def startup_event() -> None:
    global MODEL
    MODEL = load_model()

@app.get("/")
def root():
    return {"status": "ok", "model_name": MODEL_NAME, "stage": MODEL_STAGE}

@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest) -> PredictResponse:
    if MODEL is None:
        raise RuntimeError("Model is not loaded")

    df = pd.DataFrame([req.features])
    y_pred = MODEL.predict(df)[0]

    proba: Optional[float] = None
    if hasattr(MODEL, "predict_proba"):
        proba = float(MODEL.predict_proba(df)[0, 1])

    return PredictResponse(
        prediction=int(y_pred),
        probability=proba,
        raw_output={"input": req.features},
    )
