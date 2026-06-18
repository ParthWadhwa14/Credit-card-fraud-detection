from pathlib import Path
import json
from typing import List

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field


MODEL_PATH = Path("models/tuned_xgboost_weighted.pkl")
CONFIG_PATH = Path("reports/xgboost_champion_config.json")


FEATURE_COLUMNS = [
    "Time",
    "V1", "V2", "V3", "V4", "V5", "V6", "V7",
    "V8", "V9", "V10", "V11", "V12", "V13", "V14",
    "V15", "V16", "V17", "V18", "V19", "V20", "V21",
    "V22", "V23", "V24", "V25", "V26", "V27", "V28",
    "Amount",
    "Hour",
    "Day",
]


app = FastAPI(
    title="Fraud Detection API",
    description="Production-focused XGBoost fraud detection API",
    version="1.0.0",
)


class TransactionRequest(BaseModel):
    features: List[float] = Field(
        ...,
        description="Transaction features in the exact training order."
    )


class PredictionResponse(BaseModel):
    fraud_probability: float
    prediction: int
    risk_level: str
    threshold: float


def load_config():
    if not CONFIG_PATH.exists():
        raise FileNotFoundError(f"Config not found: {CONFIG_PATH}")

    with open(CONFIG_PATH, "r") as f:
        return json.load(f)


def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

    return joblib.load(MODEL_PATH)


model = load_model()
config = load_config()
DEPLOYMENT_THRESHOLD = float(config["deployment_threshold"])


@app.get("/")
def root():
    return {
        "message": "Fraud Detection API is running",
        "model": "xgboost_weighted",
        "threshold": DEPLOYMENT_THRESHOLD,
    }


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "threshold_loaded": DEPLOYMENT_THRESHOLD is not None,
    }


@app.get("/model-info")
def model_info():
    return {
        "model_name": config.get("champion_model_name"),
        "selected_trial_number": config.get("selected_trial_number"),
        "validation_pr_auc": config.get("validation_objective_pr_auc"),
        "deployment_threshold": DEPLOYMENT_THRESHOLD,
        "threshold_source": config.get("threshold_source"),
        "selection_reason": config.get("selection_reason"),
    }


@app.post("/predict", response_model=PredictionResponse)
def predict(request: TransactionRequest):
    if len(request.features) != len(FEATURE_COLUMNS):
        raise HTTPException(
            status_code=400,
            detail=f"Expected {len(FEATURE_COLUMNS)} features, got {len(request.features)}."
        )

    X = pd.DataFrame([request.features], columns=FEATURE_COLUMNS)

    fraud_probability = float(model.predict_proba(X)[:, 1][0])
    prediction = int(fraud_probability >= DEPLOYMENT_THRESHOLD)

    if fraud_probability >= 0.80:
        risk_level = "high"
    elif fraud_probability >= DEPLOYMENT_THRESHOLD:
        risk_level = "medium"
    else:
        risk_level = "low"

    return PredictionResponse(
        fraud_probability=fraud_probability,
        prediction=prediction,
        risk_level=risk_level,
        threshold=DEPLOYMENT_THRESHOLD,
    )


@app.post("/predict-batch")
def predict_batch(requests: List[TransactionRequest]):
    rows = []

    for request in requests:
        if len(request.features) != len(FEATURE_COLUMNS):
            raise HTTPException(
                status_code=400,
                detail=f"Expected {len(FEATURE_COLUMNS)} features, got {len(request.features)}."
            )

        rows.append(request.features)

    X = pd.DataFrame(rows, columns=FEATURE_COLUMNS)

    probabilities = model.predict_proba(X)[:, 1]
    predictions = (probabilities >= DEPLOYMENT_THRESHOLD).astype(int)

    results = []

    for prob, pred in zip(probabilities, predictions):
        if prob >= 0.80:
            risk_level = "high"
        elif prob >= DEPLOYMENT_THRESHOLD:
            risk_level = "medium"
        else:
            risk_level = "low"

        results.append({
            "fraud_probability": float(prob),
            "prediction": int(pred),
            "risk_level": risk_level,
            "threshold": DEPLOYMENT_THRESHOLD,
        })

    return {
        "count": len(results),
        "results": results,
    }