from pathlib import Path
import json
from typing import List, Dict

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from prometheus_fastapi_instrumentator import Instrumentator
from prometheus_client import Counter, Histogram


# ---------------------------------------------------------
class FeatureExplanation(BaseModel):
    feature: str
    value: float
    shap_value: float
    impact: str


class ExplanationResponse(BaseModel):
    fraud_probability: float
    prediction: int
    risk_level: str
    threshold: float
    base_value: float
    top_features: List[FeatureExplanation]
# ---------------------------------------------------------

MODEL_PATH = Path("models/tuned_xgboost_weighted.pkl")
CONFIG_PATH = Path("reports/xgboost_champion_config.json")


# ---------------------------------------------------------
# Feature order used during training
# ---------------------------------------------------------

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


# ---------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------

app = FastAPI(
    title="Fraud Detection API",
    description="Production-focused XGBoost fraud detection API with Prometheus monitoring",
    version="1.0.0",
)


# ---------------------------------------------------------
# Prometheus default HTTP metrics
# ---------------------------------------------------------

Instrumentator().instrument(app).expose(app)


# ---------------------------------------------------------
# Custom ML monitoring metrics
# ---------------------------------------------------------

PREDICTION_COUNTER = Counter(
    "fraud_predictions_total",
    "Total number of fraud prediction requests"
)

FRAUD_FLAG_COUNTER = Counter(
    "fraud_flags_total",
    "Total number of transactions flagged as fraud"
)

PREDICTION_CONFIDENCE = Histogram(
    "fraud_prediction_probability",
    "Distribution of predicted fraud probabilities",
    buckets=[0.01, 0.05, 0.10, 0.20, 0.40, 0.60, 0.80, 0.90, 0.95, 0.99]
)


# ---------------------------------------------------------
# Request and response schemas
# ---------------------------------------------------------

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


class FeatureExplanation(BaseModel):
    feature: str
    value: float
    shap_value: float
    impact: str


class ExplanationResponse(BaseModel):
    fraud_probability: float
    prediction: int
    risk_level: str
    threshold: float
    base_value: float
    top_features: List[FeatureExplanation]
# ---------------------------------------------------------
# Load model and config
# ---------------------------------------------------------

def load_config():
    if not CONFIG_PATH.exists():
        raise FileNotFoundError(f"Config not found: {CONFIG_PATH}")

    with open(CONFIG_PATH, "r") as f:
        return json.load(f)


def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

    return joblib.load(MODEL_PATH)


def coerce_base_value(base_value):
    if isinstance(base_value, list):
        return float(base_value[1])

    try:
        return float(base_value)
    except TypeError:
        return float(base_value[0])


def get_tree_explainer():
    global explainer

    if explainer is None:
        import shap

        explainer = shap.TreeExplainer(model)

    return explainer


def get_shap_explanation_values(X: pd.DataFrame):
    tree_explainer = get_tree_explainer()
    shap_values = tree_explainer.shap_values(X)

    # SHAP output can differ by version/model type.
    # For binary classification, sometimes it returns a list [class_0, class_1].
    if isinstance(shap_values, list):
        shap_values_for_fraud = shap_values[1][0]
    else:
        shap_values_for_fraud = shap_values[0]

    return shap_values_for_fraud, coerce_base_value(tree_explainer.expected_value)


def get_xgboost_contribution_values(X: pd.DataFrame):
    import xgboost as xgb

    booster = model.get_booster()
    contributions = booster.predict(xgb.DMatrix(X), pred_contribs=True)
    row_contributions = contributions[0]

    return row_contributions[:-1], float(row_contributions[-1])


def explain_transaction(features: List[float], top_n: int = 8) -> Dict:
    validate_feature_length(features)

    X = pd.DataFrame([features], columns=FEATURE_COLUMNS)

    fraud_probability = float(model.predict_proba(X)[:, 1][0])
    prediction = int(fraud_probability >= DEPLOYMENT_THRESHOLD)
    risk_level = get_risk_level(fraud_probability, prediction)

    try:
        shap_values_for_fraud, base_value = get_shap_explanation_values(X)
    except Exception:
        shap_values_for_fraud, base_value = get_xgboost_contribution_values(X)

    explanation_rows = []

    for feature, value, shap_value in zip(
        FEATURE_COLUMNS,
        X.iloc[0].tolist(),
        shap_values_for_fraud,
    ):
        explanation_rows.append({
            "feature": feature,
            "value": float(value),
            "shap_value": float(shap_value),
            "impact": "increases fraud risk" if shap_value > 0 else "decreases fraud risk",
            "abs_shap_value": abs(float(shap_value)),
        })

    explanation_rows = sorted(
        explanation_rows,
        key=lambda x: x["abs_shap_value"],
        reverse=True,
    )[:top_n]

    for row in explanation_rows:
        row.pop("abs_shap_value")

    return {
        "fraud_probability": fraud_probability,
        "prediction": prediction,
        "risk_level": risk_level,
        "threshold": DEPLOYMENT_THRESHOLD,
        "base_value": base_value,
        "top_features": explanation_rows,
    }

model = load_model()
config = load_config()
DEPLOYMENT_THRESHOLD = float(config["deployment_threshold"])
explainer = None


# ---------------------------------------------------------
# Helper functions
# ---------------------------------------------------------

def get_risk_level(fraud_probability: float, prediction: int) -> str:
    if fraud_probability >= 0.80:
        return "high"

    if prediction == 1:
        return "medium"

    return "low"


def validate_feature_length(features: List[float]):
    expected = len(FEATURE_COLUMNS)
    actual = len(features)

    if actual != expected:
        raise HTTPException(
            status_code=400,
            detail=f"Expected {expected} features, got {actual}."
        )


def update_prediction_metrics(fraud_probability: float, prediction: int):
    PREDICTION_COUNTER.inc()
    PREDICTION_CONFIDENCE.observe(fraud_probability)

    if prediction == 1:
        FRAUD_FLAG_COUNTER.inc()


# ---------------------------------------------------------
# API endpoints
# ---------------------------------------------------------

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
    validate_feature_length(request.features)

    X = pd.DataFrame([request.features], columns=FEATURE_COLUMNS)

    fraud_probability = float(model.predict_proba(X)[:, 1][0])
    prediction = int(fraud_probability >= DEPLOYMENT_THRESHOLD)
    risk_level = get_risk_level(fraud_probability, prediction)

    update_prediction_metrics(fraud_probability, prediction)

    return PredictionResponse(
        fraud_probability=fraud_probability,
        prediction=prediction,
        risk_level=risk_level,
        threshold=DEPLOYMENT_THRESHOLD,
    )

@app.post("/explain", response_model=ExplanationResponse)
def explain(request: TransactionRequest):
    try:
        explanation = explain_transaction(request.features, top_n=8)
        return explanation

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Explanation failed: {str(e)}"
        )

@app.post("/predict-batch")
def predict_batch(requests: List[TransactionRequest]):
    rows = []

    for request in requests:
        validate_feature_length(request.features)
        rows.append(request.features)

    X = pd.DataFrame(rows, columns=FEATURE_COLUMNS)

    probabilities = model.predict_proba(X)[:, 1]
    predictions = (probabilities >= DEPLOYMENT_THRESHOLD).astype(int)

    results = []

    for prob, pred in zip(probabilities, predictions):
        prob = float(prob)
        pred = int(pred)

        risk_level = get_risk_level(prob, pred)
        update_prediction_metrics(prob, pred)

        results.append({
            "fraud_probability": prob,
            "prediction": pred,
            "risk_level": risk_level,
            "threshold": DEPLOYMENT_THRESHOLD,
        })

    return {
        "count": len(results),
        "results": results,
    }
