from pathlib import Path
import json
import os

import joblib
import mlflow
import mlflow.xgboost
import pandas as pd
from mlflow.models import infer_signature


DATA_DIR = Path("data/processed")
MODELS_DIR = Path("models")
REPORTS_DIR = Path("reports")

CHAMPION_MODEL_NAME = "xgboost_weighted"
REGISTERED_MODEL_NAME = "fraud_xgboost_champion"

MODEL_PATH = MODELS_DIR / "tuned_xgboost_weighted.pkl"
CONFIG_PATH = REPORTS_DIR / "xgboost_champion_config.json"
TEST_METRICS_PATH = REPORTS_DIR / "xgboost_champion_test_metrics.json"


def load_json(path):
    if not path.exists():
        raise FileNotFoundError(f"Required file not found: {path}")

    with open(path, "r") as f:
        return json.load(f)


def main():
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment("fraud-detection-model-registry")

    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")

    print("Loading model...")
    model = joblib.load(MODEL_PATH)

    print("Loading champion config and test metrics...")
    champion_config = load_json(CONFIG_PATH)
    test_metrics = load_json(TEST_METRICS_PATH)

    X_test = pd.read_csv(DATA_DIR / "X_test.csv")
    input_example = X_test.head(5)

    prediction_example = model.predict_proba(input_example)[:, 1]
    signature = infer_signature(input_example, prediction_example)

    with mlflow.start_run(run_name="REGISTER_XGBOOST_CHAMPION") as run:
        mlflow.log_param("champion_model_name", CHAMPION_MODEL_NAME)
        mlflow.log_param("registered_model_name", REGISTERED_MODEL_NAME)
        mlflow.log_param("deployment_threshold", champion_config["deployment_threshold"])
        mlflow.log_param("threshold_source", champion_config["threshold_source"])
        mlflow.log_param("selection_reason", champion_config["selection_reason"])

        for key, value in test_metrics.items():
            if isinstance(value, (int, float)):
                mlflow.log_metric(f"final_test_{key}", value)

        mlflow.log_artifact(str(CONFIG_PATH), artifact_path="champion_config")
        mlflow.log_artifact(str(TEST_METRICS_PATH), artifact_path="champion_metrics")
        mlflow.log_artifact(str(MODEL_PATH), artifact_path="champion_pickle")

        mlflow.xgboost.log_model(
            xgb_model=model,
            name="model",
            signature=signature,
            input_example=input_example,
            registered_model_name=REGISTERED_MODEL_NAME,
        )

        print("\nChampion model registered successfully.")
        print("Run ID:", run.info.run_id)
        print("Registered model name:", REGISTERED_MODEL_NAME)


if __name__ == "__main__":
    main()