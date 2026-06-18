import os
import mlflow


EXPERIMENT_NAME = "fraud-detection-full-model-zoo"


def setup_mlflow():
    """
    Central MLflow setup.

    Uses sqlite backend so MLflow UI and training scripts read from the same store.
    """
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(EXPERIMENT_NAME)

    print(f"MLflow tracking URI: {tracking_uri}")
    print(f"MLflow experiment: {EXPERIMENT_NAME}")