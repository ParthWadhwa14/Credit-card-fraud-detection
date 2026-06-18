from pathlib import Path

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from app.main import app, FEATURE_COLUMNS


client = TestClient(app)

PRIMARY_X_TEST_PATH = Path("data/processed/X_test.csv")
FIXTURE_X_TEST_PATH = Path("tests/fixtures/X_test_sample.csv")


def get_valid_features():
    """
    Load one valid processed test sample.

    Priority:
    1. data/processed/X_test.csv for local testing
    2. tests/fixtures/X_test_sample.csv for GitHub Actions CI
    """
    if PRIMARY_X_TEST_PATH.exists():
        path = PRIMARY_X_TEST_PATH
    elif FIXTURE_X_TEST_PATH.exists():
        path = FIXTURE_X_TEST_PATH
    else:
        pytest.skip("No processed test sample found.")

    X_test = pd.read_csv(path)

    missing_columns = [col for col in FEATURE_COLUMNS if col not in X_test.columns]

    if missing_columns:
        pytest.skip(f"{path} missing required columns: {missing_columns}")

    return X_test.iloc[0][FEATURE_COLUMNS].astype(float).tolist()


def test_health_endpoint():
    response = client.get("/health")

    assert response.status_code == 200

    data = response.json()
    assert isinstance(data, dict)


def test_model_info_endpoint():
    response = client.get("/model-info")

    assert response.status_code == 200

    data = response.json()
    assert isinstance(data, dict)


def test_predict_valid_features():
    features = get_valid_features()

    response = client.post(
        "/predict",
        json={"features": features},
    )

    assert response.status_code == 200

    data = response.json()

    assert "fraud_probability" in data
    assert "prediction" in data
    assert "risk_level" in data
    assert "threshold" in data

    assert isinstance(data["fraud_probability"], float)
    assert data["prediction"] in [0, 1]
    assert isinstance(data["risk_level"], str)
    assert isinstance(data["threshold"], float)


def test_predict_invalid_feature_length():
    wrong_features = [0.0] * 30

    response = client.post(
        "/predict",
        json={"features": wrong_features},
    )

    assert response.status_code in [400, 422]


def test_explain_valid_features():
    features = get_valid_features()

    response = client.post(
        "/explain",
        json={"features": features},
    )

    assert response.status_code == 200

    data = response.json()

    assert "fraud_probability" in data
    assert "prediction" in data
    assert "risk_level" in data
    assert "threshold" in data
    assert "base_value" in data
    assert "top_features" in data

    assert isinstance(data["top_features"], list)
    assert len(data["top_features"]) > 0

    first_feature = data["top_features"][0]

    assert "feature" in first_feature
    assert "value" in first_feature
    assert "shap_value" in first_feature
    assert "impact" in first_feature


def test_metrics_endpoint():
    response = client.get("/metrics")

    assert response.status_code == 200

    text = response.text

    assert "# HELP" in text or "# TYPE" in text