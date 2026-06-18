from pathlib import Path
import json
import os

import joblib
import mlflow
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    classification_report,
)


DATA_DIR = Path("data/processed")
MODELS_DIR = Path("models")
REPORTS_DIR = Path("reports")
OPTUNA_DIR = REPORTS_DIR / "optuna"

CHAMPION_MODEL_NAME = "xgboost_weighted"
CHAMPION_MODEL_PATH = MODELS_DIR / "tuned_xgboost_weighted.pkl"
OPTUNA_TRIALS_PATH = OPTUNA_DIR / "xgboost_weighted_optuna_trials.csv"

FALSE_NEGATIVE_COST = 5000
FALSE_POSITIVE_COST = 100

REPORTS_DIR.mkdir(parents=True, exist_ok=True)


def load_test_data():
    X_test = pd.read_csv(DATA_DIR / "X_test.csv")
    y_test = pd.read_csv(DATA_DIR / "y_test.csv")["Class"]
    return X_test, y_test


def load_best_xgboost_trial():
    """
    Load best XGBoost trial from Optuna CSV.
    We select by objective_pr_auc because that was our tuning objective.
    """
    if not OPTUNA_TRIALS_PATH.exists():
        raise FileNotFoundError(
            f"Optuna trials file not found: {OPTUNA_TRIALS_PATH}"
        )

    df = pd.read_csv(OPTUNA_TRIALS_PATH)

    best_trial = (
        df.sort_values("objective_pr_auc", ascending=False)
        .iloc[0]
        .to_dict()
    )

    return best_trial


def choose_deployment_threshold(best_trial):
    """
    Prefer best cost threshold if available.
    Otherwise fall back to best F1 threshold.
    Otherwise use 0.5.
    """
    if "best_cost_threshold" in best_trial and pd.notna(best_trial["best_cost_threshold"]):
        return float(best_trial["best_cost_threshold"]), "best_cost_threshold"

    if "best_f1_threshold" in best_trial and pd.notna(best_trial["best_f1_threshold"]):
        return float(best_trial["best_f1_threshold"]), "best_f1_threshold"

    return 0.5, "default_0.5"


def evaluate_at_threshold(y_true, y_score, threshold):
    y_pred = (y_score >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    metrics = {
        "threshold": float(threshold),
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1_score": f1_score(y_true, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, y_score),
        "pr_auc": average_precision_score(y_true, y_score),
        "true_negatives": int(tn),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_positives": int(tp),
    }

    metrics["business_cost"] = (
        metrics["false_negatives"] * FALSE_NEGATIVE_COST
        + metrics["false_positives"] * FALSE_POSITIVE_COST
    )

    return metrics, y_pred


def save_json(data, path):
    with open(path, "w") as f:
        json.dump(data, f, indent=4)


def main():
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment("fraud-detection-final-xgboost")

    if not CHAMPION_MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Champion model not found at {CHAMPION_MODEL_PATH}. "
            "Make sure tuned_xgboost_weighted.pkl exists."
        )

    print("Loading best XGBoost trial...")
    best_trial = load_best_xgboost_trial()

    deployment_threshold, threshold_source = choose_deployment_threshold(best_trial)

    print("\nSelected champion model:", CHAMPION_MODEL_NAME)
    print("Best trial number:", best_trial.get("trial_number"))
    print("Validation PR-AUC:", best_trial.get("objective_pr_auc"))
    print("Deployment threshold:", deployment_threshold)
    print("Threshold source:", threshold_source)

    print("\nLoading champion model...")
    model = joblib.load(CHAMPION_MODEL_PATH)

    print("Loading untouched test data...")
    X_test, y_test = load_test_data()

    print("Generating test probabilities...")
    y_score = model.predict_proba(X_test)[:, 1]

    print("Evaluating on test set...")
    test_metrics, y_pred = evaluate_at_threshold(
        y_true=y_test,
        y_score=y_score,
        threshold=deployment_threshold,
    )

    print("\nFinal Test Metrics:")
    print(json.dumps(test_metrics, indent=4))

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, zero_division=0))

    champion_config = {
        "champion_model_name": CHAMPION_MODEL_NAME,
        "champion_model_path": str(CHAMPION_MODEL_PATH),
        "selected_trial_number": best_trial.get("trial_number"),
        "validation_objective_pr_auc": best_trial.get("objective_pr_auc"),
        "deployment_threshold": deployment_threshold,
        "threshold_source": threshold_source,
        "false_negative_cost": FALSE_NEGATIVE_COST,
        "false_positive_cost": FALSE_POSITIVE_COST,
        "selection_reason": (
            "Selected weighted XGBoost as the production-focused fraud detection model "
            "because it maintained strong PR-AUC while minimizing false positives, "
            "reducing customer friction from unnecessary fraud alerts."
        ),
    }

    metrics_path = REPORTS_DIR / "xgboost_champion_test_metrics.json"
    config_path = REPORTS_DIR / "xgboost_champion_config.json"
    predictions_path = REPORTS_DIR / "xgboost_champion_test_predictions.csv"

    save_json(test_metrics, metrics_path)
    save_json(champion_config, config_path)

    predictions_df = pd.DataFrame({
        "y_true": y_test,
        "fraud_probability": y_score,
        "prediction": y_pred,
    })

    predictions_df.to_csv(predictions_path, index=False)

    with mlflow.start_run(run_name="FINAL_XGBOOST_TEST_EVALUATION"):
        mlflow.log_param("champion_model", CHAMPION_MODEL_NAME)
        mlflow.log_param("model_path", str(CHAMPION_MODEL_PATH))
        mlflow.log_param("selected_trial_number", best_trial.get("trial_number"))
        mlflow.log_param("threshold_source", threshold_source)
        mlflow.log_param("deployment_threshold", deployment_threshold)
        mlflow.log_param("false_negative_cost", FALSE_NEGATIVE_COST)
        mlflow.log_param("false_positive_cost", FALSE_POSITIVE_COST)

        for key, value in test_metrics.items():
            mlflow.log_metric(f"test_{key}", value)

        mlflow.log_artifact(str(metrics_path), artifact_path="final_reports")
        mlflow.log_artifact(str(config_path), artifact_path="final_reports")
        mlflow.log_artifact(str(predictions_path), artifact_path="final_reports")
        mlflow.log_artifact(str(CHAMPION_MODEL_PATH), artifact_path="champion_model_pickle")

    print("\nSaved:")
    print(metrics_path)
    print(config_path)
    print(predictions_path)

    print("\nFinal XGBoost champion evaluation completed.")


if __name__ == "__main__":
    main()