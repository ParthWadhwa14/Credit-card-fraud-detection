from pathlib import Path
import warnings
import os
import joblib
import mlflow
import mlflow.sklearn
import mlflow.xgboost
import mlflow.lightgbm

import pandas as pd
os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    RandomForestClassifier,
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    AdaBoostClassifier,
    HistGradientBoostingClassifier,
)
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
)

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier


warnings.filterwarnings("ignore")


PROCESSED_DATA_DIR = Path("data/processed")
MODELS_DIR = Path("models")
REPORTS_DIR = Path("reports")

MODELS_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)


def load_processed_data():
    X_train = pd.read_csv(PROCESSED_DATA_DIR / "X_train.csv")
    X_valid = pd.read_csv(PROCESSED_DATA_DIR / "X_valid.csv")

    y_train = pd.read_csv(PROCESSED_DATA_DIR / "y_train.csv")["Class"]
    y_valid = pd.read_csv(PROCESSED_DATA_DIR / "y_valid.csv")["Class"]

    return X_train, X_valid, y_train, y_valid


def get_probability_scores(model, X):
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]

    if hasattr(model, "decision_function"):
        scores = model.decision_function(X)
        return (scores - scores.min()) / (scores.max() - scores.min())

    raise ValueError("Model does not support predict_proba or decision_function.")


def evaluate_predictions(y_true, y_proba, threshold=0.5):
    y_pred = (y_proba >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1_score": f1_score(y_true, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, y_proba),
        "pr_auc": average_precision_score(y_true, y_proba),
        "true_negatives": int(tn),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_positives": int(tp),
    }

    return metrics


def get_models(y_train):
    num_negative = (y_train == 0).sum()
    num_positive = (y_train == 1).sum()
    scale_pos_weight = num_negative / num_positive

    models = {
        "dummy_most_frequent": {
            "model": DummyClassifier(strategy="most_frequent"),
            "imbalance_strategy": "none",
            "mlflow_flavor": "sklearn",
        },

        "logistic_regression_balanced": {
            "model": LogisticRegression(
                class_weight="balanced",
                max_iter=1000,
                random_state=42,
                n_jobs=-1,
            ),
            "imbalance_strategy": "class_weight_balanced",
            "mlflow_flavor": "sklearn",
        },

        "gaussian_naive_bayes": {
            "model": GaussianNB(),
            "imbalance_strategy": "none",
            "mlflow_flavor": "sklearn",
        },

        "decision_tree_balanced": {
            "model": DecisionTreeClassifier(
                class_weight="balanced",
                max_depth=8,
                random_state=42,
            ),
            "imbalance_strategy": "class_weight_balanced",
            "mlflow_flavor": "sklearn",
        },

        "random_forest_balanced": {
            "model": RandomForestClassifier(
                n_estimators=300,
                max_depth=12,
                class_weight="balanced",
                random_state=42,
                n_jobs=-1,
            ),
            "imbalance_strategy": "class_weight_balanced",
            "mlflow_flavor": "sklearn",
        },

        "extra_trees_balanced": {
            "model": ExtraTreesClassifier(
                n_estimators=300,
                max_depth=12,
                class_weight="balanced",
                random_state=42,
                n_jobs=-1,
            ),
            "imbalance_strategy": "class_weight_balanced",
            "mlflow_flavor": "sklearn",
        },

        "gradient_boosting": {
            "model": GradientBoostingClassifier(
                n_estimators=200,
                learning_rate=0.05,
                max_depth=3,
                random_state=42,
            ),
            "imbalance_strategy": "none_baseline",
            "mlflow_flavor": "sklearn",
        },

        "adaboost": {
            "model": AdaBoostClassifier(
                n_estimators=200,
                learning_rate=0.05,
                random_state=42,
            ),
            "imbalance_strategy": "none_baseline",
            "mlflow_flavor": "sklearn",
        },

        "hist_gradient_boosting": {
            "model": HistGradientBoostingClassifier(
                max_iter=300,
                learning_rate=0.05,
                max_leaf_nodes=31,
                random_state=42,
            ),
            "imbalance_strategy": "none_baseline",
            "mlflow_flavor": "sklearn",
        },

        "xgboost_weighted": {
            "model": XGBClassifier(
                n_estimators=300,
                max_depth=5,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                scale_pos_weight=scale_pos_weight,
                eval_metric="logloss",
                random_state=42,
                n_jobs=-1,
            ),
            "imbalance_strategy": "scale_pos_weight",
            "mlflow_flavor": "xgboost",
        },

        "lightgbm_balanced": {
            "model": LGBMClassifier(
                n_estimators=300,
                learning_rate=0.05,
                num_leaves=31,
                class_weight="balanced",
                random_state=42,
                n_jobs=-1,
                verbosity=-1,
            ),
            "imbalance_strategy": "class_weight_balanced",
            "mlflow_flavor": "lightgbm",
        },
    }

    return models


def log_model_to_mlflow(model, flavor):
    """
    Log model using correct MLflow flavor.
    """
    if flavor == "xgboost":
        mlflow.xgboost.log_model(model, name="model")

    elif flavor == "lightgbm":
        mlflow.lightgbm.log_model(model, name="model")

    elif flavor == "sklearn":
        mlflow.sklearn.log_model(model, name="model")

    else:
        raise ValueError(f"Unsupported MLflow flavor: {flavor}")


def run_single_experiment(
    model_name,
    model_config,
    X_train,
    y_train,
    X_valid,
    y_valid,
):
    model = model_config["model"]
    imbalance_strategy = model_config["imbalance_strategy"]
    mlflow_flavor = model_config["mlflow_flavor"]

    with mlflow.start_run(run_name=model_name):
        print(f"\nTraining {model_name}...")

        model.fit(X_train, y_train)

        y_proba = get_probability_scores(model, X_valid)
        metrics = evaluate_predictions(y_valid, y_proba, threshold=0.5)

        mlflow.log_param("model_name", model_name)
        mlflow.log_param("mlflow_flavor", mlflow_flavor)
        mlflow.log_param("imbalance_strategy", imbalance_strategy)
        mlflow.log_param("threshold", 0.5)
        mlflow.log_param("train_rows", X_train.shape[0])
        mlflow.log_param("valid_rows", X_valid.shape[0])
        mlflow.log_param("num_features", X_train.shape[1])
        mlflow.log_param("positive_class_train", int((y_train == 1).sum()))
        mlflow.log_param("negative_class_train", int((y_train == 0).sum()))

        model_params = model.get_params()

        for param_name, param_value in model_params.items():
            if isinstance(param_value, (str, int, float, bool, type(None))):
                mlflow.log_param(param_name, param_value)

        for metric_name, metric_value in metrics.items():
            mlflow.log_metric(metric_name, metric_value)

        model_path = MODELS_DIR / f"{model_name}.pkl"
        joblib.dump(model, model_path)

        mlflow.log_artifact(str(model_path), artifact_path="pickle_model")

        log_model_to_mlflow(model, mlflow_flavor)

        print(f"{model_name} completed.")
        print("PR-AUC:", metrics["pr_auc"])
        print("Recall:", metrics["recall"])
        print("Precision:", metrics["precision"])
        print("False Negatives:", metrics["false_negatives"])
        print("False Positives:", metrics["false_positives"])

        return {
            "model_name": model_name,
            "imbalance_strategy": imbalance_strategy,
            "mlflow_flavor": mlflow_flavor,
            **metrics,
        }


def main():
    mlflow.set_tracking_uri("file:./mlruns")
    mlflow.set_experiment("fraud-detection-model-zoo")

    X_train, X_valid, y_train, y_valid = load_processed_data()

    print("Training data shape:", X_train.shape)
    print("Validation data shape:", X_valid.shape)

    print("\nTraining class distribution:")
    print(y_train.value_counts())

    models = get_models(y_train)

    results = []

    for model_name, model_config in models.items():
        try:
            result = run_single_experiment(
                model_name=model_name,
                model_config=model_config,
                X_train=X_train,
                y_train=y_train,
                X_valid=X_valid,
                y_valid=y_valid,
            )
            results.append(result)

        except Exception as e:
            print(f"\nModel {model_name} failed.")
            print("Error:", e)

    results_df = pd.DataFrame(results)

    if not results_df.empty:
        results_df = results_df.sort_values(by="pr_auc", ascending=False)
        results_path = REPORTS_DIR / "model_zoo_results.csv"
        results_df.to_csv(results_path, index=False)

        print("\nFinal model comparison:")
        print(results_df)

        print(f"\nSaved results to {results_path}")
    else:
        print("\nNo models completed successfully.")


if __name__ == "__main__":
    main()