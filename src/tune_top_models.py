from pathlib import Path
import os
import json
import warnings
import traceback

import joblib
import mlflow
import mlflow.sklearn
import mlflow.xgboost
import mlflow.lightgbm
import optuna

import numpy as np
import pandas as pd

from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
)

from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE

from lightgbm import LGBMClassifier
from xgboost import XGBClassifier


warnings.filterwarnings("ignore")


RANDOM_STATE = 42
DATA_DIR = Path("data/processed")
MODELS_DIR = Path("models")
REPORTS_DIR = Path("reports")
OPTUNA_DIR = REPORTS_DIR / "optuna"

MODELS_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)
OPTUNA_DIR.mkdir(parents=True, exist_ok=True)


EXPERIMENT_NAME = "fraud-detection-optuna-top-4"


def setup_mlflow():
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(EXPERIMENT_NAME)

    print(f"MLflow tracking URI: {tracking_uri}")
    print(f"MLflow experiment: {EXPERIMENT_NAME}")


def load_data():
    X_train = pd.read_csv(DATA_DIR / "X_train.csv")
    X_valid = pd.read_csv(DATA_DIR / "X_valid.csv")

    y_train = pd.read_csv(DATA_DIR / "y_train.csv")["Class"]
    y_valid = pd.read_csv(DATA_DIR / "y_valid.csv")["Class"]

    return X_train, X_valid, y_train, y_valid


def metrics_at_threshold(y_true, y_score, threshold=0.5):
    y_pred = (y_score >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    return {
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


def optimize_threshold_by_f1(y_true, y_score):
    best_threshold = 0.5
    best_metrics = None
    best_f1 = -1

    for threshold in np.linspace(0.01, 0.99, 99):
        metrics = metrics_at_threshold(y_true, y_score, threshold)

        if metrics["f1_score"] > best_f1:
            best_f1 = metrics["f1_score"]
            best_threshold = float(threshold)
            best_metrics = metrics

    return best_threshold, best_metrics


def optimize_threshold_by_cost(
    y_true,
    y_score,
    false_negative_cost=5000,
    false_positive_cost=100,
):
    best_threshold = 0.5
    best_cost = float("inf")
    best_metrics = None

    for threshold in np.linspace(0.01, 0.99, 99):
        metrics = metrics_at_threshold(y_true, y_score, threshold)

        total_cost = (
            metrics["false_negatives"] * false_negative_cost
            + metrics["false_positives"] * false_positive_cost
        )

        if total_cost < best_cost:
            best_cost = total_cost
            best_threshold = float(threshold)
            best_metrics = metrics

    return best_threshold, best_cost, best_metrics


def get_scores(model, X):
    return model.predict_proba(X)[:, 1]


def calculate_scale_pos_weight(y_train):
    negative = int((y_train == 0).sum())
    positive = int((y_train == 1).sum())
    return negative / positive


# ----------------------------------------------------
# Model builders
# ----------------------------------------------------

def build_smote_lightgbm(trial):
    smote_sampling_strategy = trial.suggest_float(
        "smote_sampling_strategy",
        0.05,
        0.50,
    )

    smote_k_neighbors = trial.suggest_int(
        "smote_k_neighbors",
        3,
        10,
    )

    model = LGBMClassifier(
        n_estimators=trial.suggest_int("n_estimators", 200, 1200),
        learning_rate=trial.suggest_float("learning_rate", 0.005, 0.20, log=True),
        num_leaves=trial.suggest_int("num_leaves", 16, 256),
        max_depth=trial.suggest_int("max_depth", 3, 12),
        min_child_samples=trial.suggest_int("min_child_samples", 5, 120),
        subsample=trial.suggest_float("subsample", 0.50, 1.00),
        colsample_bytree=trial.suggest_float("colsample_bytree", 0.50, 1.00),
        reg_alpha=trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
        reg_lambda=trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbosity=-1,
    )

    pipeline = Pipeline([
        (
            "smote",
            SMOTE(
                sampling_strategy=smote_sampling_strategy,
                k_neighbors=smote_k_neighbors,
                random_state=RANDOM_STATE,
            ),
        ),
        ("model", model),
    ])

    return pipeline


def build_lightgbm_balanced(trial):
    model = LGBMClassifier(
        n_estimators=trial.suggest_int("n_estimators", 200, 1200),
        learning_rate=trial.suggest_float("learning_rate", 0.005, 0.20, log=True),
        num_leaves=trial.suggest_int("num_leaves", 16, 256),
        max_depth=trial.suggest_int("max_depth", 3, 12),
        min_child_samples=trial.suggest_int("min_child_samples", 5, 120),
        subsample=trial.suggest_float("subsample", 0.50, 1.00),
        colsample_bytree=trial.suggest_float("colsample_bytree", 0.50, 1.00),
        reg_alpha=trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
        reg_lambda=trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbosity=-1,
    )

    return model


def build_xgboost_weighted(trial, y_train):
    scale_pos_weight = calculate_scale_pos_weight(y_train)

    model = XGBClassifier(
        n_estimators=trial.suggest_int("n_estimators", 200, 1200),
        max_depth=trial.suggest_int("max_depth", 3, 12),
        learning_rate=trial.suggest_float("learning_rate", 0.005, 0.20, log=True),
        min_child_weight=trial.suggest_float("min_child_weight", 1e-2, 20.0, log=True),
        subsample=trial.suggest_float("subsample", 0.50, 1.00),
        colsample_bytree=trial.suggest_float("colsample_bytree", 0.50, 1.00),
        gamma=trial.suggest_float("gamma", 1e-8, 10.0, log=True),
        reg_alpha=trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
        reg_lambda=trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
        scale_pos_weight=trial.suggest_float(
            "scale_pos_weight",
            scale_pos_weight * 0.25,
            scale_pos_weight * 2.0,
        ),
        eval_metric="logloss",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    return model


def build_extra_trees_balanced(trial):
    model = ExtraTreesClassifier(
        n_estimators=trial.suggest_int("n_estimators", 200, 1000),
        max_depth=trial.suggest_int("max_depth", 5, 30),
        min_samples_split=trial.suggest_int("min_samples_split", 2, 30),
        min_samples_leaf=trial.suggest_int("min_samples_leaf", 1, 20),
        max_features=trial.suggest_categorical(
            "max_features",
            ["sqrt", "log2", None],
        ),
        bootstrap=trial.suggest_categorical("bootstrap", [True, False]),
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    return model


def build_model(model_name, trial, y_train):
    if model_name == "smote_lightgbm":
        return build_smote_lightgbm(trial)

    if model_name == "lightgbm_balanced":
        return build_lightgbm_balanced(trial)

    if model_name == "xgboost_weighted":
        return build_xgboost_weighted(trial, y_train)

    if model_name == "extra_trees_balanced":
        return build_extra_trees_balanced(trial)

    raise ValueError(f"Unsupported model name: {model_name}")


def safe_log_best_model(model, model_name):
    """
    Save best model reliably. MLflow native model logging can be fragile for
    pipelines, so we always save a joblib artifact first.
    """
    model_path = MODELS_DIR / f"tuned_{model_name}.pkl"
    joblib.dump(model, model_path)
    mlflow.log_artifact(str(model_path), artifact_path="best_model_pickle")

    try:
        if model_name == "xgboost_weighted":
            mlflow.xgboost.log_model(model, name="best_model")
        elif model_name == "lightgbm_balanced":
            mlflow.lightgbm.log_model(model, name="best_model")
        else:
            mlflow.sklearn.log_model(model, name="best_model")

        mlflow.log_param("best_model_logged_to_mlflow_flavor", True)

    except Exception as e:
        mlflow.log_param("best_model_logged_to_mlflow_flavor", False)
        mlflow.log_param("best_model_log_error", str(e)[:250])
        print(f"Native MLflow model logging skipped for {model_name}: {e}")

    return model_path


def tune_single_model(
    model_name,
    X_train,
    y_train,
    X_valid,
    y_valid,
    n_trials=50,
):
    print(f"\n==============================")
    print(f"Tuning {model_name}")
    print(f"Trials: {n_trials}")
    print(f"==============================")

    trial_results = []

    with mlflow.start_run(run_name=f"OPTUNA_PARENT_{model_name}") as parent_run:
        mlflow.log_param("model_name", model_name)
        mlflow.log_param("n_trials", n_trials)
        mlflow.log_param("objective_metric", "pr_auc")
        mlflow.log_param("selection_note", "Primary objective is PR-AUC; F1 and cost thresholds are logged for deployment selection.")

        def objective(trial):
            with mlflow.start_run(run_name=f"{model_name}_trial_{trial.number}", nested=True):
                model = build_model(model_name, trial, y_train)

                model.fit(X_train, y_train)
                y_score = get_scores(model, X_valid)

                default_metrics = metrics_at_threshold(y_valid, y_score, threshold=0.5)
                best_f1_threshold, best_f1_metrics = optimize_threshold_by_f1(y_valid, y_score)
                best_cost_threshold, best_cost, best_cost_metrics = optimize_threshold_by_cost(y_valid, y_score)

                params = trial.params

                mlflow.log_params(params)
                mlflow.log_param("trial_number", trial.number)
                mlflow.log_param("model_name", model_name)
                mlflow.log_param("parent_run_id", parent_run.info.run_id)
                mlflow.log_param("default_threshold", 0.5)
                mlflow.log_param("best_f1_threshold", best_f1_threshold)
                mlflow.log_param("best_cost_threshold", best_cost_threshold)

                for key, value in default_metrics.items():
                    mlflow.log_metric(f"default_{key}", value)

                for key, value in best_f1_metrics.items():
                    mlflow.log_metric(f"best_f1_{key}", value)

                for key, value in best_cost_metrics.items():
                    mlflow.log_metric(f"best_cost_{key}", value)

                mlflow.log_metric("minimum_business_cost", best_cost)

                row = {
                    "model_name": model_name,
                    "trial_number": trial.number,
                    "objective_pr_auc": default_metrics["pr_auc"],
                    "default_precision": default_metrics["precision"],
                    "default_recall": default_metrics["recall"],
                    "default_f1_score": default_metrics["f1_score"],
                    "default_roc_auc": default_metrics["roc_auc"],
                    "best_f1_threshold": best_f1_threshold,
                    "best_f1_precision": best_f1_metrics["precision"],
                    "best_f1_recall": best_f1_metrics["recall"],
                    "best_f1_score": best_f1_metrics["f1_score"],
                    "best_cost_threshold": best_cost_threshold,
                    "minimum_business_cost": best_cost,
                    "best_cost_precision": best_cost_metrics["precision"],
                    "best_cost_recall": best_cost_metrics["recall"],
                    "best_cost_f1_score": best_cost_metrics["f1_score"],
                    "best_cost_false_negatives": best_cost_metrics["false_negatives"],
                    "best_cost_false_positives": best_cost_metrics["false_positives"],
                    **params,
                }

                trial_results.append(row)

                return default_metrics["pr_auc"]

        study = optuna.create_study(
            study_name=f"{model_name}_study",
            direction="maximize",
            sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE),
            pruner=optuna.pruners.MedianPruner(n_warmup_steps=10),
        )

        study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

        best_params = study.best_params
        best_value = study.best_value

        print(f"\nBest PR-AUC for {model_name}: {best_value}")
        print("Best params:")
        print(json.dumps(best_params, indent=4))

        # Refit best model on train data and evaluate on validation data.
        best_trial = optuna.trial.FixedTrial(best_params)
        best_model = build_model(model_name, best_trial, y_train)
        best_model.fit(X_train, y_train)

        y_score = get_scores(best_model, X_valid)

        default_metrics = metrics_at_threshold(y_valid, y_score, threshold=0.5)
        best_f1_threshold, best_f1_metrics = optimize_threshold_by_f1(y_valid, y_score)
        best_cost_threshold, best_cost, best_cost_metrics = optimize_threshold_by_cost(y_valid, y_score)

        mlflow.log_params({f"best_{k}": v for k, v in best_params.items()})
        mlflow.log_metric("best_objective_pr_auc", best_value)
        mlflow.log_metric("final_default_pr_auc", default_metrics["pr_auc"])
        mlflow.log_metric("final_default_precision", default_metrics["precision"])
        mlflow.log_metric("final_default_recall", default_metrics["recall"])
        mlflow.log_metric("final_default_f1_score", default_metrics["f1_score"])
        mlflow.log_metric("final_best_f1_threshold", best_f1_threshold)
        mlflow.log_metric("final_best_f1_score", best_f1_metrics["f1_score"])
        mlflow.log_metric("final_best_f1_precision", best_f1_metrics["precision"])
        mlflow.log_metric("final_best_f1_recall", best_f1_metrics["recall"])
        mlflow.log_metric("final_best_cost_threshold", best_cost_threshold)
        mlflow.log_metric("final_minimum_business_cost", best_cost)
        mlflow.log_metric("final_best_cost_precision", best_cost_metrics["precision"])
        mlflow.log_metric("final_best_cost_recall", best_cost_metrics["recall"])
        mlflow.log_metric("final_best_cost_false_negatives", best_cost_metrics["false_negatives"])
        mlflow.log_metric("final_best_cost_false_positives", best_cost_metrics["false_positives"])

        model_path = safe_log_best_model(best_model, model_name)

        trials_df = pd.DataFrame(trial_results)
        trials_path = OPTUNA_DIR / f"{model_name}_optuna_trials.csv"
        trials_df.to_csv(trials_path, index=False)
        mlflow.log_artifact(str(trials_path), artifact_path="optuna_trials")

        study_path = OPTUNA_DIR / f"{model_name}_study.pkl"
        joblib.dump(study, study_path)
        mlflow.log_artifact(str(study_path), artifact_path="optuna_study")

        final_result = {
            "model_name": model_name,
            "best_objective_pr_auc": best_value,
            "final_default_pr_auc": default_metrics["pr_auc"],
            "final_default_precision": default_metrics["precision"],
            "final_default_recall": default_metrics["recall"],
            "final_default_f1_score": default_metrics["f1_score"],
            "final_best_f1_threshold": best_f1_threshold,
            "final_best_f1_score": best_f1_metrics["f1_score"],
            "final_best_f1_precision": best_f1_metrics["precision"],
            "final_best_f1_recall": best_f1_metrics["recall"],
            "final_best_cost_threshold": best_cost_threshold,
            "final_minimum_business_cost": best_cost,
            "final_best_cost_precision": best_cost_metrics["precision"],
            "final_best_cost_recall": best_cost_metrics["recall"],
            "final_best_cost_false_negatives": best_cost_metrics["false_negatives"],
            "final_best_cost_false_positives": best_cost_metrics["false_positives"],
            "model_path": str(model_path),
            **{f"best_param_{k}": v for k, v in best_params.items()},
        }

        return final_result


def main():
    setup_mlflow()

    X_train, X_valid, y_train, y_valid = load_data()

    print("Train shape:", X_train.shape)
    print("Validation shape:", X_valid.shape)
    print("Train class distribution:")
    print(y_train.value_counts())

    top_models = [
        "smote_lightgbm",
        "lightgbm_balanced",
        "xgboost_weighted",
        "extra_trees_balanced",
    ]

    # Increase this later to 100 or 200 for stronger final results.
    n_trials_per_model = int(os.getenv("OPTUNA_N_TRIALS", "50"))

    final_results = []
    failures = []

    for model_name in top_models:
        try:
            result = tune_single_model(
                model_name=model_name,
                X_train=X_train,
                y_train=y_train,
                X_valid=X_valid,
                y_valid=y_valid,
                n_trials=n_trials_per_model,
            )
            final_results.append(result)

        except Exception as e:
            print(f"\nFAILED tuning {model_name}")
            print(e)
            traceback.print_exc()

            failures.append({
                "model_name": model_name,
                "error": str(e),
            })

    results_df = pd.DataFrame(final_results)

    if not results_df.empty:
        results_df = results_df.sort_values(
            by=[
                "final_default_pr_auc",
                "final_best_f1_score",
                "final_minimum_business_cost",
            ],
            ascending=[False, False, True],
        )

        output_path = REPORTS_DIR / "top_4_tuned_model_results.csv"
        results_df.to_csv(output_path, index=False)

        print("\nFinal tuned model comparison:")
        print(results_df[[
            "model_name",
            "final_default_pr_auc",
            "final_best_f1_score",
            "final_best_f1_precision",
            "final_best_f1_recall",
            "final_minimum_business_cost",
            "final_best_cost_false_negatives",
            "final_best_cost_false_positives",
            "model_path",
        ]])

        print(f"\nSaved tuned results to {output_path}")

    if failures:
        failures_df = pd.DataFrame(failures)
        failure_path = REPORTS_DIR / "top_4_tuning_failures.csv"
        failures_df.to_csv(failure_path, index=False)
        print(f"\nSaved failures to {failure_path}")


if __name__ == "__main__":
    main()