from pathlib import Path
import warnings
import traceback

import joblib
import mlflow
import mlflow.sklearn
import mlflow.xgboost
import mlflow.lightgbm

import numpy as np
import pandas as pd

from sklearn.base import clone
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import (
    LogisticRegression,
    RidgeClassifier,
    SGDClassifier,
    PassiveAggressiveClassifier,
    Perceptron,
)
from sklearn.svm import LinearSVC, OneClassSVM
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier, LocalOutlierFactor
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.tree import DecisionTreeClassifier, ExtraTreeClassifier
from sklearn.ensemble import (
    RandomForestClassifier,
    ExtraTreesClassifier,
    BaggingClassifier,
    AdaBoostClassifier,
    GradientBoostingClassifier,
    HistGradientBoostingClassifier,
    IsolationForest,
)
from sklearn.neural_network import MLPClassifier
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
from imblearn.under_sampling import RandomUnderSampler
from imblearn.combine import SMOTETomek
from imblearn.ensemble import (
    BalancedRandomForestClassifier,
    EasyEnsembleClassifier,
    RUSBoostClassifier,
)

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

from mlflow_config import setup_mlflow


warnings.filterwarnings("ignore")


try:
    from catboost import CatBoostClassifier
    CATBOOST_AVAILABLE = True
except Exception:
    CATBOOST_AVAILABLE = False


DATA_DIR = Path("data/processed")
MODELS_DIR = Path("models")
REPORTS_DIR = Path("reports")

MODELS_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)


RANDOM_STATE = 42
DEFAULT_THRESHOLD = 0.5


def load_data():
    X_train = pd.read_csv(DATA_DIR / "X_train.csv")
    X_valid = pd.read_csv(DATA_DIR / "X_valid.csv")

    y_train = pd.read_csv(DATA_DIR / "y_train.csv")["Class"]
    y_valid = pd.read_csv(DATA_DIR / "y_valid.csv")["Class"]

    return X_train, X_valid, y_train, y_valid


def normalize_scores(scores):
    scores = np.asarray(scores)

    if scores.max() == scores.min():
        return np.zeros_like(scores, dtype=float)

    return (scores - scores.min()) / (scores.max() - scores.min())


def get_scores(model, X):
    """
    Convert model output into fraud probability-like scores.
    Works for:
    - predict_proba models
    - decision_function models
    - anomaly detection models
    """
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]

    if hasattr(model, "decision_function"):
        raw_scores = model.decision_function(X)
        return normalize_scores(raw_scores)

    if hasattr(model, "score_samples"):
        raw_scores = model.score_samples(X)

        # For anomaly models, lower score usually means more abnormal.
        fraud_scores = -raw_scores
        return normalize_scores(fraud_scores)

    raise ValueError("Model does not support predict_proba, decision_function, or score_samples.")


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

    thresholds = np.linspace(0.01, 0.99, 99)

    for threshold in thresholds:
        metrics = metrics_at_threshold(y_true, y_score, threshold)

        if metrics["f1_score"] > best_f1:
            best_f1 = metrics["f1_score"]
            best_threshold = threshold
            best_metrics = metrics

    return best_threshold, best_metrics


def optimize_threshold_by_cost(
    y_true,
    y_score,
    false_negative_cost=5000,
    false_positive_cost=100,
):
    """
    Business-cost threshold optimization.

    False negative = missed fraud.
    False positive = legitimate transaction flagged.
    """
    best_threshold = 0.5
    best_cost = float("inf")
    best_metrics = None

    thresholds = np.linspace(0.01, 0.99, 99)

    for threshold in thresholds:
        metrics = metrics_at_threshold(y_true, y_score, threshold)

        cost = (
            metrics["false_negatives"] * false_negative_cost
            + metrics["false_positives"] * false_positive_cost
        )

        if cost < best_cost:
            best_cost = cost
            best_threshold = threshold
            best_metrics = metrics

    return best_threshold, best_cost, best_metrics


def train_anomaly_model(model, X_train, y_train):
    """
    Train anomaly detection models mostly on legitimate transactions.
    """
    X_normal = X_train[y_train == 0]
    model.fit(X_normal)
    return model


def get_model_registry(y_train):
    neg = int((y_train == 0).sum())
    pos = int((y_train == 1).sum())
    scale_pos_weight = neg / pos

    models = {}

    # -------------------------
    # Simple baselines
    # -------------------------
    models["dummy_most_frequent"] = {
        "model": DummyClassifier(strategy="most_frequent"),
        "family": "baseline",
        "imbalance_strategy": "none",
        "type": "supervised",
        "flavor": "sklearn",
    }

    models["dummy_stratified"] = {
        "model": DummyClassifier(strategy="stratified", random_state=RANDOM_STATE),
        "family": "baseline",
        "imbalance_strategy": "none",
        "type": "supervised",
        "flavor": "sklearn",
    }

    # -------------------------
    # Linear models
    # -------------------------
    models["logistic_regression_balanced"] = {
        "model": LogisticRegression(
            class_weight="balanced",
            max_iter=2000,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
        "family": "linear",
        "imbalance_strategy": "class_weight_balanced",
        "type": "supervised",
        "flavor": "sklearn",
    }

    models["ridge_classifier_balanced"] = {
        "model": RidgeClassifier(class_weight="balanced", random_state=RANDOM_STATE),
        "family": "linear",
        "imbalance_strategy": "class_weight_balanced",
        "type": "supervised",
        "flavor": "sklearn",
    }

    models["sgd_log_loss_balanced"] = {
        "model": SGDClassifier(
            loss="log_loss",
            class_weight="balanced",
            max_iter=2000,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
        "family": "linear",
        "imbalance_strategy": "class_weight_balanced",
        "type": "supervised",
        "flavor": "sklearn",
    }

    models["passive_aggressive_balanced"] = {
        "model": PassiveAggressiveClassifier(
            class_weight="balanced",
            max_iter=2000,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
        "family": "linear",
        "imbalance_strategy": "class_weight_balanced",
        "type": "supervised",
        "flavor": "sklearn",
    }

    models["perceptron_balanced"] = {
        "model": Perceptron(
            class_weight="balanced",
            max_iter=2000,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
        "family": "linear",
        "imbalance_strategy": "class_weight_balanced",
        "type": "supervised",
        "flavor": "sklearn",
    }

    models["linear_svc_balanced"] = {
        "model": LinearSVC(
            class_weight="balanced",
            max_iter=5000,
            random_state=RANDOM_STATE,
        ),
        "family": "svm",
        "imbalance_strategy": "class_weight_balanced",
        "type": "supervised",
        "flavor": "sklearn",
    }

    # -------------------------
    # Probability / statistical models
    # -------------------------
    models["gaussian_naive_bayes"] = {
        "model": GaussianNB(),
        "family": "naive_bayes",
        "imbalance_strategy": "none",
        "type": "supervised",
        "flavor": "sklearn",
    }

    models["linear_discriminant_analysis"] = {
        "model": LinearDiscriminantAnalysis(),
        "family": "discriminant_analysis",
        "imbalance_strategy": "none",
        "type": "supervised",
        "flavor": "sklearn",
    }

    models["quadratic_discriminant_analysis"] = {
        "model": QuadraticDiscriminantAnalysis(),
        "family": "discriminant_analysis",
        "imbalance_strategy": "none",
        "type": "supervised",
        "flavor": "sklearn",
    }

    # -------------------------
    # Neighbor model
    # -------------------------
    models["knn_distance_weighted"] = {
        "model": KNeighborsClassifier(
            n_neighbors=7,
            weights="distance",
            n_jobs=-1,
        ),
        "family": "neighbors",
        "imbalance_strategy": "distance_weighting",
        "type": "supervised",
        "flavor": "sklearn",
    }

    # -------------------------
    # Tree models
    # -------------------------
    models["decision_tree_balanced"] = {
        "model": DecisionTreeClassifier(
            class_weight="balanced",
            max_depth=10,
            random_state=RANDOM_STATE,
        ),
        "family": "tree",
        "imbalance_strategy": "class_weight_balanced",
        "type": "supervised",
        "flavor": "sklearn",
    }

    models["extra_tree_balanced"] = {
        "model": ExtraTreeClassifier(
            class_weight="balanced",
            max_depth=10,
            random_state=RANDOM_STATE,
        ),
        "family": "tree",
        "imbalance_strategy": "class_weight_balanced",
        "type": "supervised",
        "flavor": "sklearn",
    }

    # -------------------------
    # Ensemble models
    # -------------------------
    models["random_forest_balanced"] = {
        "model": RandomForestClassifier(
            n_estimators=300,
            max_depth=12,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
        "family": "ensemble",
        "imbalance_strategy": "class_weight_balanced",
        "type": "supervised",
        "flavor": "sklearn",
    }

    models["extra_trees_balanced"] = {
        "model": ExtraTreesClassifier(
            n_estimators=300,
            max_depth=12,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
        "family": "ensemble",
        "imbalance_strategy": "class_weight_balanced",
        "type": "supervised",
        "flavor": "sklearn",
    }

    models["bagging_decision_tree"] = {
        "model": BaggingClassifier(
            estimator=DecisionTreeClassifier(
                max_depth=10,
                class_weight="balanced",
                random_state=RANDOM_STATE,
            ),
            n_estimators=100,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
        "family": "ensemble",
        "imbalance_strategy": "base_estimator_class_weight",
        "type": "supervised",
        "flavor": "sklearn",
    }

    models["adaboost"] = {
        "model": AdaBoostClassifier(
            n_estimators=300,
            learning_rate=0.05,
            random_state=RANDOM_STATE,
        ),
        "family": "boosting",
        "imbalance_strategy": "none_baseline",
        "type": "supervised",
        "flavor": "sklearn",
    }

    models["gradient_boosting"] = {
        "model": GradientBoostingClassifier(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=3,
            random_state=RANDOM_STATE,
        ),
        "family": "boosting",
        "imbalance_strategy": "none_baseline",
        "type": "supervised",
        "flavor": "sklearn",
    }

    models["hist_gradient_boosting"] = {
        "model": HistGradientBoostingClassifier(
            max_iter=300,
            learning_rate=0.05,
            max_leaf_nodes=31,
            random_state=RANDOM_STATE,
        ),
        "family": "boosting",
        "imbalance_strategy": "none_baseline",
        "type": "supervised",
        "flavor": "sklearn",
    }

    # -------------------------
    # Gradient boosting libraries
    # -------------------------
    models["xgboost_weighted"] = {
        "model": XGBClassifier(
            n_estimators=400,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=scale_pos_weight,
            eval_metric="logloss",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
        "family": "boosting",
        "imbalance_strategy": "scale_pos_weight",
        "type": "supervised",
        "flavor": "xgboost",
    }

    models["lightgbm_balanced"] = {
        "model": LGBMClassifier(
            n_estimators=400,
            learning_rate=0.05,
            num_leaves=31,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=-1,
            verbosity=-1,
        ),
        "family": "boosting",
        "imbalance_strategy": "class_weight_balanced",
        "type": "supervised",
        "flavor": "lightgbm",
    }

    if CATBOOST_AVAILABLE:
        models["catboost_balanced"] = {
            "model": CatBoostClassifier(
                iterations=400,
                learning_rate=0.05,
                depth=6,
                auto_class_weights="Balanced",
                random_seed=RANDOM_STATE,
                verbose=False,
            ),
            "family": "boosting",
            "imbalance_strategy": "auto_class_weights_balanced",
            "type": "supervised",
            "flavor": "sklearn",
        }

    # -------------------------
    # Imbalanced-learn ensembles
    # -------------------------
    models["balanced_random_forest"] = {
        "model": BalancedRandomForestClassifier(
            n_estimators=300,
            max_depth=12,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
        "family": "imbalance_ensemble",
        "imbalance_strategy": "balanced_bootstrap",
        "type": "supervised",
        "flavor": "sklearn",
    }

    models["easy_ensemble"] = {
        "model": EasyEnsembleClassifier(
            n_estimators=20,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
        "family": "imbalance_ensemble",
        "imbalance_strategy": "under_sampling_ensemble",
        "type": "supervised",
        "flavor": "sklearn",
    }

    models["rusboost"] = {
        "model": RUSBoostClassifier(
            n_estimators=300,
            learning_rate=0.05,
            random_state=RANDOM_STATE,
        ),
        "family": "imbalance_ensemble",
        "imbalance_strategy": "random_under_sampling_boosting",
        "type": "supervised",
        "flavor": "sklearn",
    }

    # -------------------------
    # Neural network
    # -------------------------
    models["mlp_classifier"] = {
        "model": MLPClassifier(
            hidden_layer_sizes=(64, 32),
            activation="relu",
            alpha=1e-4,
            batch_size=512,
            learning_rate_init=1e-3,
            max_iter=50,
            early_stopping=True,
            random_state=RANDOM_STATE,
        ),
        "family": "neural_network",
        "imbalance_strategy": "none_baseline",
        "type": "supervised",
        "flavor": "sklearn",
    }

    # -------------------------
    # Anomaly detection models
    # -------------------------
    models["isolation_forest"] = {
        "model": IsolationForest(
            n_estimators=300,
            contamination=pos / (pos + neg),
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
        "family": "anomaly_detection",
        "imbalance_strategy": "train_on_normal_class",
        "type": "anomaly",
        "flavor": "sklearn",
    }

    models["local_outlier_factor"] = {
        "model": LocalOutlierFactor(
            n_neighbors=35,
            contamination=pos / (pos + neg),
            novelty=True,
            n_jobs=-1,
        ),
        "family": "anomaly_detection",
        "imbalance_strategy": "train_on_normal_class",
        "type": "anomaly",
        "flavor": "sklearn",
    }

    models["one_class_svm"] = {
        "model": OneClassSVM(
            kernel="rbf",
            gamma="scale",
            nu=pos / (pos + neg),
        ),
        "family": "anomaly_detection",
        "imbalance_strategy": "train_on_normal_class",
        "type": "anomaly",
        "flavor": "sklearn",
    }

    return models


def add_resampling_models(base_models):
    """
    Add selected resampling combinations.

    We do not combine every sampler with every model because that creates
    dozens of redundant and slow experiments. These cover the important cases.
    """
    resampling_models = {}

    selected_estimators = {
        "logistic_regression": LogisticRegression(
            max_iter=2000,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=300,
            max_depth=12,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
        "extra_trees": ExtraTreesClassifier(
            n_estimators=300,
            max_depth=12,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
        "xgboost": XGBClassifier(
            n_estimators=300,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            eval_metric="logloss",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
        "lightgbm": LGBMClassifier(
            n_estimators=300,
            learning_rate=0.05,
            num_leaves=31,
            random_state=RANDOM_STATE,
            n_jobs=-1,
            verbosity=-1,
        ),
    }

    samplers = {
        "smote": SMOTE(random_state=RANDOM_STATE),
        "random_undersampling": RandomUnderSampler(random_state=RANDOM_STATE),
        "smote_tomek": SMOTETomek(random_state=RANDOM_STATE),
    }

    for sampler_name, sampler in samplers.items():
        for estimator_name, estimator in selected_estimators.items():
            model_name = f"{sampler_name}_{estimator_name}"

            resampling_models[model_name] = {
                "model": Pipeline([
                    ("sampler", sampler),
                    ("model", estimator),
                ]),
                "family": "resampling_pipeline",
                "imbalance_strategy": sampler_name,
                "type": "supervised",
                "flavor": "sklearn_pipeline",
            }

    base_models.update(resampling_models)

    return base_models


def safe_log_model(model, model_name, flavor):
    """
    MLflow model logging can fail for some third-party estimators or imblearn pipelines.
    We never allow that to fail the experiment run.
    Metrics and pickle artifacts are still logged.
    """
    try:
        if flavor == "xgboost":
            mlflow.xgboost.log_model(model, name="model")
        elif flavor == "lightgbm":
            mlflow.lightgbm.log_model(model, name="model")
        else:
            mlflow.sklearn.log_model(model, name="model")

        mlflow.log_param("mlflow_model_logged", True)

    except Exception as e:
        mlflow.log_param("mlflow_model_logged", False)
        mlflow.log_param("mlflow_model_log_error", str(e)[:250])
        print(f"MLflow model flavor logging skipped for {model_name}: {e}")


def log_params_safely(model):
    try:
        params = model.get_params()
    except Exception:
        return

    for key, value in params.items():
        if isinstance(value, (str, int, float, bool, type(None))):
            try:
                mlflow.log_param(key, value)
            except Exception:
                pass


def run_single_model(model_name, config, X_train, y_train, X_valid, y_valid):
    model = config["model"]

    with mlflow.start_run(run_name=model_name):
        mlflow.log_param("model_name", model_name)
        mlflow.log_param("family", config["family"])
        mlflow.log_param("imbalance_strategy", config["imbalance_strategy"])
        mlflow.log_param("model_type", config["type"])
        mlflow.log_param("flavor", config["flavor"])
        mlflow.log_param("train_rows", X_train.shape[0])
        mlflow.log_param("valid_rows", X_valid.shape[0])
        mlflow.log_param("num_features", X_train.shape[1])
        mlflow.log_param("fraud_train_count", int((y_train == 1).sum()))
        mlflow.log_param("non_fraud_train_count", int((y_train == 0).sum()))
        mlflow.log_param("default_threshold", DEFAULT_THRESHOLD)

        log_params_safely(model)

        print(f"\nTraining {model_name}...")

        if config["type"] == "anomaly":
            model = train_anomaly_model(model, X_train, y_train)
        else:
            model.fit(X_train, y_train)

        y_score = get_scores(model, X_valid)

        default_metrics = metrics_at_threshold(
            y_valid,
            y_score,
            threshold=DEFAULT_THRESHOLD,
        )

        best_f1_threshold, best_f1_metrics = optimize_threshold_by_f1(
            y_valid,
            y_score,
        )

        best_cost_threshold, best_cost, best_cost_metrics = optimize_threshold_by_cost(
            y_valid,
            y_score,
            false_negative_cost=5000,
            false_positive_cost=100,
        )

        for k, v in default_metrics.items():
            mlflow.log_metric(f"default_{k}", v)

        for k, v in best_f1_metrics.items():
            mlflow.log_metric(f"best_f1_{k}", v)

        for k, v in best_cost_metrics.items():
            mlflow.log_metric(f"best_cost_{k}", v)

        mlflow.log_metric("best_f1_threshold", best_f1_threshold)
        mlflow.log_metric("best_cost_threshold", best_cost_threshold)
        mlflow.log_metric("minimum_business_cost", best_cost)

        model_path = MODELS_DIR / f"{model_name}.pkl"
        joblib.dump(model, model_path)
        mlflow.log_artifact(str(model_path), artifact_path="pickle_model")

        safe_log_model(model, model_name, config["flavor"])

        print(f"{model_name} completed")
        print(f"Default PR-AUC: {default_metrics['pr_auc']:.4f}")
        print(f"Default Recall: {default_metrics['recall']:.4f}")
        print(f"Default Precision: {default_metrics['precision']:.4f}")
        print(f"Best F1 threshold: {best_f1_threshold:.2f}")
        print(f"Best F1 score: {best_f1_metrics['f1_score']:.4f}")
        print(f"Best cost threshold: {best_cost_threshold:.2f}")
        print(f"Minimum cost: {best_cost}")

        return {
            "model_name": model_name,
            "family": config["family"],
            "imbalance_strategy": config["imbalance_strategy"],
            "type": config["type"],
            "default_threshold": DEFAULT_THRESHOLD,
            "best_f1_threshold": best_f1_threshold,
            "best_cost_threshold": best_cost_threshold,
            "minimum_business_cost": best_cost,
            **{f"default_{k}": v for k, v in default_metrics.items()},
            **{f"best_f1_{k}": v for k, v in best_f1_metrics.items()},
            **{f"best_cost_{k}": v for k, v in best_cost_metrics.items()},
        }


def main():
    setup_mlflow()

    X_train, X_valid, y_train, y_valid = load_data()

    print("Training rows:", X_train.shape[0])
    print("Validation rows:", X_valid.shape[0])
    print("Features:", X_train.shape[1])
    print("\nTraining class distribution:")
    print(y_train.value_counts())

    models = get_model_registry(y_train)
    models = add_resampling_models(models)

    print(f"\nTotal models/configurations to train: {len(models)}")

    results = []
    failures = []

    for model_name, config in models.items():
        try:
            result = run_single_model(
                model_name=model_name,
                config=config,
                X_train=X_train,
                y_train=y_train,
                X_valid=X_valid,
                y_valid=y_valid,
            )
            results.append(result)

        except Exception as e:
            print(f"\nFAILED: {model_name}")
            print(e)
            traceback.print_exc()

            failures.append({
                "model_name": model_name,
                "error": str(e),
            })

            with mlflow.start_run(run_name=f"FAILED_{model_name}"):
                mlflow.log_param("model_name", model_name)
                mlflow.log_param("status", "failed")
                mlflow.log_param("error", str(e)[:500])

    results_df = pd.DataFrame(results)

    if not results_df.empty:
        results_df = results_df.sort_values(
            by=["default_pr_auc", "best_f1_f1_score", "best_cost_recall"],
            ascending=False,
        )

        results_path = REPORTS_DIR / "full_model_zoo_results.csv"
        results_df.to_csv(results_path, index=False)

        print("\nFinal full model zoo comparison:")
        print(results_df[[
            "model_name",
            "family",
            "imbalance_strategy",
            "default_pr_auc",
            "default_recall",
            "default_precision",
            "best_f1_threshold",
            "best_f1_f1_score",
            "best_cost_threshold",
            "minimum_business_cost",
            "best_cost_false_negatives",
            "best_cost_false_positives",
        ]].head(20))

        print(f"\nSaved results to {results_path}")

    if failures:
        failures_df = pd.DataFrame(failures)
        failures_path = REPORTS_DIR / "full_model_zoo_failures.csv"
        failures_df.to_csv(failures_path, index=False)

        print(f"\nSome models failed. Saved failures to {failures_path}")


if __name__ == "__main__":
    main()