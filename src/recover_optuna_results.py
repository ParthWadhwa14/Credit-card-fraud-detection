from pathlib import Path
import os

import mlflow
import pandas as pd


TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")
EXPERIMENT_NAME = "fraud-detection-optuna-top-4"

REPORTS_DIR = Path("reports")
REPORTS_DIR.mkdir(parents=True, exist_ok=True)


def main():
    mlflow.set_tracking_uri(TRACKING_URI)

    experiment = mlflow.get_experiment_by_name(EXPERIMENT_NAME)

    if experiment is None:
        raise RuntimeError(
            f"Experiment '{EXPERIMENT_NAME}' not found. "
            "Check that you are using the same mlflow.db file."
        )

    print("MLflow tracking URI:", TRACKING_URI)
    print("Experiment ID:", experiment.experiment_id)

    runs = mlflow.search_runs(
        experiment_ids=[experiment.experiment_id],
        output_format="pandas",
    )

    print("Total MLflow runs found:", len(runs))

    # Keep only Optuna child trial runs.
    # These should have params.trial_number from our tuning script.
    trial_col = "params.trial_number"

    if trial_col not in runs.columns:
        raise RuntimeError(
            "No params.trial_number column found. "
            "It looks like trial-level MLflow logging did not happen."
        )

    trials = runs[runs[trial_col].notna()].copy()

    if trials.empty:
        raise RuntimeError("No completed Optuna trial runs found in MLflow.")

    print("Completed Optuna trial runs found:", len(trials))

    required_cols = [
        "params.model_name",
        "params.trial_number",

        "metrics.default_pr_auc",
        "metrics.default_precision",
        "metrics.default_recall",
        "metrics.default_f1_score",
        "metrics.default_roc_auc",

        "metrics.best_f1_threshold",
        "metrics.best_f1_precision",
        "metrics.best_f1_recall",
        "metrics.best_f1_f1_score",

        "metrics.best_cost_threshold",
        "metrics.minimum_business_cost",
        "metrics.best_cost_precision",
        "metrics.best_cost_recall",
        "metrics.best_cost_f1_score",
        "metrics.best_cost_false_negatives",
        "metrics.best_cost_false_positives",

        "run_id",
        "status",
        "start_time",
        "end_time",
    ]

    existing_cols = [c for c in required_cols if c in trials.columns]
    recovered = trials[existing_cols].copy()

    recovered = recovered.rename(columns={
        "params.model_name": "model_name",
        "params.trial_number": "trial_number",

        "metrics.default_pr_auc": "objective_pr_auc",
        "metrics.default_precision": "default_precision",
        "metrics.default_recall": "default_recall",
        "metrics.default_f1_score": "default_f1_score",
        "metrics.default_roc_auc": "default_roc_auc",

        "metrics.best_f1_threshold": "best_f1_threshold",
        "metrics.best_f1_precision": "best_f1_precision",
        "metrics.best_f1_recall": "best_f1_recall",
        "metrics.best_f1_f1_score": "best_f1_score",

        "metrics.best_cost_threshold": "best_cost_threshold",
        "metrics.minimum_business_cost": "minimum_business_cost",
        "metrics.best_cost_precision": "best_cost_precision",
        "metrics.best_cost_recall": "best_cost_recall",
        "metrics.best_cost_f1_score": "best_cost_f1_score",
        "metrics.best_cost_false_negatives": "best_cost_false_negatives",
        "metrics.best_cost_false_positives": "best_cost_false_positives",
    })

    # Convert trial numbers from string to numeric if possible.
    recovered["trial_number"] = pd.to_numeric(
        recovered["trial_number"],
        errors="coerce"
    )

    recovered = recovered.sort_values(
        by=["model_name", "objective_pr_auc"],
        ascending=[True, False],
    )

    all_trials_path = REPORTS_DIR / "recovered_completed_optuna_trials.csv"
    recovered.to_csv(all_trials_path, index=False)

    # Best trial per model by PR-AUC.
    best_per_model = (
        recovered
        .sort_values("objective_pr_auc", ascending=False)
        .groupby("model_name", as_index=False)
        .head(1)
        .sort_values(
            by=["objective_pr_auc", "best_f1_score", "minimum_business_cost"],
            ascending=[False, False, True],
        )
    )

    best_path = REPORTS_DIR / "recovered_top4_best_trials.csv"
    best_per_model.to_csv(best_path, index=False)

    print("\nBest trial per model:")
    display_cols = [
        "model_name",
        "trial_number",
        "objective_pr_auc",
        "default_precision",
        "default_recall",
        "default_f1_score",
        "best_f1_threshold",
        "best_f1_precision",
        "best_f1_recall",
        "best_f1_score",
        "best_cost_threshold",
        "minimum_business_cost",
        "best_cost_false_negatives",
        "best_cost_false_positives",
        "run_id",
    ]

    display_cols = [c for c in display_cols if c in best_per_model.columns]
    print(best_per_model[display_cols].to_string(index=False))

    print(f"\nSaved all recovered trials to: {all_trials_path}")
    print(f"Saved best trials to: {best_path}")


if __name__ == "__main__":
    main()