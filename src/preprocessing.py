import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib


RAW_DATA_PATH = Path("data/raw/creditcard.csv")
PROCESSED_DATA_DIR = Path("data/processed")
MODEL_DIR = Path("models")

PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)


def load_data(path: Path = RAW_DATA_PATH) -> pd.DataFrame:
    """
    Load raw credit card fraud dataset.
    """
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found at {path}")

    return pd.read_csv(path)


def create_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create interpretable time-based features from the Time column.
    Time is given in seconds from the first transaction.
    """
    df = df.copy()

    df["Hour"] = (df["Time"] // 3600) % 24
    df["Day"] = df["Time"] // (3600 * 24)

    return df


def split_features_target(df: pd.DataFrame):
    """
    Separate features and target.
    """
    X = df.drop(columns=["Class"])
    y = df["Class"]

    return X, y


def stratified_data_split(X, y, test_size=0.2, valid_size=0.2, random_state=42):
    """
    Create train, validation, and test splits using stratification.

    Final split:
    - Train: 64%
    - Validation: 16%
    - Test: 20%
    """
    X_train_valid, X_test, y_train_valid, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        stratify=y,
        random_state=random_state
    )

    X_train, X_valid, y_train, y_valid = train_test_split(
        X_train_valid,
        y_train_valid,
        test_size=valid_size,
        stratify=y_train_valid,
        random_state=random_state
    )

    return X_train, X_valid, X_test, y_train, y_valid, y_test


def scale_features(X_train, X_valid, X_test):
    """
    Scale selected numerical features.

    Important:
    The scaler is fitted only on training data to avoid data leakage.
    """
    features_to_scale = ["Time", "Amount", "Hour", "Day"]

    scaler = StandardScaler()

    X_train_scaled = X_train.copy()
    X_valid_scaled = X_valid.copy()
    X_test_scaled = X_test.copy()

    X_train_scaled[features_to_scale] = scaler.fit_transform(
        X_train_scaled[features_to_scale]
    )

    X_valid_scaled[features_to_scale] = scaler.transform(
        X_valid_scaled[features_to_scale]
    )

    X_test_scaled[features_to_scale] = scaler.transform(
        X_test_scaled[features_to_scale]
    )

    return X_train_scaled, X_valid_scaled, X_test_scaled, scaler


def save_processed_data(X_train, X_valid, X_test, y_train, y_valid, y_test):
    """
    Save processed train, validation, and test datasets.
    """
    X_train.to_csv(PROCESSED_DATA_DIR / "X_train.csv", index=False)
    X_valid.to_csv(PROCESSED_DATA_DIR / "X_valid.csv", index=False)
    X_test.to_csv(PROCESSED_DATA_DIR / "X_test.csv", index=False)

    y_train.to_csv(PROCESSED_DATA_DIR / "y_train.csv", index=False)
    y_valid.to_csv(PROCESSED_DATA_DIR / "y_valid.csv", index=False)
    y_test.to_csv(PROCESSED_DATA_DIR / "y_test.csv", index=False)


def save_scaler(scaler):
    """
    Save fitted scaler for future inference.
    """
    joblib.dump(scaler, MODEL_DIR / "scaler.pkl")


def run_preprocessing():
    """
    Full preprocessing pipeline.
    """
    print("Loading raw data...")
    df = load_data()

    print("Creating time features...")
    df = create_time_features(df)

    print("Splitting features and target...")
    X, y = split_features_target(df)

    print("Creating stratified train/validation/test split...")
    X_train, X_valid, X_test, y_train, y_valid, y_test = stratified_data_split(X, y)

    print("Scaling selected features...")
    X_train_scaled, X_valid_scaled, X_test_scaled, scaler = scale_features(
        X_train,
        X_valid,
        X_test
    )

    print("Saving processed datasets...")
    save_processed_data(
        X_train_scaled,
        X_valid_scaled,
        X_test_scaled,
        y_train,
        y_valid,
        y_test
    )

    print("Saving scaler...")
    save_scaler(scaler)

    print("\nPreprocessing completed successfully!")

    print("\nDataset shapes:")
    print("X_train:", X_train_scaled.shape)
    print("X_valid:", X_valid_scaled.shape)
    print("X_test :", X_test_scaled.shape)

    print("\nClass distribution:")
    print("Train:")
    print(y_train.value_counts(normalize=True) * 100)

    print("\nValidation:")
    print(y_valid.value_counts(normalize=True) * 100)

    print("\nTest:")
    print(y_test.value_counts(normalize=True) * 100)


if __name__ == "__main__":
    run_preprocessing()