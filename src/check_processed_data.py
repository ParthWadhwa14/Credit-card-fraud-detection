import pandas as pd
from pathlib import Path


PROCESSED_DATA_DIR = Path("data/processed")


def main():
    X_train = pd.read_csv(PROCESSED_DATA_DIR / "X_train.csv")
    X_valid = pd.read_csv(PROCESSED_DATA_DIR / "X_valid.csv")
    X_test = pd.read_csv(PROCESSED_DATA_DIR / "X_test.csv")

    y_train = pd.read_csv(PROCESSED_DATA_DIR / "y_train.csv")
    y_valid = pd.read_csv(PROCESSED_DATA_DIR / "y_valid.csv")
    y_test = pd.read_csv(PROCESSED_DATA_DIR / "y_test.csv")

    print("X_train:", X_train.shape)
    print("X_valid:", X_valid.shape)
    print("X_test :", X_test.shape)

    print("\ny_train distribution:")
    print(y_train["Class"].value_counts(normalize=True) * 100)

    print("\ny_valid distribution:")
    print(y_valid["Class"].value_counts(normalize=True) * 100)

    print("\ny_test distribution:")
    print(y_test["Class"].value_counts(normalize=True) * 100)

    print("\nColumns:")
    print(X_train.columns.tolist())

    assert "Class" not in X_train.columns, "Target leakage: Class found in X_train"
    assert X_train.shape[1] == X_valid.shape[1] == X_test.shape[1], "Feature mismatch"

    print("\nSanity checks passed!")


if __name__ == "__main__":
    main()