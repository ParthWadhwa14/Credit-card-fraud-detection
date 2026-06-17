import pandas as pd
from pathlib import Path


RAW_DATA_PATH = Path("data/raw/creditcard.csv")


def load_raw_data(path: Path = RAW_DATA_PATH) -> pd.DataFrame:
    """
    Load raw credit card fraud dataset.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {path}. "
            "Please download it using Kaggle CLI first."
        )

    df = pd.read_csv(path)
    return df


if __name__ == "__main__":
    df = load_raw_data()

    print("Dataset loaded successfully!")
    print("Shape:", df.shape)
    print("\nColumns:")
    print(df.columns.tolist())

    print("\nClass distribution:")
    print(df["Class"].value_counts())

    print("\nClass percentage:")
    print(df["Class"].value_counts(normalize=True) * 100)