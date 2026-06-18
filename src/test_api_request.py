import json
from pathlib import Path

import pandas as pd
import requests


DATA_PATH = Path("data/processed/X_test.csv")

API_URL = "http://127.0.0.1:8000/predict"


def main():
    X_test = pd.read_csv(DATA_PATH)

    sample = X_test.iloc[0].tolist()

    payload = {
        "features": sample
    }

    response = requests.post(API_URL, json=payload)

    print("Status code:", response.status_code)
    print(json.dumps(response.json(), indent=4))


if __name__ == "__main__":
    main()