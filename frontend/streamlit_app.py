from pathlib import Path
from typing import List, Tuple, Optional
import json

import joblib
import pandas as pd
import requests
import streamlit as st
import os

# ---------------------------------------------------------
# Constants
# ---------------------------------------------------------


DEFAULT_API_URL = os.getenv("FASTAPI_URL", "http://127.0.0.1:8000")
PUBLIC_API_URL = os.getenv("PUBLIC_API_URL", "http://localhost:8000")

# Model/API expects processed feature columns:
# Time, V1-V28, Amount, Hour, Day
FEATURE_COLUMNS = [
    "Time",
    "V1", "V2", "V3", "V4", "V5", "V6", "V7",
    "V8", "V9", "V10", "V11", "V12", "V13", "V14",
    "V15", "V16", "V17", "V18", "V19", "V20", "V21",
    "V22", "V23", "V24", "V25", "V26", "V27", "V28",
    "Amount",
    "Hour",
    "Day",
]

# Raw Kaggle dataset feature columns:
# Time, V1-V28, Amount
RAW_FEATURE_COLUMNS = [
    "Time",
    "V1", "V2", "V3", "V4", "V5", "V6", "V7",
    "V8", "V9", "V10", "V11", "V12", "V13", "V14",
    "V15", "V16", "V17", "V18", "V19", "V20", "V21",
    "V22", "V23", "V24", "V25", "V26", "V27", "V28",
    "Amount",
]

SCALE_COLUMNS = ["Time", "Amount", "Hour", "Day"]

LOCAL_X_TEST_PATH = Path("data/processed/X_test.csv")
LOCAL_Y_TEST_PATH = Path("data/processed/y_test.csv")
SCALER_PATH = Path("models/scaler.pkl")


# ---------------------------------------------------------
# Streamlit page config
# ---------------------------------------------------------

st.set_page_config(
    page_title="Fraud Detection Dashboard",
    page_icon="💳",
    layout="wide",
)


# ---------------------------------------------------------
# Cached loading functions
# ---------------------------------------------------------

@st.cache_resource
def load_scaler():
    """
    Load scaler used during training.

    The model was trained on processed data where Time, Amount, Hour, and Day
    were scaled. Raw manual inputs must therefore be transformed before sending
    to FastAPI.
    """
    if not SCALER_PATH.exists():
        return None

    return joblib.load(SCALER_PATH)


@st.cache_data
def load_local_sample_data():
    """
    Load already processed test data.

    X_test.csv should already contain the exact 32 processed columns expected
    by the API.
    """
    if not LOCAL_X_TEST_PATH.exists():
        return None

    X_test = pd.read_csv(LOCAL_X_TEST_PATH)

    if LOCAL_Y_TEST_PATH.exists():
        y_test = pd.read_csv(LOCAL_Y_TEST_PATH)
        if "Class" in y_test.columns:
            X_test["actual_class"] = y_test["Class"]

    return X_test


# ---------------------------------------------------------
# API helper functions
# ---------------------------------------------------------

def get_api_url() -> str:
    api_url = st.sidebar.text_input(
        "FastAPI Backend URL",
        value=DEFAULT_API_URL,
        help="URL where your FastAPI fraud detection API is running.",
    )
    return api_url.rstrip("/")


def check_health(api_url: str):
    try:
        response = requests.get(f"{api_url}/health", timeout=5)
        return response.status_code, response.json()
    except Exception as e:
        return None, {"error": str(e)}


def get_model_info(api_url: str):
    try:
        response = requests.get(f"{api_url}/model-info", timeout=5)
        return response.status_code, response.json()
    except Exception as e:
        return None, {"error": str(e)}


def predict_single(api_url: str, features: List[float]):
    payload = {"features": features}

    response = requests.post(
        f"{api_url}/predict",
        json=payload,
        timeout=10,
    )

    response.raise_for_status()
    return response.json()


def predict_batch(api_url: str, dataframe: pd.DataFrame):
    payload = []

    for _, row in dataframe.iterrows():
        payload.append({
            "features": row[FEATURE_COLUMNS].astype(float).tolist()
        })

    response = requests.post(
        f"{api_url}/predict-batch",
        json=payload,
        timeout=30,
    )

    response.raise_for_status()
    return response.json()


# ---------------------------------------------------------
# Preprocessing helper functions
# ---------------------------------------------------------

def derive_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add Hour and Day from raw Time column.

    Time is in seconds from the first transaction.
    """
    df = df.copy()

    df["Hour"] = (df["Time"] // 3600) % 24
    df["Day"] = df["Time"] // (3600 * 24)

    return df


def scale_processed_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Scale Time, Amount, Hour, and Day using training scaler.
    """
    scaler = load_scaler()

    if scaler is None:
        raise FileNotFoundError(
            "models/scaler.pkl not found. Raw inputs cannot be processed safely "
            "without the training scaler."
        )

    df = df.copy()
    df[SCALE_COLUMNS] = scaler.transform(df[SCALE_COLUMNS])

    return df


def raw_features_to_processed(raw_values: List[float]) -> List[float]:
    """
    Convert 30 raw Kaggle values into 32 processed model features.

    Input:
    Time + V1-V28 + Amount = 30 values

    Output:
    scaled Time + V1-V28 + scaled Amount + scaled Hour + scaled Day = 32 values
    """
    if len(raw_values) != len(RAW_FEATURE_COLUMNS):
        raise ValueError(
            f"Expected {len(RAW_FEATURE_COLUMNS)} raw values, got {len(raw_values)}."
        )

    raw_df = pd.DataFrame([raw_values], columns=RAW_FEATURE_COLUMNS)
    processed_df = derive_time_features(raw_df)
    processed_df = processed_df[FEATURE_COLUMNS]
    processed_df = scale_processed_features(processed_df)

    return processed_df.iloc[0].astype(float).tolist()


def parse_comma_separated_values(raw_input: str) -> List[float]:
    """
    Parse comma-separated numeric values from text area.
    """
    values = []

    for item in raw_input.replace("\n", "").split(","):
        item = item.strip()
        if item:
            values.append(float(item))

    return values


def prepare_uploaded_dataframe(df: pd.DataFrame) -> Tuple[pd.DataFrame, Optional[pd.Series], str]:
    """
    Accept uploaded CSV in one of these formats:

    1. Processed CSV with 32 columns:
       Time, V1-V28, Amount, Hour, Day

    2. Raw Kaggle CSV with 30 columns:
       Time, V1-V28, Amount

    3. Raw Kaggle CSV with target label:
       Time, V1-V28, Amount, Class

    Returns:
    processed_df, actual_labels, detected_format
    """
    df = df.copy()

    actual_labels = None

    if "Class" in df.columns:
        actual_labels = df["Class"].copy()
        df = df.drop(columns=["Class"])

    has_processed_cols = all(col in df.columns for col in FEATURE_COLUMNS)
    has_raw_cols = all(col in df.columns for col in RAW_FEATURE_COLUMNS)

    if has_processed_cols:
        processed_df = df[FEATURE_COLUMNS].copy()
        detected_format = "processed_32_features"

    elif has_raw_cols:
        processed_df = df[RAW_FEATURE_COLUMNS].copy()
        processed_df = derive_time_features(processed_df)
        processed_df = processed_df[FEATURE_COLUMNS]
        processed_df = scale_processed_features(processed_df)
        detected_format = "raw_30_features_scaled"

    else:
        missing_processed = [col for col in FEATURE_COLUMNS if col not in df.columns]
        missing_raw = [col for col in RAW_FEATURE_COLUMNS if col not in df.columns]

        raise ValueError(
            "Uploaded CSV does not match expected formats.\n\n"
            f"Missing processed columns: {missing_processed}\n\n"
            f"Missing raw columns: {missing_raw}"
        )

    return processed_df, actual_labels, detected_format


def render_prediction_card(result: dict):
    fraud_probability = result["fraud_probability"]
    prediction = result["prediction"]
    risk_level = result["risk_level"]
    threshold = result["threshold"]

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Fraud Probability", f"{fraud_probability:.4f}")
    col2.metric("Prediction", "Fraud" if prediction == 1 else "Non-Fraud")
    col3.metric("Risk Level", risk_level.upper())
    col4.metric("Threshold", f"{threshold:.4f}")

    if prediction == 1:
        st.error("⚠️ This transaction is flagged as potentially fraudulent.")
    else:
        st.success("✅ This transaction is predicted as legitimate.")


def show_required_feature_info():
    with st.expander("Feature format guide"):
        st.markdown("### Processed input format: 32 values")
        st.write("Use this when values already come from `data/processed/X_test.csv`.")
        st.code(", ".join(FEATURE_COLUMNS))

        st.markdown("### Raw Kaggle input format: 30 values")
        st.write("Use this when values come from raw Kaggle data without `Class` label.")
        st.code(", ".join(RAW_FEATURE_COLUMNS))

        st.info(
            "For raw input, the app automatically creates Hour and Day from Time, "
            "then scales Time, Amount, Hour, and Day using models/scaler.pkl."
        )


# ---------------------------------------------------------
# Sidebar
# ---------------------------------------------------------

st.sidebar.title("⚙️ Settings")
api_url = get_api_url()

st.sidebar.markdown("---")

if st.sidebar.button("Check API Health"):
    status, health = check_health(api_url)

    if status == 200:
        st.sidebar.success("API is healthy")
    else:
        st.sidebar.error("API is not reachable")

    st.sidebar.json(health)

st.sidebar.markdown("---")
st.sidebar.caption("Run FastAPI first:")
st.sidebar.code("uvicorn app.main:app --reload --host 127.0.0.1 --port 8000")

st.sidebar.caption("Run Streamlit:")
st.sidebar.code("streamlit run frontend/streamlit_app.py")


# ---------------------------------------------------------
# Main UI
# ---------------------------------------------------------

st.title("💳 Fraud Detection Dashboard")
st.write(
    "Interactive frontend for the production-focused weighted XGBoost fraud detection model."
)

health_status, health_response = check_health(api_url)

if health_status == 200:
    st.success("FastAPI backend is connected.")
else:
    st.warning("FastAPI backend is not connected. Start the API before predicting.")
    st.code("uvicorn app.main:app --reload --host 127.0.0.1 --port 8000")


# ---------------------------------------------------------
# Tabs
# ---------------------------------------------------------

tab1, tab2, tab3, tab4 = st.tabs([
    "Single Prediction",
    "Batch Prediction",
    "Model Info",
    "Monitoring",
])


# ---------------------------------------------------------
# Tab 1: Single prediction
# ---------------------------------------------------------

with tab1:
    st.header("Single Transaction Prediction")

    st.write(
        "Choose a sample transaction from your processed test set, paste 32 processed values, "
        "or paste 30 raw Kaggle values."
    )

    show_required_feature_info()

    sample_data = load_local_sample_data()

    input_mode = st.radio(
        "Input Mode",
        [
            "Use local processed test sample",
            "Paste 32 processed features",
            "Paste 30 raw Kaggle features",
        ],
        horizontal=False,
    )

    features = None

    if input_mode == "Use local processed test sample":
        if sample_data is None:
            st.error(
                "Local test data not found at data/processed/X_test.csv. "
                "Use manual input mode instead."
            )
        else:
            max_index = len(sample_data) - 1

            col_a, col_b = st.columns([1, 2])

            with col_a:
                sample_index = st.number_input(
                    "Sample index",
                    min_value=0,
                    max_value=max_index,
                    value=0,
                    step=1,
                )

            with col_b:
                if "actual_class" in sample_data.columns:
                    fraud_only = st.checkbox("Show only fraud samples")
                else:
                    fraud_only = False

            if fraud_only and "actual_class" in sample_data.columns:
                fraud_indices = sample_data.index[sample_data["actual_class"] == 1].tolist()

                if not fraud_indices:
                    st.warning("No fraud samples found in local y_test.csv.")
                else:
                    selected_fraud_position = st.slider(
                        "Fraud sample position",
                        min_value=0,
                        max_value=len(fraud_indices) - 1,
                        value=0,
                    )
                    sample_index = fraud_indices[selected_fraud_position]

            selected_row = sample_data.iloc[int(sample_index)]

            st.subheader("Selected Transaction")
            st.dataframe(selected_row.to_frame("value"), use_container_width=True)

            features = selected_row[FEATURE_COLUMNS].astype(float).tolist()

            if "actual_class" in selected_row.index:
                actual = int(selected_row["actual_class"])
                st.info(
                    f"Actual label in test set: {'Fraud' if actual == 1 else 'Non-Fraud'}"
                )

    elif input_mode == "Paste 32 processed features":
        example = ",".join(["0"] * len(FEATURE_COLUMNS))

        raw_input = st.text_area(
            "Paste 32 processed comma-separated feature values",
            value=example,
            height=140,
        )

        try:
            parsed_values = parse_comma_separated_values(raw_input)

            if len(parsed_values) != len(FEATURE_COLUMNS):
                st.error(
                    f"Expected {len(FEATURE_COLUMNS)} processed values, "
                    f"got {len(parsed_values)}."
                )
                features = None
            else:
                features = parsed_values
                st.success("Valid 32-value processed input.")

        except Exception as e:
            st.error(f"Could not parse feature values: {e}")
            features = None

    elif input_mode == "Paste 30 raw Kaggle features":
        example_raw = ",".join(["0"] * len(RAW_FEATURE_COLUMNS))

        raw_input = st.text_area(
            "Paste 30 raw Kaggle comma-separated feature values",
            value=example_raw,
            height=140,
            help="Paste Time, V1-V28, Amount. Do not include Class label.",
        )

        try:
            parsed_values = parse_comma_separated_values(raw_input)

            if len(parsed_values) != len(RAW_FEATURE_COLUMNS):
                st.error(
                    f"Expected {len(RAW_FEATURE_COLUMNS)} raw values, "
                    f"got {len(parsed_values)}."
                )
                features = None
            else:
                features = raw_features_to_processed(parsed_values)
                st.success(
                    "Raw input converted successfully: Hour and Day were created, "
                    "and scaling was applied."
                )

                with st.expander("Processed 32 features sent to API"):
                    processed_preview = pd.DataFrame(
                        [features],
                        columns=FEATURE_COLUMNS,
                    )
                    st.dataframe(processed_preview, use_container_width=True)

        except Exception as e:
            st.error(f"Could not process raw input: {e}")
            features = None

    if st.button("Predict Transaction", type="primary"):
        if features is None:
            st.error("No valid features available.")
        else:
            try:
                result = predict_single(api_url, features)
                render_prediction_card(result)

                with st.expander("Raw API response"):
                    st.json(result)

            except Exception as e:
                st.error(f"Prediction failed: {e}")


# ---------------------------------------------------------
# Tab 2: Batch prediction
# ---------------------------------------------------------

with tab2:
    st.header("Batch Prediction")

    st.write(
        "Upload either a processed CSV with 32 model features or a raw Kaggle CSV "
        "with 30 raw features. If the CSV includes `Class`, it will be preserved "
        "only for comparison and not sent to the model."
    )

    show_required_feature_info()

    uploaded_file = st.file_uploader(
        "Upload CSV",
        type=["csv"],
    )

    if uploaded_file is not None:
        try:
            uploaded_df = pd.read_csv(uploaded_file)

            st.subheader("Uploaded Data Preview")
            st.dataframe(uploaded_df.head(), use_container_width=True)

            processed_df, actual_labels, detected_format = prepare_uploaded_dataframe(uploaded_df)

            st.success(f"Detected format: {detected_format}")
            st.write(f"Rows available for scoring: {len(processed_df)}")

            max_rows = st.slider(
                "Rows to score",
                min_value=1,
                max_value=min(len(processed_df), 1000),
                value=min(len(processed_df), 100),
            )

            if st.button("Run Batch Prediction", type="primary"):
                scored_features_df = processed_df.head(max_rows).copy()
                response = predict_batch(api_url, scored_features_df)

                results_df = pd.DataFrame(response["results"])

                output_df = pd.concat(
                    [
                        scored_features_df.reset_index(drop=True),
                        results_df.reset_index(drop=True),
                    ],
                    axis=1,
                )

                if actual_labels is not None:
                    output_df["actual_class"] = actual_labels.head(max_rows).reset_index(drop=True)

                st.subheader("Prediction Results")
                st.dataframe(output_df, use_container_width=True)

                fraud_count = int(output_df["prediction"].sum())
                total_count = len(output_df)

                col1, col2, col3 = st.columns(3)
                col1.metric("Total Scored", total_count)
                col2.metric("Flagged Fraud", fraud_count)
                col3.metric("Fraud Flag Rate", f"{fraud_count / total_count:.2%}")

                if "actual_class" in output_df.columns:
                    correct = (output_df["prediction"] == output_df["actual_class"]).sum()
                    col4, col5 = st.columns(2)
                    col4.metric("Correct Predictions", int(correct))
                    col5.metric("Sample Accuracy", f"{correct / total_count:.2%}")

                csv_bytes = output_df.to_csv(index=False).encode("utf-8")

                st.download_button(
                    label="Download Predictions CSV",
                    data=csv_bytes,
                    file_name="fraud_predictions.csv",
                    mime="text/csv",
                )

        except Exception as e:
            st.error(f"Batch prediction failed: {e}")


# ---------------------------------------------------------
# Tab 3: Model info
# ---------------------------------------------------------

with tab3:
    st.header("Model Information")

    status, info = get_model_info(api_url)

    if status == 200:
        st.json(info)

        st.markdown("### Selection Rationale")
        st.write(
            info.get(
                "selection_reason",
                "Weighted XGBoost was selected as the production-focused model."
            )
        )

        st.markdown("### Why this model was selected")
        st.write(
            "The final model was selected to balance fraud-detection performance "
            "with low false positives, because false positives can block genuine "
            "customers and create customer dissatisfaction."
        )

    else:
        st.error("Could not fetch model info.")
        st.json(info)


# ---------------------------------------------------------
# Tab 4: Monitoring
# ---------------------------------------------------------

with tab4:
    st.header("Monitoring")

    st.write("Prometheus metrics are exposed by the FastAPI backend.")

    internal_metrics_url = f"{api_url}/metrics"
    public_metrics_url = f"{PUBLIC_API_URL}/metrics"
    
    st.markdown("### Internal metrics URL")
    st.write("Used by Streamlit container to talk to FastAPI:")
    st.code(internal_metrics_url)
    
    st.markdown("### Browser metrics URL")
    st.write("Open this in your browser:")
    st.code(public_metrics_url)
    
    st.markdown(f"[Open metrics endpoint in browser]({public_metrics_url})")

    st.code(metrics_url)

    st.markdown(f"[Open metrics endpoint]({metrics_url})")

    st.write("Important custom metrics:")

    st.code(
        """
fraud_predictions_total
fraud_flags_total
fraud_prediction_probability
        """
    )

    if st.button("Check Metrics Endpoint"):
        try:
            response = requests.get(metrics_url, timeout=5)

            if response.status_code == 200:
                st.success("Metrics endpoint is working.")
                st.text(response.text[:3000])
            else:
                st.error(f"Metrics endpoint returned status code {response.status_code}.")
                st.text(response.text[:1000])

        except Exception as e:
            st.error(f"Could not reach metrics endpoint: {e}")

    st.info(
        "Use these metrics later with Prometheus and Grafana to monitor request volume, "
        "fraud flag rate, and model confidence distribution."
    )