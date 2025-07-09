import pandas as pd
import joblib
import argparse
import logging
import os

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

MODEL_PATH = "breath_classifier.joblib"

def load_model():
    if not os.path.exists(MODEL_PATH):
        logging.error(f"Trained model not found: {MODEL_PATH}")
        return None
    return joblib.load(MODEL_PATH)

def load_sample(csv_path, expected_columns):
    df = pd.read_csv(csv_path)

    missing = [col for col in expected_columns if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required feature columns: {missing}")

    return df[expected_columns], df

def run_supervised(model, features_df):
    predictions = model.predict(features_df)
    for i, pred in enumerate(predictions):
        print(f"Sample {i + 1}: Predicted user → {pred}")

def run_unsupervised(full_df):
    if "isolation_flag" in full_df.columns:
        print("\nIsolation Forest Predictions:")
        print(full_df["isolation_flag"].values)

    if "svm_flag" in full_df.columns:
        print("\nOne-Class SVM Predictions:")
        print(full_df["svm_flag"].values)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="CSV file with extracted features")
    args = parser.parse_args()

    try:
        model = load_model()
        if model:
            expected_features = model.feature_names_in_
            X_new, full_df = load_sample(args.input, expected_features)
            run_supervised(model, X_new)
        else:
            logging.warning("Supervised model not found. Falling back to unsupervised mode.")
            full_df = pd.read_csv(args.input)
            run_unsupervised(full_df)
    except Exception as e:
        logging.error(f"Prediction failed: {e}")
