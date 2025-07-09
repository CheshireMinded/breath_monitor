import pandas as pd
import numpy as np
import logging
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler
import os

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

INPUT_CSV = "features.csv"
OUTPUT_CSV = "features.csv"

def load_data():
    df = pd.read_csv(INPUT_CSV)
    feature_cols = [
        "mean_breath_duration", "cov_breath_duration", "envelope_variability",
        "inhale_exhale_ratio", "pause_duration_mean", "duty_cycle_mean",
        "mean_inhale_duration", "mean_exhale_duration"
    ]
    df_clean = df.dropna(subset=feature_cols).copy()
    X = df_clean[feature_cols].values
    return df, df_clean, X

def apply_isolation_forest(X):
    model = IsolationForest(n_estimators=100, contamination=0.1, random_state=42)
    preds = model.fit_predict(X)
    return (preds == -1).astype(int)  # 1 = anomaly

def apply_one_class_svm(X):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    model = OneClassSVM(kernel='rbf', nu=0.05, gamma='scale')
    preds = model.fit_predict(X_scaled)
    return (preds == -1).astype(int)

def main():
    if not os.path.exists(INPUT_CSV):
        logging.error(f"Missing input file: {INPUT_CSV}")
        return

    df, df_clean, X = load_data()

    logging.info("Running Isolation Forest...")
    isolation_flags = apply_isolation_forest(X)
    df.loc[df_clean.index, "isolation_flag"] = isolation_flags

    logging.info("Running One-Class SVM...")
    svm_flags = apply_one_class_svm(X)
    df.loc[df_clean.index, "svm_flag"] = svm_flags

    logging.info(f"Saving anomaly flags to {OUTPUT_CSV}...")
    df.to_csv(OUTPUT_CSV, index=False)
    logging.info("Done.")

if __name__ == "__main__":
    main()
