import os
import logging
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from collections import Counter
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupKFold, GridSearchCV
from sklearn.metrics import (
    classification_report,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

# ========== Setup Logging ==========
os.makedirs("reports", exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("reports/pipeline.log", mode="w"),
        logging.StreamHandler()
    ]
)

try:
    # ========== Load Data ==========
    df = pd.read_csv("features.csv").replace([np.inf, -np.inf], np.nan).dropna()

    if "isolation_flag" in df.columns and "svm_flag" in df.columns:
        df = df[(df["isolation_flag"] == 0) & (df["svm_flag"] == 0)]
        logging.info("Filtered out anomaly-flagged samples")

    logging.info(f"Loaded {len(df)} cleaned samples from features.csv")

    if "group" not in df.columns:
        raise ValueError("Missing 'group' column. Add a 'group' column indicating person/subject identity.")

    # Filter out classes with fewer than 2 samples
    class_counts = df["group"].value_counts()
    valid_classes = class_counts[class_counts >= 2].index
    df = df[df["group"].isin(valid_classes)]
    logging.info(f"Filtered to classes with at least 2 samples: {len(df)} remaining")

    X = df.drop(columns=["timestamp", "id", "sample_id"], errors="ignore")
    y = df["group"]  # Use group as target label
    groups = df["group"]

    class_counts = Counter(y)
    with open("reports/class_distribution.txt", "w") as f:
        for label, count in class_counts.items():
            f.write(f"{label}: {count}\n")
    logging.info(f"Class distribution: {class_counts}")

    # Check if we have at least 2 groups
    unique_group_count = len(np.unique(groups))
    if unique_group_count < 2:
        logging.warning(f"Only {unique_group_count} group(s) found with ≥2 samples. Skipping training.")
        logging.info("Switching to unsupervised evaluation (e.g., anomaly detection, clustering).")

        try:
            logging.info("Running cluster_users.py...")
            os.system("python cluster_users.py")

            logging.info("Running detect_anomalies.py...")
            os.system("python detect_anomalies.py")

            logging.info("Unsupervised evaluation completed.")
        except Exception as e:
            logging.error("Error during unsupervised fallback.", exc_info=True)
        exit(0)

    # ========== Model Training ==========
    param_grid = {
        "n_estimators": [100, 200],
        "max_depth": [None, 10, 20],
    }
    base_clf = RandomForestClassifier(class_weight="balanced", random_state=42)
    all_reports = []
    metrics_summary = []

    gkf = GroupKFold(n_splits=min(unique_group_count, 5))
    for i, (train_idx, test_idx) in enumerate(gkf.split(X, y, groups), 1):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        min_class_count = y_train.value_counts().min()
        if min_class_count < 2:
            logging.warning(f"Fold_{i}: Skipping due to insufficient samples in some classes")
            continue

        cv_folds = int(max(2, min(min_class_count, 3)))
        clf = GridSearchCV(base_clf, param_grid, cv=cv_folds, scoring="accuracy", n_jobs=-1)
        clf.fit(X_train, y_train)
        best_model = clf.best_estimator_

        logging.info(f"Fold_{i} Best Params: {clf.best_params_}")

        y_pred = best_model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, average="weighted", zero_division=0)
        rec = recall_score(y_test, y_pred, average="weighted", zero_division=0)
        f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)

        logging.info(f"Fold_{i} - Acc: {acc:.2f}, Prec: {prec:.2f}, Rec: {rec:.2f}, F1: {f1:.2f}")

        report = classification_report(y_test, y_pred, zero_division=0)
        all_reports.append(f"Fold_{i}\n{report}\n")
        metrics_summary.append([f"Fold_{i}", acc, prec, rec, f1])

        cm = confusion_matrix(y_test, y_pred, labels=np.unique(y))
        plt.figure(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                    xticklabels=np.unique(y), yticklabels=np.unique(y))
        plt.title(f"Confusion Matrix - Fold_{i}")
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.tight_layout()
        plt.savefig(f"reports/confusion_matrix_fold_{i}.png")
        plt.close()

    avg_acc = np.mean([m[1] for m in metrics_summary]) if metrics_summary else 0
    logging.info(f"Average Accuracy across folds: {avg_acc:.2f}")

    # ========== Final Model ==========
    final_model = RandomForestClassifier(n_estimators=100, class_weight="balanced", random_state=42)
    final_model.fit(X, y)
    joblib.dump(final_model, "breath_classifier.joblib")
    logging.info("Trained final model and saved to breath_classifier.joblib")

    # ========== Feature Importances ==========
    importances = pd.Series(final_model.feature_importances_, index=X.columns).sort_values(ascending=False)
    plt.figure(figsize=(10, 6))
    sns.barplot(x=importances, y=importances.index)
    plt.title("Feature Importance (Trained on Full Dataset)")
    plt.tight_layout()
    plt.savefig("reports/feature_importance.png")
    plt.close()
    logging.info("Saved feature importance plot")

    # ========== Save Reports ==========
    with open("reports/classification_report.txt", "w") as f:
        f.write("\n".join(all_reports))
        f.write(f"\nAverage Accuracy: {avg_acc:.2f}\n")

    pd.DataFrame(metrics_summary, columns=["Fold", "Accuracy", "Precision", "Recall", "F1"]).to_csv(
        "reports/fold_accuracies.csv", index=False
    )
    logging.info("Saved fold metrics to CSV")

except Exception as e:
    logging.error("Fatal error occurred in training pipeline.", exc_info=True)
    raise
