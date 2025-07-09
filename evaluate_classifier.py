(venv) gemini@gemini-VMware-Virtual-Platform:~/breath_monitor$ cat evaluate_classifier.py
import os
import logging
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    classification_report,
    accuracy_score,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
)
from sklearn.model_selection import GroupShuffleSplit

# ========== Setup Logging ==========
os.makedirs("reports", exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("reports/evaluation.log", mode="w"),
        logging.StreamHandler()
    ]
)

try:
    # ========== Load Data ==========
    df = pd.read_csv("features.csv").replace([np.inf, -np.inf], np.nan).dropna()
    logging.info(f"Loaded {len(df)} cleaned samples from features.csv")

    X = df.drop(columns=["timestamp", "id"])
    y = df["id"]
    
    # GROUPS: Must be person-level IDs (not sample IDs)
    if "group" in df.columns:
        groups = df["group"]
        logging.info(f"Using group labels from column 'group'")
    else:
        raise ValueError("Missing 'group' column. Add a 'group' column indicating person/subject identity.")

    # ========== Train/Test Split ==========
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, test_idx = next(gss.split(X, y, groups))

    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    logging.info(f"Split data: {len(X_train)} train / {len(X_test)} test")
    logging.info(f"Test group count: {len(set(groups.iloc[test_idx]))}")

    # ========== Load Model ==========
    clf = joblib.load("breath_classifier.joblib")
    logging.info("Loaded trained model")

    # ========== Predict ==========
    y_pred = clf.predict(X_test)

    # ========== Metrics ==========
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average="weighted", zero_division=0)
    rec = recall_score(y_test, y_pred, average="weighted", zero_division=0)
    f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)

    logging.info(f"Accuracy: {acc:.2f}, Precision: {prec:.2f}, Recall: {rec:.2f}, F1: {f1:.2f}")

    report = classification_report(y_test, y_pred, zero_division=0)
    print("\nClassification Report (Test Set):")
    print(report)

    # ========== Save Report ==========
    with open("reports/classification_report_eval.txt", "w") as f:
        f.write(report)
        f.write(f"\nAccuracy: {acc:.2f}, Precision: {prec:.2f}, Recall: {rec:.2f}, F1: {f1:.2f}\n")

    # ========== Confusion Matrix ==========
    cm_labels = np.unique(np.concatenate([y_test, y_pred]))
    cm = confusion_matrix(y_test, y_pred, labels=cm_labels)

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=cm_labels, yticklabels=cm_labels)
    plt.title("Confusion Matrix (Test Set)")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig("reports/confusion_matrix_eval.png")
    plt.close()
    logging.info("Saved confusion matrix to reports/confusion_matrix_eval.png")

    # ========== Per-Class Accuracy ==========
    class_acc = cm.diagonal() / cm.sum(axis=1)
    plt.figure(figsize=(10, 4))
    sns.barplot(x=cm_labels, y=class_acc)
    plt.ylim(0, 1)
    plt.title("Per-Class Accuracy (Unseen Individuals)")
    plt.ylabel("Accuracy")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig("reports/per_class_accuracy_eval.png")
    plt.close()
    logging.info("Saved per-class accuracy plot to reports/per_class_accuracy_eval.png")

except Exception as e:
    logging.error("Fatal error in evaluation script.", exc_info=True)
    raise
