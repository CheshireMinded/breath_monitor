import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import os
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("reports/feature_importances.log"),
        logging.StreamHandler()
    ]
)

# Load data and model
df = pd.read_csv("features.csv")
clf = joblib.load("breath_classifier.joblib")
logging.info("Loaded features.csv and breath_classifier.joblib")

# Clean data
df = df.dropna()
X = df.drop(columns=["timestamp", "id"])

# Extract feature importances
importances = clf.feature_importances_
feature_names = X.columns

# Sort
sorted_idx = importances.argsort()[::-1]
sorted_importances = importances[sorted_idx]
sorted_features = feature_names[sorted_idx]

# Plot
plt.figure(figsize=(10, 6))
sns.barplot(x=sorted_importances, y=sorted_features, palette="viridis")
plt.title("Feature Importances from Random Forest")
plt.xlabel("Importance")
plt.ylabel("Feature")
plt.tight_layout()

# Save
os.makedirs("reports", exist_ok=True)
output_path = "reports/feature_importances.png"
plt.savefig(output_path)
logging.info(f"Saved feature importances plot to {output_path}")
