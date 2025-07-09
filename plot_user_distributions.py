import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("reports/user_distributions.log"),
        logging.StreamHandler()
    ]
)

# Load and clean data
df = pd.read_csv("features.csv")
df = df.dropna()
if "id" not in df.columns:
    logging.error("'id' column not found in features.csv. Exiting.")
    exit(1)

# Select features to visualize (excluding metadata columns)
exclude = {"timestamp", "id"}
feature_columns = [col for col in df.columns if col not in exclude]

# Output directory
os.makedirs("reports/user_feature_distributions", exist_ok=True)

# Plot per-feature distribution by user ID
for feature in feature_columns:
    plt.figure(figsize=(8, 5))
    sns.boxplot(x="id", y=feature, data=df)
    plt.title(f"Distribution of '{feature}' by User")
    plt.xlabel("User ID")
    plt.ylabel(feature)
    plt.xticks(rotation=45)
    plt.tight_layout()

    out_path = f"reports/user_feature_distributions/{feature}_by_user.png"
    plt.savefig(out_path)
    logging.info(f"Saved distribution plot for '{feature}' to {out_path}")
    plt.close()

logging.info("All user distribution plots saved.")
