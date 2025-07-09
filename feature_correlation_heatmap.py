import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

# Load the features
df = pd.read_csv("features.csv")

# Keep only numeric columns (auto-drops strings like 'group', 'id', etc.)
df_numeric = df.select_dtypes(include="number")

# Check if there are numeric features to correlate
if df_numeric.empty:
    print("No numeric features available for correlation heatmap.")
else:
    # Compute correlation matrix
    correlation = df_numeric.corr()

    # Create heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation, annot=True, fmt=".2f", cmap="coolwarm", square=True, linewidths=0.5)
    plt.title("Feature Correlation Heatmap")
    plt.tight_layout()

    # Save to file
    os.makedirs("reports", exist_ok=True)
    plot_path = "reports/feature_correlation_heatmap.png"
    plt.savefig(plot_path)
    print(f"Saved correlation heatmap to {plot_path}")
