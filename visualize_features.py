import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import seaborn as sns
import subprocess
import os

# Load and clean data
df = pd.read_csv("features.csv").dropna()

# Use only numeric columns for PCA
X = df.select_dtypes(include="number")

# Label for coloring the plot
y = df["id"]  # or df["group"] if you want group coloring

# PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

# Plotting
plt.figure(figsize=(8, 6))
sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=y, palette="Set2", s=100)
plt.title("PCA of Breathing Features")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.legend(title="Sample ID", bbox_to_anchor=(1.05, 1), loc="upper left")
plt.grid(True)
plt.tight_layout()

# Save plot
os.makedirs("png", exist_ok=True)
output_file = os.path.join("png", "pca_plot.png")
plt.savefig(output_file)
print(f"Saved PCA plot to {output_file}")

# Optional: open the plot
try:
    subprocess.run(["xdg-open", output_file], check=False)
except Exception as e:
    print("Could not open the plot automatically:", e)
