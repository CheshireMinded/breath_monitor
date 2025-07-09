import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import subprocess
import os

df = pd.read_csv("features.csv")
df = df.dropna()

features = ["mean_breath_duration", "cov_breath_duration", "envelope_variability", "inhale_exhale_ratio"]

# Ensure PNG output directory exists
png_dir = "png"
os.makedirs(png_dir, exist_ok=True)

for feat in features:
    plt.figure(figsize=(8, 5))
    sns.boxplot(x="id", y=feat, data=df, palette="Set3")
    plt.title(f"Boxplot of {feat} by Sample")
    plt.xlabel("Sample ID")
    plt.ylabel(feat)
    plt.tight_layout()

    filename = os.path.join(png_dir, f"boxplot_{feat}.png")
    plt.savefig(filename)
    print(f"Saved boxplot to {filename}")

    try:
        subprocess.run(["xdg-open", filename], check=False)
    except Exception as e:
        print(f"Could not open {filename}: {e}")
