import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score
import argparse
import os

# Load and clean data
df = pd.read_csv("features.csv").dropna()

# Keep labels before removing them from features
labels = df["id"]

# Select only numeric features for clustering
features = df.select_dtypes(include="number")

# Standardize numeric features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(features)

# Argument parser
parser = argparse.ArgumentParser()
parser.add_argument("--method", choices=["pca", "tsne"], default="pca", help="Dimensionality reduction method")
parser.add_argument("--clusters", type=int, default=3, help="Number of clusters for KMeans")
args = parser.parse_args()

# Dimensionality reduction
if args.method == "pca":
    reducer = PCA(n_components=2)
    X_reduced = reducer.fit_transform(X_scaled)
    title = "PCA Clustering of Users"
else:
    reducer = TSNE(n_components=2, perplexity=30, random_state=42)
    X_reduced = reducer.fit_transform(X_scaled)
    title = "t-SNE Clustering of Users"

# KMeans clustering
kmeans = KMeans(n_clusters=args.clusters, random_state=42)
clusters = kmeans.fit_predict(X_reduced)

# Add cluster info to dataframe
df["cluster"] = clusters
os.makedirs("reports", exist_ok=True)
df.to_csv("reports/features_with_clusters.csv", index=False)
print("Saved updated features with cluster assignments to reports/features_with_clusters.csv")

# Clustering quality metrics
sil_score = silhouette_score(X_reduced, clusters)
db_score = davies_bouldin_score(X_reduced, clusters)
print(f"Silhouette Score: {sil_score:.3f}")
print(f"Davies-Bouldin Score: {db_score:.3f}")

# Plotting
plt.figure(figsize=(10, 7))
sns.scatterplot(x=X_reduced[:, 0], y=X_reduced[:, 1], hue=clusters, palette="tab10", s=70, alpha=0.8)
for i, label in enumerate(labels):
    plt.text(X_reduced[i, 0], X_reduced[i, 1], str(label), fontsize=8, alpha=0.6)

plt.title(f"{title}\nSilhouette: {sil_score:.2f} | Davies-Bouldin: {db_score:.2f}")
plt.xlabel("Component 1")
plt.ylabel("Component 2")
plt.legend(title="Cluster")
plt.tight_layout()

plot_path = f"reports/clusters_{args.method}.png"
plt.savefig(plot_path)
print(f"Saved clustering plot to {plot_path}")
