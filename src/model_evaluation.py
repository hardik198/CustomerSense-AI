
import pandas as pd
import numpy as np
import joblib

from pathlib import Path
from scipy.sparse import load_npz
from sklearn.metrics import silhouette_score

import warnings
warnings.filterwarnings("ignore")


# ============================================
# CUSTOMERSENSE AI - MODEL EVALUATION
# ============================================

print("\n" + "=" * 55)
print("       CUSTOMERSENSE AI - MODEL EVALUATION")
print("=" * 55)


# ============================================
# STEP 1: PROJECT PATHS
# ============================================

ROOT = Path(__file__).resolve().parent.parent

DATA_PATH = ROOT / "outputs" / "clustered_reviews.csv"

MATRIX_PATH = ROOT / "outputs" / "tfidf_matrix.npz"

MODEL_PATH = ROOT / "models" / "kmeans_model.pkl"

OUTPUT_PATH = ROOT / "outputs"

OUTPUT_PATH.mkdir(parents=True, exist_ok=True)


# ============================================
# STEP 2: LOAD FILES
# ============================================

print("\n[1/5] Loading Model and Dataset...")

df = pd.read_csv(DATA_PATH)

X = load_npz(MATRIX_PATH)

model = joblib.load(MODEL_PATH)

if len(df) != X.shape[0]:
    raise ValueError("Review data and TF-IDF matrix sizes do not match.")

if "cluster" not in df.columns:
    raise ValueError("Cluster column is missing.")

labels = df["cluster"].to_numpy()

print("Dataset Loaded Successfully!")
print("Total Reviews:", len(df))
print("Total Clusters:", len(np.unique(labels)))


# ============================================
# STEP 3: SILHOUETTE SCORE
# ============================================

print("\n[2/5] Calculating Silhouette Score...")

sample_size = min(2000, X.shape[0])

rng = np.random.default_rng(42)

indices = rng.choice(
    X.shape[0],
    size=sample_size,
    replace=False
)

X_sample = X[indices]
labels_sample = labels[indices]

if len(np.unique(labels_sample)) < 2:
    raise ValueError(
        "The evaluation sample contains fewer than two clusters. "
        "Try increasing the sample size."
    )

silhouette = silhouette_score(
    X_sample,
    labels_sample,
    metric="cosine"
)

print("Silhouette Score:", round(silhouette, 4))


# ============================================
# STEP 4: CLUSTER DISTRIBUTION
# ============================================

print("\n[3/5] Analyzing Cluster Distribution...")

cluster_counts = (
    df["cluster"]
    .value_counts()
    .sort_index()
)

cluster_distribution = pd.DataFrame({
    "Cluster": cluster_counts.index,
    "Review_Count": cluster_counts.values
})

cluster_distribution["Percentage"] = (
    cluster_distribution["Review_Count"]
    / len(df) * 100
).round(2)

cluster_distribution.to_csv(
    OUTPUT_PATH / "cluster_evaluation.csv",
    index=False
)

print("\nCluster Distribution:")
print(cluster_distribution.to_string(index=False))


# ============================================
# STEP 5: MODEL METRICS
# ============================================

print("\n[4/5] Calculating Final Model Metrics...")

inertia = model.inertia_

total_reviews = len(df)

number_of_clusters = len(cluster_counts)

largest_cluster = cluster_counts.max()

smallest_cluster = cluster_counts.min()

largest_cluster_percentage = (
    largest_cluster / total_reviews
) * 100

smallest_cluster_percentage = (
    smallest_cluster / total_reviews
) * 100

metrics = pd.DataFrame({
    "Metric": [
        "Total Reviews",
        "Number of Clusters",
        "Silhouette Score (Sampled)",
        "Model Inertia",
        "Largest Cluster Size",
        "Smallest Cluster Size",
        "Largest Cluster Percentage",
        "Smallest Cluster Percentage"
    ],

    "Value": [
        total_reviews,
        number_of_clusters,
        round(silhouette, 4),
        round(inertia, 3),
        int(largest_cluster),
        int(smallest_cluster),
        round(largest_cluster_percentage, 2),
        round(smallest_cluster_percentage, 2)
    ]
})

metrics.to_csv(
    OUTPUT_PATH / "model_evaluation_metrics.csv",
    index=False
)


# ============================================
# STEP 6: FINAL REPORT
# ============================================

print("\n[5/5] Saving Final Evaluation Report...")

print("\n" + "=" * 55)
print("             FINAL MODEL EVALUATION")
print("=" * 55)

print("Total Reviews:", total_reviews)

print("Number of Clusters:", number_of_clusters)

print("Silhouette Score:", round(silhouette, 4))

print("Model Inertia:", round(inertia, 3))

print("\nEvaluation Files Generated:")

print("1. outputs/cluster_evaluation.csv")

print("2. outputs/model_evaluation_metrics.csv")

print("\nMODEL EVALUATION COMPLETED SUCCESSFULLY!")

print("=" * 55)
