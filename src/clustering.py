
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from pathlib import Path
from scipy.sparse import load_npz
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import silhouette_score
import warnings

warnings.filterwarnings("ignore")

# ============================================
# CUSTOMERSENSE AI - K-MEANS CLUSTERING
# ============================================

print("\n" + "=" * 60)
print("       CUSTOMERSENSE AI - K-MEANS CLUSTERING")
print("=" * 60)


# ============================================
# STEP 1: PROJECT PATHS
# ============================================

ROOT = Path(__file__).resolve().parent.parent

DATA_PATH = (
    ROOT / "data" / "processed" /
    "nlp_cleaned_reviews.csv"
)

MATRIX_PATH = ROOT / "outputs" / "tfidf_matrix.npz"

MODEL_PATH = ROOT / "models"

OUTPUT_PATH = ROOT / "outputs"

MODEL_PATH.mkdir(parents=True, exist_ok=True)
OUTPUT_PATH.mkdir(parents=True, exist_ok=True)

KMEANS_PATH = MODEL_PATH / "kmeans_model.pkl"

CLUSTERED_PATH = OUTPUT_PATH / "clustered_reviews.csv"

METRICS_PATH = OUTPUT_PATH / "clustering_metrics.csv"

KEYWORDS_PATH = OUTPUT_PATH / "cluster_keywords.csv"

ELBOW_PATH = OUTPUT_PATH / "elbow_method.png"

SILHOUETTE_PATH = OUTPUT_PATH / "silhouette_analysis.png"


# ============================================
# STEP 2: LOAD DATA
# ============================================

print("\n[1/7] Loading Dataset...")

if not DATA_PATH.exists():
    raise FileNotFoundError(DATA_PATH)

if not MATRIX_PATH.exists():
    raise FileNotFoundError(MATRIX_PATH)

df = pd.read_csv(DATA_PATH)

df["cleaned_text"] = (
    df["cleaned_text"]
    .fillna("")
    .astype(str)
)

df = df[
    df["cleaned_text"].str.strip() != ""
].copy()

df.reset_index(drop=True, inplace=True)

X = load_npz(MATRIX_PATH)

print("Dataset Loaded Successfully!")

print("Reviews:", len(df))
print("TF-IDF Matrix Shape:", X.shape)

if X.shape[0] != len(df):
    raise ValueError(
        "Dataset and TF-IDF matrix row counts do not match. "
        "Rerun feature_extraction.py before clustering."
    )


# ============================================
# STEP 3: FIND OPTIMAL CLUSTERS
# ============================================

print("\n[2/7] Finding Optimal Number of Clusters...")

k_values = range(2, 9)

inertia_values = []
silhouette_values = []

# Use a reproducible sample for silhouette evaluation.
sample_size = min(2000, X.shape[0])

rng = np.random.default_rng(42)

sample_indices = rng.choice(
    X.shape[0],
    size=sample_size,
    replace=False
)

X_sample = X[sample_indices]

for k in k_values:

    print(f"\nTesting K = {k}")

    model = MiniBatchKMeans(
        n_clusters=k,
        random_state=42,
        batch_size=1024,
        n_init=3,
        max_iter=100
    )

    model.fit(X)

    inertia_values.append(model.inertia_)

    sample_labels = model.predict(X_sample)

    score = silhouette_score(
        X_sample,
        sample_labels,
        metric="cosine"
    )

    silhouette_values.append(score)

    print("Inertia:", round(model.inertia_, 3))
    print("Silhouette Score:", round(score, 4))


# ============================================
# STEP 4: SAVE EVALUATION RESULTS
# ============================================

print("\n[3/7] Saving Clustering Metrics...")

metrics_df = pd.DataFrame({
    "K": list(k_values),
    "Inertia": inertia_values,
    "Silhouette_Score": silhouette_values
})

metrics_df.to_csv(
    METRICS_PATH,
    index=False
)

best_k = int(
    metrics_df.loc[
        metrics_df["Silhouette_Score"].idxmax(),
        "K"
    ]
)

print("\nBest K based on sampled Silhouette Score:", best_k)


# ============================================
# STEP 5: ELBOW METHOD VISUALIZATION
# ============================================

print("\n[4/7] Creating Elbow Method Graph...")

plt.figure(figsize=(9, 5))

plt.plot(
    list(k_values),
    inertia_values,
    marker="o"
)

plt.title("Elbow Method - Optimal K")

plt.xlabel("Number of Clusters (K)")

plt.ylabel("Inertia")

plt.xticks(list(k_values))

plt.grid(alpha=0.3)

plt.tight_layout()

plt.savefig(
    ELBOW_PATH,
    dpi=300
)

plt.close()


# ============================================
# STEP 6: SILHOUETTE VISUALIZATION
# ============================================

plt.figure(figsize=(9, 5))

plt.plot(
    list(k_values),
    silhouette_values,
    marker="o"
)

plt.title("Silhouette Score Analysis")

plt.xlabel("Number of Clusters (K)")

plt.ylabel("Silhouette Score")

plt.xticks(list(k_values))

plt.grid(alpha=0.3)

plt.tight_layout()

plt.savefig(
    SILHOUETTE_PATH,
    dpi=300
)

plt.close()


# ============================================
# STEP 7: TRAIN FINAL MODEL
# ============================================

print("\n[5/7] Training Final K-Means Model...")

FINAL_K = best_k

kmeans = MiniBatchKMeans(
    n_clusters=FINAL_K,
    random_state=42,
    batch_size=1024,
    n_init=5,
    max_iter=150
)

df["cluster"] = kmeans.fit_predict(X)

joblib.dump(
    kmeans,
    KMEANS_PATH
)

print("Final Model Trained Successfully!")

print("Selected Clusters:", FINAL_K)


# ============================================
# STEP 8: IDENTIFY CLUSTER KEYWORDS
# ============================================

print("\n[6/7] Identifying Cluster Keywords...")

vectorizer = joblib.load(
    MODEL_PATH / "tfidf_vectorizer.pkl"
)

feature_names = vectorizer.get_feature_names_out()

centers = kmeans.cluster_centers_

keyword_records = []

for cluster_id in range(FINAL_K):

    top_indices = np.argsort(
        centers[cluster_id]
    )[-15:][::-1]

    keywords = [
        feature_names[index]
        for index in top_indices
    ]

    keyword_records.append({
        "Cluster": cluster_id,
        "Keywords": ", ".join(keywords)
    })

    print(
        f"\nCluster {cluster_id}:",
        ", ".join(keywords)
    )

keyword_df = pd.DataFrame(keyword_records)

keyword_df.to_csv(
    KEYWORDS_PATH,
    index=False
)


# ============================================
# STEP 9: CLUSTER DISTRIBUTION
# ============================================

print("\n[7/7] Analyzing Cluster Distribution...")

cluster_counts = df["cluster"].value_counts().sort_index()

print("\nCluster Distribution:")

print(cluster_counts)

plt.figure(figsize=(9, 5))

sns.barplot(
    x=cluster_counts.index,
    y=cluster_counts.values
)

plt.title("Customer Feedback Cluster Distribution")

plt.xlabel("Cluster")

plt.ylabel("Number of Reviews")

plt.tight_layout()

plt.savefig(
    OUTPUT_PATH / "cluster_distribution.png",
    dpi=300
)

plt.close()


# ============================================
# STEP 10: SAVE CLUSTERED REVIEWS
# ============================================

df.to_csv(
    CLUSTERED_PATH,
    index=False
)


# ============================================
# FINAL SUMMARY
# ============================================

print("\n" + "=" * 60)

print("K-MEANS CLUSTERING COMPLETED SUCCESSFULLY!")

print("=" * 60)

print("\nFinal Clusters:", FINAL_K)

print("Total Reviews:", len(df))

print("\nGenerated Files:")

print("1. models/kmeans_model.pkl")

print("2. outputs/clustered_reviews.csv")

print("3. outputs/clustering_metrics.csv")

print("4. outputs/cluster_keywords.csv")

print("5. outputs/elbow_method.png")

print("6. outputs/silhouette_analysis.png")

print("7. outputs/cluster_distribution.png")

print("\nPHASE 6 COMPLETED!")

print("=" * 60)
