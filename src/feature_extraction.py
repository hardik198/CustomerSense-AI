import pandas as pd
import numpy as np
import joblib

from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import save_npz
import warnings

warnings.filterwarnings("ignore")

# ==================================================
# CUSTOMERSENSE AI - TF-IDF FEATURE EXTRACTION
# ==================================================

print("\n" + "=" * 60)
print("       CUSTOMERSENSE AI - TF-IDF VECTORIZATION")
print("=" * 60)


# ==================================================
# STEP 1: PROJECT PATHS
# ==================================================

ROOT = Path(__file__).resolve().parent.parent

DATA_PATH = (
    ROOT / "data" / "processed" /
    "nlp_cleaned_reviews.csv"
)

MODEL_PATH = ROOT / "models"

OUTPUT_PATH = ROOT / "outputs"

MODEL_PATH.mkdir(parents=True, exist_ok=True)

OUTPUT_PATH.mkdir(parents=True, exist_ok=True)

VECTOR_PATH = MODEL_PATH / "tfidf_vectorizer.pkl"

MATRIX_PATH = OUTPUT_PATH / "tfidf_matrix.npz"

FEATURE_PATH = OUTPUT_PATH / "tfidf_features.csv"


# ==================================================
# STEP 2: LOAD PREPROCESSED DATASET
# ==================================================

print("\n[1/6] Loading NLP Dataset...")

if not DATA_PATH.exists():

    raise FileNotFoundError(
        f"Dataset not found: {DATA_PATH}"
    )

df = pd.read_csv(DATA_PATH)

if "cleaned_text" not in df.columns:

    raise ValueError(
        "cleaned_text column not found."
    )

df["cleaned_text"] = (
    df["cleaned_text"]
    .fillna("")
    .astype(str)
)

# Remove empty text

df = df[
    df["cleaned_text"].str.strip() != ""
].copy()

df.reset_index(drop=True, inplace=True)

print("Dataset Loaded Successfully!")

print("Total Reviews:", len(df))


# ==================================================
# STEP 3: INITIALIZE TF-IDF
# ==================================================

print("\n[2/6] Initializing TF-IDF Vectorizer...")

tfidf = TfidfVectorizer(

    max_features=5000,

    min_df=5,

    max_df=0.95,

    ngram_range=(1, 2),

    sublinear_tf=True,

    dtype=np.float32

)

print("TF-IDF Vectorizer Initialized!")


# ==================================================
# STEP 4: CONVERT TEXT INTO NUMERICAL FEATURES
# ==================================================

print("\n[3/6] Performing TF-IDF Vectorization...")

tfidf_matrix = tfidf.fit_transform(
    df["cleaned_text"]
)

print("TF-IDF Vectorization Completed!")


# ==================================================
# STEP 5: DISPLAY MATRIX INFORMATION
# ==================================================

print("\n[4/6] TF-IDF Matrix Information")

print("Matrix Shape:", tfidf_matrix.shape)

print("Total Reviews:", tfidf_matrix.shape[0])

print("Total Features:", tfidf_matrix.shape[1])

print(
    "Non-zero Values:",
    tfidf_matrix.nnz
)

total_values = (
    tfidf_matrix.shape[0] *
    tfidf_matrix.shape[1]
)

sparsity = (
    1 - tfidf_matrix.nnz / total_values
) * 100

print("Matrix Sparsity:", round(sparsity, 2), "%")


# ==================================================
# STEP 6: SAVE TF-IDF MODEL AND MATRIX
# ==================================================

print("\n[5/6] Saving TF-IDF Model...")

joblib.dump(
    tfidf,
    VECTOR_PATH
)

print("TF-IDF Vectorizer Saved!")

print("\nSaving TF-IDF Matrix...")

save_npz(
    MATRIX_PATH,
    tfidf_matrix
)

print("TF-IDF Matrix Saved!")


# ==================================================
# STEP 7: SAVE FEATURE NAMES
# ==================================================

print("\n[6/6] Saving Feature Names...")

feature_names = tfidf.get_feature_names_out()

feature_df = pd.DataFrame({
    "Feature": feature_names
})

feature_df.to_csv(
    FEATURE_PATH,
    index=False
)

print("Feature Names Saved!")


# ==================================================
# FINAL SUMMARY
# ==================================================

print("\n" + "=" * 60)

print("CUSTOMERSENSE AI - TF-IDF COMPLETED!")

print("=" * 60)

print("\nGenerated Files:")

print("1. models/tfidf_vectorizer.pkl")

print("2. outputs/tfidf_matrix.npz")

print("3. outputs/tfidf_features.csv")

print("\nFinal Matrix Shape:", tfidf_matrix.shape)

print("\nPHASE 5 COMPLETED SUCCESSFULLY!")

print("=" * 60)