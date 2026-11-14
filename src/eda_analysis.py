
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from pathlib import Path
from collections import Counter
import re
import warnings

warnings.filterwarnings("ignore")

# ============================================
# CUSTOMER SENSE AI - EXPLORATORY DATA ANALYSIS
# ============================================

print("\n" + "=" * 60)
print("       CUSTOMERSENSE AI - EDA ANALYSIS")
print("=" * 60)

# ============================================
# STEP 1: PROJECT PATHS
# ============================================

ROOT = Path(__file__).resolve().parent.parent

DATA_PATH = ROOT / "data" / "processed" / "customer_reviews_sample.csv"

OUTPUT_PATH = ROOT / "outputs"

OUTPUT_PATH.mkdir(parents=True, exist_ok=True)

# ============================================
# STEP 2: LOAD DATASET
# ============================================

print("\n[1/8] Loading Dataset...")

if not DATA_PATH.exists():
    raise FileNotFoundError(
        f"Dataset not found: {DATA_PATH}"
    )

df = pd.read_csv(DATA_PATH)

print("Dataset Loaded Successfully!")

print("\nDataset Shape:", df.shape)

print("\nAvailable Columns:")
print(df.columns.tolist())

# ============================================
# STEP 3: CHECK REVIEW TEXT
# ============================================

print("\n[2/8] Checking Review Text...")

if "text" not in df.columns:
    raise ValueError(
        "The required 'text' column is missing."
    )

df["text"] = df["text"].fillna("").astype(str)

print("Review text column found!")

# ============================================
# STEP 4: MISSING VALUES
# ============================================

print("\n[3/8] Missing Value Analysis")

missing_values = pd.DataFrame({
    "Column": df.columns,
    "Missing_Count": df.isnull().sum().values,
    "Missing_Percentage": (
        df.isnull().mean().values * 100
    ).round(2)
})

print(missing_values)

missing_values.to_csv(
    OUTPUT_PATH / "missing_values_report.csv",
    index=False
)

# ============================================
# STEP 5: DUPLICATE ANALYSIS
# ============================================

print("\n[4/8] Duplicate Analysis")

duplicate_count = df.duplicated().sum()

print("Duplicate Records:", duplicate_count)

df = df.drop_duplicates().copy()

# Remove empty reviews for text analysis
df = df[df["text"].str.strip() != ""].copy()

print("Records After Cleaning:", len(df))

# ============================================
# STEP 6: REVIEW LENGTH ANALYSIS
# ============================================

print("\n[5/8] Review Length Analysis")

df["review_length"] = df["text"].apply(
    lambda x: len(x.split())
)

print(df["review_length"].describe())

plt.figure(figsize=(10, 5))

sns.histplot(
    df["review_length"].clip(upper=300),
    bins=40,
    kde=True
)

plt.title("Customer Review Length Distribution")

plt.xlabel("Number of Words")

plt.ylabel("Number of Reviews")

plt.tight_layout()

plt.savefig(
    OUTPUT_PATH / "review_length_distribution.png",
    dpi=300
)

plt.close()

# ============================================
# STEP 7: CUSTOMER RATING ANALYSIS
# ============================================

print("\n[6/8] Customer Rating Analysis")

if "rating" in df.columns:

    print("\nRating Distribution:")

    print(df["rating"].value_counts().sort_index())

    plt.figure(figsize=(9, 5))

    sns.countplot(
        data=df,
        x="rating",
        order=sorted(df["rating"].dropna().unique())
    )

    plt.title("Customer Rating Distribution")

    plt.xlabel("Customer Rating")

    plt.ylabel("Number of Reviews")

    plt.tight_layout()

    plt.savefig(
        OUTPUT_PATH / "rating_distribution.png",
        dpi=300
    )

    plt.close()

else:
    print("Rating column not available.")

# ============================================
# STEP 8: FREQUENT WORD ANALYSIS
# ============================================

print("\n[7/8] Frequent Word Analysis")

stopwords = {
    "the", "is", "a", "an", "and", "or",
    "to", "of", "in", "for", "on", "with",
    "it", "this", "that", "was", "are",
    "i", "my", "you", "me", "be", "as",
    "at", "but", "have", "has", "had",
    "not", "so", "very", "from", "they",
    "them", "we", "our", "he", "she"
}

all_words = []

for review in df["text"]:

    words = re.findall(
        r"\b[a-zA-Z]{3,}\b",
        review.lower()
    )

    words = [
        word for word in words
        if word not in stopwords
    ]

    all_words.extend(words)

word_counts = Counter(all_words)

top_words = word_counts.most_common(20)

word_df = pd.DataFrame(
    top_words,
    columns=["Word", "Frequency"]
)

print("\nTop 20 Frequently Used Words:")

print(word_df)

word_df.to_csv(
    OUTPUT_PATH / "top_frequent_words.csv",
    index=False
)

# Word frequency chart

plt.figure(figsize=(12, 7))

sns.barplot(
    data=word_df,
    x="Frequency",
    y="Word"
)

plt.title("Top 20 Frequently Used Customer Words")

plt.tight_layout()

plt.savefig(
    OUTPUT_PATH / "frequent_words.png",
    dpi=300
)

plt.close()

# ============================================
# STEP 9: SAVE CLEANED EDA DATASET
# ============================================

print("\n[8/8] Saving EDA Dataset")

df.to_csv(
    ROOT / "data" / "processed" / "eda_reviews.csv",
    index=False
)

print("\nEDA Dataset Saved Successfully!")

# ============================================
# FINAL SUMMARY
# ============================================

print("\n" + "=" * 60)

print("CUSTOMERSENSE AI - EDA COMPLETED!")

print("=" * 60)

print("\nTotal Reviews:", len(df))

print("Total Columns:", len(df.columns))

print("\nGenerated Reports:")

print("1. missing_values_report.csv")
print("2. top_frequent_words.csv")

print("\nGenerated Visualizations:")

print("1. review_length_distribution.png")
print("2. rating_distribution.png (if rating exists)")
print("3. frequent_words.png")

print("\nAll files saved inside outputs/")

print("\nPHASE 3 COMPLETED SUCCESSFULLY!")

print("=" * 60)
