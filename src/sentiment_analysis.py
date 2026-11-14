
import pandas as pd
import matplotlib.pyplot as plt
import nltk

from pathlib import Path
from nltk.sentiment import SentimentIntensityAnalyzer

import warnings
warnings.filterwarnings("ignore")


# ============================================
# CUSTOMERSENSE AI - SENTIMENT ANALYSIS
# ============================================

print("\n" + "=" * 55)
print("       CUSTOMERSENSE AI - SENTIMENT ANALYSIS")
print("=" * 55)


# ============================================
# STEP 1: PROJECT PATHS
# ============================================

ROOT = Path(__file__).resolve().parent.parent

DATA_PATH = ROOT / "outputs" / "clustered_reviews.csv"

OUTPUT_PATH = ROOT / "outputs"

OUTPUT_PATH.mkdir(parents=True, exist_ok=True)

REVIEW_OUTPUT = OUTPUT_PATH / "sentiment_reviews.csv"

SUMMARY_OUTPUT = OUTPUT_PATH / "sentiment_summary.csv"

CLUSTER_OUTPUT = OUTPUT_PATH / "sentiment_by_cluster.csv"

CHART_OUTPUT = OUTPUT_PATH / "sentiment_distribution.png"


# ============================================
# STEP 2: LOAD DATASET
# ============================================

print("\n[1/6] Loading Customer Reviews...")

if not DATA_PATH.exists():
    raise FileNotFoundError(
        "clustered_reviews.csv not found. "
        "Run clustering.py first."
    )

df = pd.read_csv(DATA_PATH)

if "cluster" not in df.columns:
    raise ValueError("Cluster column is missing.")

# Prefer original review text over cleaned/stemmed text.
if "text" in df.columns:
    text_column = "text"
elif "cleaned_text" in df.columns:
    text_column = "cleaned_text"
else:
    raise ValueError("No review text column found.")

df[text_column] = df[text_column].fillna("").astype(str)

df = df[df[text_column].str.strip() != ""].copy()

df.reset_index(drop=True, inplace=True)

print("Reviews Loaded:", len(df))


# ============================================
# STEP 3: DOWNLOAD VADER LEXICON
# ============================================

print("\n[2/6] Loading VADER Sentiment Dictionary...")

try:
    nltk.download("vader_lexicon", quiet=True)

    sia = SentimentIntensityAnalyzer()

except Exception as error:
    raise RuntimeError(
        "Unable to load VADER lexicon. "
        "Check your internet connection and try again."
    ) from error

print("VADER Loaded Successfully!")


# ============================================
# STEP 4: CALCULATE SENTIMENT
# ============================================

print("\n[3/6] Analyzing Customer Sentiment...")

scores = df[text_column].apply(sia.polarity_scores)

df["positive_score"] = scores.apply(
    lambda x: x["pos"]
)

df["negative_score"] = scores.apply(
    lambda x: x["neg"]
)

df["neutral_score"] = scores.apply(
    lambda x: x["neu"]
)

df["compound_score"] = scores.apply(
    lambda x: x["compound"]
)


# ============================================
# STEP 5: CLASSIFY SENTIMENT
# ============================================

def classify_sentiment(score):

    if score >= 0.05:
        return "Positive"

    elif score <= -0.05:
        return "Negative"

    else:
        return "Neutral"


df["sentiment"] = df["compound_score"].apply(
    classify_sentiment
)

print("Sentiment Analysis Completed!")


# ============================================
# STEP 6: OVERALL SENTIMENT SUMMARY
# ============================================

print("\n[4/6] Generating Sentiment Summary...")

sentiment_counts = (
    df["sentiment"]
    .value_counts()
    .reindex(
        ["Positive", "Negative", "Neutral"],
        fill_value=0
    )
)

summary_df = pd.DataFrame({
    "Sentiment": sentiment_counts.index,
    "Review_Count": sentiment_counts.values
})

summary_df["Percentage"] = (
    summary_df["Review_Count"] / len(df) * 100
).round(2)

summary_df.to_csv(
    SUMMARY_OUTPUT,
    index=False
)

print("\nOverall Sentiment:")
print(summary_df.to_string(index=False))


# ============================================
# STEP 7: SENTIMENT BY CLUSTER
# ============================================

print("\n[5/6] Analyzing Sentiment by Cluster...")

cluster_summary = pd.crosstab(
    df["cluster"],
    df["sentiment"]
).reindex(
    columns=["Positive", "Negative", "Neutral"],
    fill_value=0
)

cluster_summary["Total_Reviews"] = (
    cluster_summary.sum(axis=1)
)

cluster_summary.to_csv(
    CLUSTER_OUTPUT
)

print("\nSentiment by Cluster:")
print(cluster_summary)


# ============================================
# STEP 8: VISUALIZATION
# ============================================

print("\n[6/6] Creating Sentiment Chart...")

plt.figure(figsize=(9, 5))

plt.bar(
    summary_df["Sentiment"],
    summary_df["Review_Count"]
)

plt.title("Customer Sentiment Distribution")

plt.xlabel("Sentiment")

plt.ylabel("Number of Reviews")

plt.tight_layout()

plt.savefig(
    CHART_OUTPUT,
    dpi=300
)

plt.close()


# ============================================
# STEP 9: SAVE FINAL DATASET
# ============================================

df.to_csv(
    REVIEW_OUTPUT,
    index=False
)


# ============================================
# FINAL SUMMARY
# ============================================

print("\n" + "=" * 55)

print("SENTIMENT ANALYSIS COMPLETED SUCCESSFULLY!")

print("=" * 55)

print("\nTotal Reviews:", len(df))

print("Positive:", sentiment_counts["Positive"])

print("Negative:", sentiment_counts["Negative"])

print("Neutral:", sentiment_counts["Neutral"])

print("\nGenerated Files:")

print("1. outputs/sentiment_reviews.csv")

print("2. outputs/sentiment_summary.csv")

print("3. outputs/sentiment_by_cluster.csv")

print("4. outputs/sentiment_distribution.png")

print("\nPHASE 8 COMPLETED!")

print("=" * 55)
