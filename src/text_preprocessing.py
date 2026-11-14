import pandas as pd
import re
import html
import warnings

from pathlib import Path
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

warnings.filterwarnings("ignore")

# ==========================================
# CUSTOMERSENSE AI
# NLP TEXT PREPROCESSING
# ==========================================

print("\n" + "=" * 60)
print("       CUSTOMERSENSE AI - NLP PREPROCESSING")
print("=" * 60)


# ==========================================
# STEP 1: PROJECT PATHS
# ==========================================

ROOT = Path(__file__).resolve().parent.parent

DATA_PATH = ROOT / "data" / "processed" / "eda_reviews.csv"

OUTPUT_PATH = ROOT / "data" / "processed"

OUTPUT_PATH.mkdir(parents=True, exist_ok=True)

SAVE_PATH = OUTPUT_PATH / "nlp_cleaned_reviews.csv"


# ==========================================
# STEP 2: LOAD DATASET
# ==========================================

print("\n[1/7] Loading Dataset...")

if not DATA_PATH.exists():
    raise FileNotFoundError(
        f"Dataset not found: {DATA_PATH}"
    )

df = pd.read_csv(DATA_PATH)

if "text" not in df.columns:
    raise ValueError("Review text column not found.")

print("Dataset Loaded Successfully!")

print("Original Dataset Shape:", df.shape)


# ==========================================
# STEP 3: INITIALIZE NLP TOOLS
# ==========================================

print("\n[2/7] Initializing NLP Tools...")

stemmer = PorterStemmer()

# Preserve negative words because they
# are important for customer feedback.

stop_words = set(ENGLISH_STOP_WORDS)

important_words = {
    "no", "not", "never", "neither",
    "nor", "none", "nobody", "nothing"
}

stop_words = stop_words - important_words

print("NLP Tools Initialized!")


# ==========================================
# STEP 4: TEXT CLEANING FUNCTION
# ==========================================

def clean_text(text):

    if pd.isna(text):
        return ""

    text = str(text)

    # Convert HTML entities
    text = html.unescape(text)

    # Convert lowercase
    text = text.lower()

    # Remove HTML tags
    text = re.sub(r"<[^>]+>", " ", text)

    # Remove URLs
    text = re.sub(
        r"http\S+|www\.\S+",
        " ",
        text
    )

    # Remove email addresses
    text = re.sub(
        r"\S+@\S+",
        " ",
        text
    )

    # Remove numbers
    text = re.sub(r"\d+", " ", text)

    # Remove special characters
    text = re.sub(
        r"[^a-z\s]",
        " ",
        text
    )

    # Remove extra spaces
    text = re.sub(r"\s+", " ", text).strip()

    return text


# ==========================================
# STEP 5: TOKENIZATION AND STOPWORD REMOVAL
# ==========================================

def preprocess_text(text):

    text = clean_text(text)

    # Tokenization
    words = text.split()

    # Stopword removal and stemming
    processed_words = []

    for word in words:

        if word not in stop_words and len(word) > 2:

            stemmed_word = stemmer.stem(word)

            processed_words.append(stemmed_word)

    return " ".join(processed_words)


print("\n[3/7] Text Cleaning Functions Created!")


# ==========================================
# STEP 6: APPLY NLP PREPROCESSING
# ==========================================

print("\n[4/7] Processing Customer Reviews...")

df["text"] = df["text"].fillna("").astype(str)

if "title" in df.columns:

    df["title"] = df["title"].fillna("").astype(str)

    df["combined_text"] = (
        df["title"] + " " + df["text"]
    )

else:

    df["combined_text"] = df["text"]

df["cleaned_text"] = df["combined_text"].apply(
    preprocess_text
)

print("Text Preprocessing Completed!")


# ==========================================
# STEP 7: REMOVE EMPTY REVIEWS AND SAVE
# ==========================================

print("\n[5/7] Removing Empty Reviews...")

df = df[
    df["cleaned_text"].str.strip() != ""
].copy()

print("Remaining Reviews:", len(df))


print("\n[6/7] Comparing Original and Cleaned Text...")

print(
    df[
        ["combined_text", "cleaned_text"]
    ].head(10).to_string(index=False)
)


print("\n[7/7] Saving Processed Dataset...")

df.to_csv(
    SAVE_PATH,
    index=False
)

print("\n" + "=" * 60)

print("NLP PREPROCESSING COMPLETED SUCCESSFULLY!")

print("=" * 60)

print("\nOriginal Dataset:", DATA_PATH)

print("Processed Dataset:", SAVE_PATH)

print("Final Dataset Shape:", df.shape)

print("\nPhase 4 Completed!")