import pandas as pd
from pathlib import Path
import gc

# ==========================================
# CUSTOMER SENSE AI
# DATASET LOADING AND SAMPLING
# ==========================================

print("\n" + "=" * 55)
print("       CUSTOMERSENSE AI - DATASET LOADING")
print("=" * 55)


# ==========================================
# STEP 1: PROJECT PATH
# ==========================================

ROOT = Path(__file__).resolve().parent.parent

DATA_PATH = ROOT / "data" / "raw" / "customer_reviews.parquet"

OUTPUT_PATH = ROOT / "data" / "processed"

OUTPUT_PATH.mkdir(parents=True, exist_ok=True)

SAMPLE_PATH = OUTPUT_PATH / "customer_reviews_sample.csv"


# ==========================================
# STEP 2: CHECK DATASET
# ==========================================

print("\n[1/6] Checking dataset...")

if not DATA_PATH.exists():

    raise FileNotFoundError(
        f"Dataset not found: {DATA_PATH}"
    )

print("Dataset found successfully!")


# ==========================================
# STEP 3: READ PARQUET IN BATCHES
# ==========================================

print("\n[2/6] Reading dataset in small batches...")

import pyarrow.parquet as pq

parquet_file = pq.ParquetFile(DATA_PATH)

print("\nAvailable Columns:")

print(parquet_file.schema.names)


# ==========================================
# STEP 4: SELECT REQUIRED COLUMNS
# ==========================================

available_columns = parquet_file.schema.names

possible_columns = [
    "rating",
    "title",
    "text",
    "asin",
    "parent_asin",
    "user_id",
    "timestamp",
    "helpful_vote",
    "verified_purchase"
]

selected_columns = [
    col for col in possible_columns
    if col in available_columns
]

if "text" not in selected_columns:

    raise ValueError(
        "Review text column 'text' was not found. "
        "Check the available columns above."
    )

print("\nSelected Columns:")

print(selected_columns)


# ==========================================
# STEP 5: LOAD ONLY 50,000 REVIEWS
# ==========================================

print("\n[3/6] Loading customer reviews...")

MAX_ROWS = 50000
BATCH_SIZE = 5000

batches = []

total_rows = 0

for batch in parquet_file.iter_batches(
    batch_size=BATCH_SIZE,
    columns=selected_columns
):

    batch_df = batch.to_pandas()

    remaining = MAX_ROWS - total_rows

    if len(batch_df) > remaining:

        batch_df = batch_df.iloc[:remaining]

    batches.append(batch_df)

    total_rows += len(batch_df)

    print(f"Loaded {total_rows:,} reviews")

    del batch
    del batch_df

    gc.collect()

    if total_rows >= MAX_ROWS:
        break


# ==========================================
# STEP 6: COMBINE BATCHES
# ==========================================

print("\n[4/6] Combining batches...")

if not batches:

    raise ValueError("No customer reviews were loaded.")

df = pd.concat(
    batches,
    ignore_index=True
)

del batches

gc.collect()


# ==========================================
# STEP 7: DATASET INFORMATION
# ==========================================

print("\n[5/6] Dataset Information")

print("\nDataset Shape:")

print(df.shape)

print("\nFirst 5 Records:")

print(df.head())

print("\nColumn Names:")

print(df.columns.tolist())

print("\nMissing Values:")

print(df.isnull().sum())

print("\nDuplicate Records:")

print(df.duplicated().sum())


# ==========================================
# STEP 8: SAVE PROCESSED SAMPLE
# ==========================================

print("\n[6/6] Saving customer reviews...")

df.to_csv(
    SAMPLE_PATH,
    index=False
)

print("\nDataset saved successfully!")

print(f"Saved Location: {SAMPLE_PATH}")


# ==========================================
# FINAL RESULT
# ==========================================

print("\n" + "=" * 55)

print("CUSTOMERSENSE AI - DATASET LOADED SUCCESSFULLY!")

print("=" * 55)

print(f"\nTotal Reviews: {len(df):,}")

print(f"Total Columns: {len(df.columns)}")

print("\nNext File:")

print("data/processed/customer_reviews_sample.csv")

print("\nPhase 2 Dataset Loading Completed!")

print("=" * 55)
