import os
import json
import numpy as np
import pandas as pd

DATA_PATH = r"T_ONTIME_REPORTING.csv"

RESULTS_DIR = "results"
MODELS_DIR = "models"

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)


def hhmm_to_minutes(value):
    """Convert HHMM scheduled time to minutes since midnight."""
    try:
        if pd.isna(value):
            return np.nan

        value = int(value)
        hours = value // 100
        minutes = value % 100

        if hours == 24:
            hours = 0

        if minutes >= 60:
            return np.nan

        return hours * 60 + minutes

    except Exception:
        return np.nan


print("Loading dataset...")

df = pd.read_csv(
    DATA_PATH,
    low_memory=False
)

print(f"Original dataset shape: {df.shape}")
print(f"Columns available: {len(df.columns)}")


# ============================================================
# 1. KEEP ONLY FLIGHTS WITH TARGET AVAILABLE
# ============================================================

if "DEP_DEL15" not in df.columns:
    raise ValueError("DEP_DEL15 target column not found!")

df = df[df["DEP_DEL15"].notna()].copy()

df["DEP_DEL15"] = df["DEP_DEL15"].astype(int)


# ============================================================
# 2. CREATE DATE
# ============================================================

date_columns = ["YEAR", "MONTH", "DAY_OF_MONTH"]

for col in date_columns:
    if col not in df.columns:
        raise ValueError(f"Required column missing: {col}")

df["FL_DATE"] = pd.to_datetime(
    {
        "year": df["YEAR"],
        "month": df["MONTH"],
        "day": df["DAY_OF_MONTH"]
    },
    errors="coerce"
)

df = df[df["FL_DATE"].notna()].copy()


# ============================================================
# 3. TEMPORAL FEATURES
# ============================================================

df["DEP_MINUTES"] = df["CRS_DEP_TIME"].apply(hhmm_to_minutes)
df["ARR_MINUTES"] = df["CRS_ARR_TIME"].apply(hhmm_to_minutes)

df["DEP_HOUR"] = df["DEP_MINUTES"] / 60
df["ARR_HOUR"] = df["ARR_MINUTES"] / 60

df["DEP_HOUR_SIN"] = np.sin(
    2 * np.pi * df["DEP_HOUR"] / 24
)

df["DEP_HOUR_COS"] = np.cos(
    2 * np.pi * df["DEP_HOUR"] / 24
)

df["DOW_SIN"] = np.sin(
    2 * np.pi * df["DAY_OF_WEEK"] / 7
)

df["DOW_COS"] = np.cos(
    2 * np.pi * df["DAY_OF_WEEK"] / 7
)

df["IS_WEEKEND"] = (
    df["DAY_OF_WEEK"].isin([6, 7])
).astype(int)


def time_bucket(minutes):
    if pd.isna(minutes):
        return "unknown"

    hour = minutes / 60

    if hour < 6:
        return "early_morning"
    elif hour < 10:
        return "morning_rush"
    elif hour < 16:
        return "midday"
    elif hour < 20:
        return "evening_rush"
    else:
        return "late_night"


df["TIME_BUCKET"] = df["DEP_MINUTES"].apply(time_bucket)

time_dummies = pd.get_dummies(
    df["TIME_BUCKET"],
    prefix="TIME",
    dtype=int
)

df = pd.concat(
    [df, time_dummies],
    axis=1
)


# ============================================================
# 4. SCHEDULED BLOCK SPEED
# ============================================================

if (
    "DISTANCE" in df.columns
    and "CRS_ELAPSED_TIME" in df.columns
):
    df["SCHEDULED_BLOCK_SPEED"] = (
        df["DISTANCE"] /
        (df["CRS_ELAPSED_TIME"] + 1e-5)
    )


# ============================================================
# 5. AIRPORT CONGESTION PROXIES
# ============================================================

df["DEP_HOUR_INT"] = (
    df["DEP_MINUTES"] // 60
).fillna(-1).astype(int)


if "ORIGIN" in df.columns:

    df["ORIGIN_HOURLY_DEPARTURES"] = (
        df.groupby(
            ["FL_DATE", "ORIGIN", "DEP_HOUR_INT"]
        )["DEP_DEL15"]
        .transform("size")
    )

    df["ORIGIN_DAILY_DEPARTURES"] = (
        df.groupby(
            ["FL_DATE", "ORIGIN"]
        )["DEP_DEL15"]
        .transform("size")
    )


if "DEST" in df.columns:

    df["DEST_DAILY_ARRIVALS"] = (
        df.groupby(
            ["FL_DATE", "DEST"]
        )["DEP_DEL15"]
        .transform("size")
    )


# ============================================================
# 6. CARRIER MARKET SHARE
# ============================================================

CARRIER_COL = None

if "OP_UNIQUE_CARRIER" in df.columns:
    CARRIER_COL = "OP_UNIQUE_CARRIER"

elif "OP_CARRIER" in df.columns:
    CARRIER_COL = "OP_CARRIER"


if (
    CARRIER_COL is not None
    and "ORIGIN" in df.columns
):

    carrier_count = (
        df.groupby(
            ["FL_DATE", "ORIGIN", CARRIER_COL]
        )["DEP_DEL15"]
        .transform("size")
    )

    origin_count = (
        df.groupby(
            ["FL_DATE", "ORIGIN"]
        )["DEP_DEL15"]
        .transform("size")
    )

    df["CARRIER_ORIGIN_MARKET_SHARE"] = (
        carrier_count / origin_count
    )


# ============================================================
# 7. CHRONOLOGICAL SPLIT
# ============================================================

df = df.sort_values("FL_DATE").reset_index(drop=True)

unique_dates = sorted(df["FL_DATE"].unique())

n_dates = len(unique_dates)

train_end = int(n_dates * 0.70)
val_end = int(n_dates * 0.85)

train_dates = unique_dates[:train_end]
val_dates = unique_dates[train_end:val_end]
test_dates = unique_dates[val_end:]


train_df = df[
    df["FL_DATE"].isin(train_dates)
].copy()

val_df = df[
    df["FL_DATE"].isin(val_dates)
].copy()

test_df = df[
    df["FL_DATE"].isin(test_dates)
].copy()


print("\nChronological split:")
print(f"Train: {train_df.shape}")
print(f"Validation: {val_df.shape}")
print(f"Test: {test_df.shape}")


# ============================================================
# 8. BAYESIAN HISTORICAL TARGET ENCODING
# IMPORTANT: TRAIN DATA STATISTICS ONLY
# ============================================================

GLOBAL_MEAN = train_df["DEP_DEL15"].mean()
SMOOTHING = 10


def create_smoothed_encoding(
    train,
    validation,
    test,
    column,
    feature_name
):

    stats = train.groupby(
        column
    )["DEP_DEL15"].agg(
        ["mean", "count"]
    )

    smoothed = (
        (
            stats["mean"] * stats["count"]
            + GLOBAL_MEAN * SMOOTHING
        )
        /
        (
            stats["count"] + SMOOTHING
        )
    )

    train[feature_name] = (
        train[column]
        .map(smoothed)
        .fillna(GLOBAL_MEAN)
    )

    validation[feature_name] = (
        validation[column]
        .map(smoothed)
        .fillna(GLOBAL_MEAN)
    )

    test[feature_name] = (
        test[column]
        .map(smoothed)
        .fillna(GLOBAL_MEAN)
    )

    return {
        str(key): float(value)
        for key, value in smoothed.items()
    }


historical_stats = {}

encoding_columns = []


if CARRIER_COL is not None:

    encoding_columns.append(
        (
            CARRIER_COL,
            "CARRIER_DELAY_RATE"
        )
    )


if "ORIGIN" in df.columns:

    encoding_columns.append(
        (
            "ORIGIN",
            "ORIGIN_DELAY_RATE"
        )
    )


if "DEST" in df.columns:

    encoding_columns.append(
        (
            "DEST",
            "DEST_DELAY_RATE"
        )
    )


# Create route
if (
    "ORIGIN" in df.columns
    and "DEST" in df.columns
):

    for split in [
        train_df,
        val_df,
        test_df
    ]:

        split["ROUTE"] = (
            split["ORIGIN"].astype(str)
            + "_"
            + split["DEST"].astype(str)
        )

    encoding_columns.append(
        (
            "ROUTE",
            "ROUTE_DELAY_RATE"
        )
    )


for column, feature_name in encoding_columns:

    print(
        f"Creating Bayesian encoding: {feature_name}"
    )

    historical_stats[feature_name] = (
        create_smoothed_encoding(
            train_df,
            val_df,
            test_df,
            column,
            feature_name
        )
    )


# ============================================================
# 9. SELECT FINAL FEATURES
# ============================================================

base_features = [

    "DAY_OF_MONTH",
    "DAY_OF_WEEK",
    "MONTH",
    "QUARTER",

    "DEP_MINUTES",
    "ARR_MINUTES",

    "DEP_HOUR_SIN",
    "DEP_HOUR_COS",

    "DOW_SIN",
    "DOW_COS",

    "IS_WEEKEND",

    "DISTANCE",
    "CRS_ELAPSED_TIME",

    "SCHEDULED_BLOCK_SPEED",

    "ORIGIN_HOURLY_DEPARTURES",
    "ORIGIN_DAILY_DEPARTURES",
    "DEST_DAILY_ARRIVALS",

    "CARRIER_ORIGIN_MARKET_SHARE"
]


# Add ONLY numeric TIME dummy features.
# Explicitly exclude TIME_BUCKET because it contains strings.
time_features = [
    col
    for col in df.columns
    if col.startswith("TIME_")
    and col != "TIME_BUCKET"
]

base_features.extend(time_features)


# Add Bayesian historical features
for _, feature_name in encoding_columns:
    base_features.append(feature_name)


# Keep only features that actually exist
features = [
    col
    for col in base_features
    if col in train_df.columns
]


print("\nFinal features:")
print(features)

print(f"\nTotal features: {len(features)}")


# ============================================================
# 10. CREATE FINAL DATASETS
# ============================================================

final_columns = features + [
    "DEP_DEL15"
]

train_output = train_df[
    final_columns
].copy()

val_output = val_df[
    final_columns
].copy()

test_output = test_df[
    final_columns
].copy()


# ============================================================
# 11. HANDLE MISSING VALUES
# MEDIAN CALCULATED ONLY FROM TRAIN DATA
# ============================================================

for column in features:

    # All final features should be numeric.
    train_output[column] = pd.to_numeric(
        train_output[column],
        errors="coerce"
    )

    val_output[column] = pd.to_numeric(
        val_output[column],
        errors="coerce"
    )

    test_output[column] = pd.to_numeric(
        test_output[column],
        errors="coerce"
    )

    median_value = train_output[
        column
    ].median()

    if pd.isna(median_value):
        median_value = 0

    train_output[column] = (
        train_output[column]
        .fillna(median_value)
    )

    val_output[column] = (
        val_output[column]
        .fillna(median_value)
    )

    test_output[column] = (
        test_output[column]
        .fillna(median_value)
    )


# ============================================================
# 12. SAVE DATASETS
# ============================================================

train_output.to_parquet(
    "results/train.parquet",
    index=False
)

val_output.to_parquet(
    "results/val.parquet",
    index=False
)

test_output.to_parquet(
    "results/test.parquet",
    index=False
)


# ============================================================
# 13. SAVE FEATURE LIST
# ============================================================

with open(
    "models/features.json",
    "w"
) as file:

    json.dump(
        features,
        file,
        indent=4
    )


# ============================================================
# 14. SAVE HISTORICAL STATS
# ============================================================

with open(
    "models/historical_stats.json",
    "w"
) as file:

    json.dump(
        {
            "global_mean": float(GLOBAL_MEAN),
            "smoothing": SMOOTHING,
            "encodings": historical_stats
        },
        file,
        indent=4
    )


# ============================================================
# 15. FINAL REPORT
# ============================================================

print("\n" + "=" * 60)
print("FEATURE PIPELINE COMPLETED SUCCESSFULLY")
print("=" * 60)

print(f"Train shape: {train_output.shape}")
print(f"Validation shape: {val_output.shape}")
print(f"Test shape: {test_output.shape}")

print(f"\nTotal ML features: {len(features)}")

print("\nFiles created:")
print("results/train.parquet")
print("results/val.parquet")
print("results/test.parquet")
print("models/features.json")
print("models/historical_stats.json")