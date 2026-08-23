import os
import json
import pickle
import warnings
import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

warnings.filterwarnings("ignore")

os.makedirs("models", exist_ok=True)
os.makedirs("results", exist_ok=True)


# ============================================================
# LOAD DATA
# ============================================================

print("Loading processed datasets...")

train_df = pd.read_parquet("results/train.parquet")
val_df = pd.read_parquet("results/val.parquet")

with open("models/features.json", "r") as f:
    features = json.load(f)

X_train = train_df[features]
y_train = train_df["DEP_DEL15"]

X_val = val_df[features]
y_val = val_df["DEP_DEL15"]

print(f"Train data: {X_train.shape}")
print(f"Validation data: {X_val.shape}")

print("\nClass distribution:")
print(y_train.value_counts(normalize=True))


# ============================================================
# EVALUATION FUNCTION
# ============================================================

def evaluate_model(name, model, X, y, threshold=0.5):

    probabilities = model.predict_proba(X)[:, 1]

    predictions = (
        probabilities >= threshold
    ).astype(int)

    metrics = {
        "Model": name,
        "Accuracy": accuracy_score(y, predictions),
        "Precision": precision_score(
            y,
            predictions,
            zero_division=0
        ),
        "Recall": recall_score(
            y,
            predictions,
            zero_division=0
        ),
        "F1_Score": f1_score(
            y,
            predictions,
            zero_division=0
        ),
        "ROC_AUC": roc_auc_score(
            y,
            probabilities
        ),
        "Threshold": threshold
    }

    return metrics


# ============================================================
# THRESHOLD SEARCH
# ============================================================

def find_best_threshold(model, X, y):

    probabilities = model.predict_proba(X)[:, 1]

    best_threshold = 0.50
    best_accuracy = 0

    thresholds = np.arange(
        0.30,
        0.71,
        0.01
    )

    for threshold in thresholds:

        predictions = (
            probabilities >= threshold
        ).astype(int)

        accuracy = accuracy_score(
            y,
            predictions
        )

        if accuracy > best_accuracy:

            best_accuracy = accuracy
            best_threshold = float(
                round(threshold, 2)
            )

    return best_threshold


results = []

models = {}


# ============================================================
# 1. MAJORITY CLASS BASELINE
# ============================================================

print("\n" + "=" * 60)
print("BASELINE")
print("=" * 60)

majority_class = int(
    y_train.mode()[0]
)

baseline_predictions = np.full(
    len(y_val),
    majority_class
)

baseline_metrics = {
    "Model": "Majority Class Baseline",
    "Accuracy": accuracy_score(
        y_val,
        baseline_predictions
    ),
    "Precision": 0.0,
    "Recall": 0.0,
    "F1_Score": 0.0,
    "ROC_AUC": 0.50,
    "Threshold": 0.50
}

results.append(baseline_metrics)

print(baseline_metrics)


# ============================================================
# 2. LOGISTIC REGRESSION
# ============================================================

print("\n" + "=" * 60)
print("TRAINING LOGISTIC REGRESSION")
print("=" * 60)

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(
    X_train
)

X_val_scaled = scaler.transform(
    X_val
)

logistic_model = LogisticRegression(
    max_iter=1000,
    random_state=42,
    n_jobs=-1
)

logistic_model.fit(
    X_train_scaled,
    y_train
)

threshold = find_best_threshold(
    logistic_model,
    X_val_scaled,
    y_val
)

metrics = evaluate_model(
    "Logistic Regression",
    logistic_model,
    X_val_scaled,
    y_val,
    threshold
)

results.append(metrics)

models["Logistic Regression"] = logistic_model

with open(
    "models/logistic_regression.pkl",
    "wb"
) as f:

    pickle.dump(
        logistic_model,
        f
    )

with open(
    "models/scaler.pkl",
    "wb"
) as f:

    pickle.dump(
        scaler,
        f
    )

print(metrics)


# ============================================================
# 3. HIST GRADIENT BOOSTING
# ============================================================

print("\n" + "=" * 60)
print("TRAINING HIST GRADIENT BOOSTING")
print("=" * 60)

hist_model = HistGradientBoostingClassifier(
    max_iter=200,
    learning_rate=0.08,
    max_leaf_nodes=31,
    random_state=42
)

hist_model.fit(
    X_train,
    y_train
)

threshold = find_best_threshold(
    hist_model,
    X_val,
    y_val
)

metrics = evaluate_model(
    "HistGradientBoosting",
    hist_model,
    X_val,
    y_val,
    threshold
)

results.append(metrics)

models["HistGradientBoosting"] = hist_model

with open(
    "models/hist_gradient_boosting.pkl",
    "wb"
) as f:

    pickle.dump(
        hist_model,
        f
    )

print(metrics)


# ============================================================
# 4. LIGHTGBM
# ============================================================

print("\n" + "=" * 60)
print("TRAINING LIGHTGBM")
print("=" * 60)

lightgbm_model = LGBMClassifier(
    n_estimators=300,
    learning_rate=0.05,
    num_leaves=31,
    random_state=42,
    n_jobs=-1,
    verbosity=-1
)

lightgbm_model.fit(
    X_train,
    y_train
)

threshold = find_best_threshold(
    lightgbm_model,
    X_val,
    y_val
)

metrics = evaluate_model(
    "LightGBM",
    lightgbm_model,
    X_val,
    y_val,
    threshold
)

results.append(metrics)

models["LightGBM"] = lightgbm_model

with open(
    "models/lightgbm.pkl",
    "wb"
) as f:

    pickle.dump(
        lightgbm_model,
        f
    )

print(metrics)


# ============================================================
# 5. XGBOOST
# ============================================================

print("\n" + "=" * 60)
print("TRAINING XGBOOST")
print("=" * 60)

xgb_model = XGBClassifier(
    n_estimators=300,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    n_jobs=-1,
    eval_metric="logloss"
)

xgb_model.fit(
    X_train,
    y_train
)

threshold = find_best_threshold(
    xgb_model,
    X_val,
    y_val
)

metrics = evaluate_model(
    "XGBoost",
    xgb_model,
    X_val,
    y_val,
    threshold
)

results.append(metrics)

models["XGBoost"] = xgb_model

with open(
    "models/xgboost.pkl",
    "wb"
) as f:

    pickle.dump(
        xgb_model,
        f
    )

print(metrics)


# ============================================================
# 6. CATBOOST
# ============================================================

print("\n" + "=" * 60)
print("TRAINING CATBOOST")
print("=" * 60)

catboost_model = CatBoostClassifier(
    iterations=300,
    depth=6,
    learning_rate=0.05,
    loss_function="Logloss",
    verbose=False,
    random_seed=42
)

catboost_model.fit(
    X_train,
    y_train
)

threshold = find_best_threshold(
    catboost_model,
    X_val,
    y_val
)

metrics = evaluate_model(
    "CatBoost",
    catboost_model,
    X_val,
    y_val,
    threshold
)

results.append(metrics)

models["CatBoost"] = catboost_model

with open(
    "models/catboost.pkl",
    "wb"
) as f:

    pickle.dump(
        catboost_model,
        f
    )

print(metrics)


# ============================================================
# SAVE MODEL COMPARISON
# ============================================================

results_df = pd.DataFrame(
    results
)

results_df = results_df.sort_values(
    "Accuracy",
    ascending=False
)

results_df.to_csv(
    "results/model_comparison.csv",
    index=False
)

print("\n" + "=" * 60)
print("MODEL COMPARISON")
print("=" * 60)

print(
    results_df.to_string(
        index=False
    )
)


# ============================================================
# SELECT BEST MODEL
# ============================================================

best_model_name = results_df.iloc[0]["Model"]

best_threshold = float(
    results_df.iloc[0]["Threshold"]
)

best_accuracy = float(
    results_df.iloc[0]["Accuracy"]
)

print("\n" + "=" * 60)
print("BEST MODEL")
print("=" * 60)

print(f"Model: {best_model_name}")
print(f"Validation Accuracy: {best_accuracy * 100:.2f}%")
print(f"Optimal Threshold: {best_threshold}")


# ============================================================
# SAVE METADATA
# ============================================================

metadata = {
    "best_model": best_model_name,
    "validation_accuracy": best_accuracy,
    "optimal_threshold": best_threshold,
    "features": features,
    "all_results": results_df.to_dict(
        orient="records"
    )
}

with open(
    "models/model_metadata.json",
    "w"
) as f:

    json.dump(
        metadata,
        f,
        indent=4
    )


print("\n" + "=" * 60)
print("MODEL TRAINING COMPLETED SUCCESSFULLY")
print("=" * 60)

print("\nFiles created:")
print("models/logistic_regression.pkl")
print("models/scaler.pkl")
print("models/hist_gradient_boosting.pkl")
print("models/lightgbm.pkl")
print("models/xgboost.pkl")
print("models/catboost.pkl")
print("results/model_comparison.csv")
print("models/model_metadata.json")