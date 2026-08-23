import json
import pickle
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix
)

print("Loading locked test dataset...")

test_df = pd.read_parquet("results/test.parquet")

with open("models/features.json", "r") as f:
    features = json.load(f)

with open("models/model_metadata.json", "r") as f:
    metadata = json.load(f)

X_test = test_df[features]
y_test = test_df["DEP_DEL15"]

best_model_name = metadata["best_model"]
threshold = metadata["optimal_threshold"]

print(f"Test data shape: {X_test.shape}")
print(f"Best model: {best_model_name}")
print(f"Optimal threshold: {threshold}")


# LOAD CORRECT MODEL
model_files = {
    "Logistic Regression": "models/logistic_regression.pkl",
    "HistGradientBoosting": "models/hist_gradient_boosting.pkl",
    "LightGBM": "models/lightgbm.pkl",
    "XGBoost": "models/xgboost.pkl",
    "CatBoost": "models/catboost.pkl"
}

model_path = model_files[best_model_name]

with open(model_path, "rb") as f:
    model = pickle.load(f)


# LOGISTIC REGRESSION NEEDS SCALING
if best_model_name == "Logistic Regression":

    with open("models/scaler.pkl", "rb") as f:
        scaler = pickle.load(f)

    X_test_input = scaler.transform(X_test)

else:
    X_test_input = X_test


# PREDICT
probabilities = model.predict_proba(X_test_input)[:, 1]

predictions = (
    probabilities >= threshold
).astype(int)


# METRICS
accuracy = accuracy_score(
    y_test,
    predictions
)

precision = precision_score(
    y_test,
    predictions,
    zero_division=0
)

recall = recall_score(
    y_test,
    predictions,
    zero_division=0
)

f1 = f1_score(
    y_test,
    predictions,
    zero_division=0
)

roc_auc = roc_auc_score(
    y_test,
    probabilities
)

cm = confusion_matrix(
    y_test,
    predictions
)

tn, fp, fn, tp = cm.ravel()


# RESULTS
results = {
    "best_model": best_model_name,
    "test_samples": int(len(y_test)),
    "accuracy": float(accuracy),
    "precision": float(precision),
    "recall": float(recall),
    "f1_score": float(f1),
    "roc_auc": float(roc_auc),
    "threshold": float(threshold),
    "confusion_matrix": {
        "true_negative": int(tn),
        "false_positive": int(fp),
        "false_negative": int(fn),
        "true_positive": int(tp)
    }
}


# SAVE
with open(
    "results/final_test_evaluation.json",
    "w"
) as f:

    json.dump(
        results,
        f,
        indent=4
    )


print("\n" + "=" * 60)
print("LOCKED TEST SET EVALUATION")
print("=" * 60)

print(f"Best Model: {best_model_name}")
print(f"Test Samples: {len(y_test)}")

print(f"\nAccuracy: {accuracy * 100:.2f}%")
print(f"Precision: {precision * 100:.2f}%")
print(f"Recall: {recall * 100:.2f}%")
print(f"F1 Score: {f1 * 100:.2f}%")
print(f"ROC-AUC: {roc_auc:.4f}")

print(f"\nOptimal Threshold: {threshold}")

print("\nConfusion Matrix:")
print(f"True Negative:  {tn}")
print(f"False Positive: {fp}")
print(f"False Negative: {fn}")
print(f"True Positive:  {tp}")

print("\n" + "=" * 60)
print("TEST EVALUATION COMPLETED SUCCESSFULLY")
print("=" * 60)

print("\nSaved:")
print("results/final_test_evaluation.json")