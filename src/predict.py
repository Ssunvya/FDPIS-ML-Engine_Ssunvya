import json
import pickle
import pandas as pd


class FDPISPredictor:

    def __init__(self, model_dir="models"):

        self.model_dir = model_dir

        # Load feature names
        with open(
            f"{model_dir}/features.json",
            "r"
        ) as f:

            self.features = json.load(f)

        # Load metadata
        with open(
            f"{model_dir}/model_metadata.json",
            "r"
        ) as f:

            self.metadata = json.load(f)

        self.model_name = self.metadata["best_model"]

        self.threshold = self.metadata[
            "optimal_threshold"
        ]

        # Model file mapping
        model_files = {
            "Logistic Regression":
                "logistic_regression.pkl",

            "HistGradientBoosting":
                "hist_gradient_boosting.pkl",

            "LightGBM":
                "lightgbm.pkl",

            "XGBoost":
                "xgboost.pkl",

            "CatBoost":
                "catboost.pkl"
        }

        model_path = (
            f"{model_dir}/"
            f"{model_files[self.model_name]}"
        )

        with open(
            model_path,
            "rb"
        ) as f:

            self.model = pickle.load(f)

        # Logistic Regression requires scaler
        self.scaler = None

        if self.model_name == "Logistic Regression":

            with open(
                f"{model_dir}/scaler.pkl",
                "rb"
            ) as f:

                self.scaler = pickle.load(f)


    def prepare_features(self, data):

        df = data.copy()

        # Add missing columns
        for feature in self.features:

            if feature not in df.columns:
                df[feature] = 0

        # Keep correct feature order
        df = df[self.features]

        # Convert everything to numeric
        df = df.apply(
            pd.to_numeric,
            errors="coerce"
        )

        # Replace missing values
        df = df.fillna(0)

        return df


    def predict_proba(self, data):

        X = self.prepare_features(data)

        if self.scaler is not None:

            X = self.scaler.transform(X)

        probabilities = (
            self.model.predict_proba(X)[:, 1]
        )

        return probabilities


    def predict(self, data):

        probabilities = self.predict_proba(data)

        predictions = (
            probabilities >= self.threshold
        ).astype(int)

        return predictions


if __name__ == "__main__":

    print("FDPIS Predictor loaded successfully.")