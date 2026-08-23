# FDPIS-ML-Engine_Ssunvya

Machine learning engine for flight delay prediction and propagation risk analysis.

# Flight Delay Propagation Intelligence System (FDPIS) — ML Engine

A machine learning project for predicting whether a scheduled flight departure will be delayed by 15 minutes or more.

## Project Objective

The objective is to predict:

- **Target:** `DEP_DEL15`
- **1:** Flight delayed by 15 minutes or more
- **0:** Flight on-time or delayed by less than 15 minutes

## ML Pipeline

Raw Flight Dataset

↓

Data Cleaning & Feature Engineering

↓

Chronological Train / Validation / Test Split

↓

Model Training

↓

Model Comparison

↓

Best Model Selection

↓

Locked Test Evaluation

↓

Flight Delay Prediction CLI

## Models Tested

- Logistic Regression
- HistGradientBoosting
- LightGBM
- XGBoost
- CatBoost

## Best Model

**CatBoost**

| Metric | Result |
|---|---:|
| Validation Accuracy | 75.67% |
| Locked Test Accuracy | 75.55% |
| Precision | 40.65% |
| Recall | 2.64% |
| F1 Score | 4.95% |
| ROC-AUC | 0.6064 |
| Optimal Threshold | 0.39 |

## Project Structure

```text
FDPIS/
├── models/
├── results/
├── src/
│   ├── feature_pipeline.py
│   ├── train_models.py
│   ├── evaluate_test.py
│   ├── predict.py
│   └── cli_predict.py
├── requirements.txt
├── .gitignore
└── README.md
```

## Installation

```bash
pip install -r requirements.txt
```

## Dataset

Place the dataset in the project directory as:

```text
T_ONTIME_REPORTING.csv
```

The raw dataset is excluded from GitHub using `.gitignore`.

## Run Feature Engineering

```bash
python src/feature_pipeline.py
```

## Train Models

```bash
python src/train_models.py
```

## Evaluate Locked Test Set

```bash
python src/evaluate_test.py
```

## Make a Flight Prediction

```bash
python src/cli_predict.py --dep_time 830 --arr_time 1100 --distance 762 --elapsed_time 150 --day_of_month 15 --day_of_week 4 --month 8
```

## Example Output

```text
Delay Probability: 1.81%
Decision Threshold: 0.39

Prediction Status: ON-TIME (< 15 minutes)
Risk Assessment: LOW DELAY RISK
```

## Author

Ssunvya Jain