# Predictive Maintenance MLOps Pipeline with Drift Monitoring and Explainability

This project builds an end-to-end MLOps pipeline for heavy-equipment machine failure prediction. The pipeline covers data validation, exploratory analysis, model training, experiment tracking, hyperparameter tuning, model registration, drift detection, and explainability using SHAP.

The goal is to predict machine failure types from operational sensor readings and connect model behavior with practical maintenance decisions.

## Project Overview

The project uses three datasets:

- `train.csv`: Historical labelled baseline readings
- `current.csv`: Stable current production readings
- `stress.csv`: Heavy-load production readings

The model predicts the following failure classes:

| Class | Failure Type | Description |
|---|---|---|
| `0` | No Failure | Normal operating condition |
| `1` | TWF | Tool Wear Failure |
| `2` | HDF | Heat Dissipation Failure |
| `3` | PWF | Power Failure |
| `4` | OSF | Overstrain Failure |

## Key Objectives

- Validate incoming sensor data using `Pandera`
- Engineer physically meaningful features such as `Power_W` and `Temp_diff`
- Handle class imbalance using `SMOTE`
- Train and compare multiple ML models using `MLflow`
- Tune the best model using `Optuna`
- Register the final model in the `MLflow Model Registry`
- Detect production drift using `Evidently`
- Explain model predictions using multiclass `SHAP`
- Convert model and monitoring outputs into maintenance recommendations

## Pipeline Stages

### 1. Data Validation and EDA

The raw data was validated using a `Pandera` schema to ensure correct data types, valid ranges, and expected categorical values.

Key checks included:

- Schema validation for `train`, `current`, and `stress` datasets
- Class imbalance inspection
- Distribution analysis of sensor features
- Derived feature engineering

Engineered features:

```python
Power_W = Torque * (Rotational speed * 2π / 60)
Temp_diff = Process temperature - Air temperature
```

These features add physical meaning by representing mechanical power demand and thermal difference.

### 2. Experiment Tracking and Model Selection

The `Type` column was encoded and a stratified 80/20 train-validation split was performed. `SMOTE` was applied only on the training split to avoid data leakage.

Models trained:

- Logistic Regression
- Random Forest
- XGBoost
- LightGBM

Model selection was based on `macro_f1`, not accuracy, because the dataset is highly imbalanced and rare failure classes are operationally important.

Initial model comparison showed that `XGBoost` performed best with a baseline `macro_f1` of `0.7481`.

After Optuna tuning, the final tuned XGBoost model achieved:

| Metric | Value |
|---|---:|
| Baseline XGBoost Macro F1 | `0.7481` |
| Tuned XGBoost Macro F1 | `0.7610` |
| Improvement | `+0.0129` |

The tuned XGBoost model was saved as:

```text
best_model.pkl
```

### 3. Drift Detection and Monitoring

Drift detection was performed using `Evidently`.

Generated reports:

```text
drift_current.html
drift_stress.html
```

The current production batch showed no drift:

| Metric | Value |
|---|---:|
| Data drift detected | `False` |
| Number of drifted features | `0` |
| Share of drifted features | `0.0` |

The stress batch showed meaningful drift in operationally important features:

- `Rotational speed`
- `Torque`
- `Tool wear`

This indicates that the stress batch represents a different operating condition compared with the historical training baseline.

### 4. Explainability with SHAP

SHAP was used to explain the tuned XGBoost model at a class-specific level.

Generated figure:

```text
shap_per_class.png
```

Key SHAP insights:

| Failure Type | Important Drivers | Interpretation |
|---|---|---|
| `TWF` | `Tool wear` | Failure is linked to accumulated tool degradation |
| `HDF` | `Temp_diff` | Failure is linked to poor heat dissipation |
| `PWF` | `Power_W` | Failure is linked to high mechanical power demand |
| `OSF` | `Tool wear`, `Torque` | Failure is linked to worn tools operating under high load |

The SHAP results also supported the drift interpretation. Since the stress batch showed drift in `Tool wear` and `Torque`, failures such as `OSF` and `TWF` are more likely to increase under stress conditions.

## Final Recommendation

The model does not require immediate retraining for the stable current batch because no drift was detected.

However, the stress batch shows drift in features that are important for failure prediction. If this stress condition becomes a recurring production pattern, the model should be monitored closely and retrained once labelled stress-condition data is available.

Actionable maintenance recommendation:

| Condition | Risked Failure Class | Action |
|---|---|---|
| High `Tool wear` and increased `Torque` under stress conditions | `OSF`, `TWF`, possible `PWF` | Prioritize tool inspection, preventive replacement, and load monitoring during heavy operating periods |

## Why Accuracy Was Not Used as the Main Metric

Accuracy is misleading in this problem because the dataset is dominated by the `No Failure` class. A model can achieve high accuracy by mostly predicting the majority class while performing poorly on rare but critical failure types.

Therefore, `macro_f1` was used because it gives equal importance to each failure class.

## Key Takeaways

- The final tuned `XGBoost` model achieved the best `macro_f1`
- `macro_f1` was more meaningful than accuracy for this imbalanced classification problem
- The weakest class, `TWF`, remained difficult due to very limited real samples
- Drift in the stress batch was concentrated in operationally important features
- SHAP explanations connected model behavior to real engineering mechanisms
- The retraining recommendation was based on evidence from both drift monitoring and explainability

## Tech Stack

- Python
- Pandas
- NumPy
- Scikit-learn
- Imbalanced-learn
- XGBoost
- LightGBM
- Optuna
- MLflow
- Pandera
- Evidently
- SHAP
- Matplotlib
- Seaborn

## Repository Structure

```text
.
├── data/
│   ├── train.csv
│   ├── current.csv
│   └── stress.csv
├── notebooks/
│   └── predictive_maintenance_mlops.ipynb
├── reports/
│   ├── drift_current.html
│   ├── drift_stress.html
│   └── shap_per_class.png
├── models/
│   └── best_model.pkl
├── requirements.txt
└── README.md
```

## How to Run

Create and activate a virtual environment:

```bash
py -3.12 -m venv venv
venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the notebook:

```bash
jupyter notebook
```

To view MLflow runs:

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

Then open:

```text
http://127.0.0.1:5000
```

This was a Capstone project as part of EPGP - Applied AI and Agentic AI from IIIT Bangalore & Upgrad

