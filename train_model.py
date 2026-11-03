"""Model Training Pipeline for Student Marks Prediction System.

This script executes the complete end-to-end Machine Learning training workflow:
1. Loads and validates the student marks dataset using modular preprocessing.
2. Isolates exactly ONE independent variable: X = Study_Hours.
3. Isolates ONE continuous dependent target: y = Final_Marks.
4. Performs an 80/20 train-test split with a fixed random seed (42).
5. Fits a Simple Linear Regression model via Ordinary Least Squares (sklearn.linear_model.LinearRegression).
6. Generates predictions on the held-out test split.
7. Evaluates regression performance: MAE, MSE, RMSE, and R² score.
8. Persists model artifacts to the artifacts/ directory (model pickle, metadata JSON, test results CSV).
"""

from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

from utils.data_preprocessing import (
    load_and_clean,
    FEATURE_COLUMN,
    TARGET_COLUMN,
)

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "student_marks_dataset.csv"
ARTIFACTS_DIR = ROOT / "artifacts"
MODEL_PATH = ARTIFACTS_DIR / "simple_linear_regression_model.pkl"
METADATA_PATH = ARTIFACTS_DIR / "model_metadata.json"
TEST_RESULTS_PATH = ARTIFACTS_DIR / "test_results.csv"

RANDOM_STATE = 42
TEST_SIZE = 0.20


def train():
    """Execute data ingestion, preprocessing, training, evaluation, and artifact serialization."""
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    # Step 1: Ensure dataset exists or generate it
    if not DATA_PATH.exists():
        print(f"Dataset not found at {DATA_PATH}. Generating synthetic dataset...")
        from generate_data import generate_dataset
        generate_dataset(output_path=DATA_PATH)

    # Step 2: Load and preprocess data via modular preprocessor
    print(f"Loading and preprocessing data from {DATA_PATH}...")
    df, audit = load_and_clean(DATA_PATH)
    print(f"Loaded {audit['rows_final']:,} valid records ({audit['rows_removed']} invalid records removed).")

    # Step 3 & 4: Strict feature selection - Single Independent Variable only
    X = df[[FEATURE_COLUMN]]  # Shape (N, 1) - Exactly ONE independent feature
    y = df[TARGET_COLUMN]     # Shape (N,) - Continuous target

    # Step 5: 80% Training, 20% Testing Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )

    # Step 6 & 7: Fit Simple Linear Regression using Ordinary Least Squares
    print("Training Simple Linear Regression model (sklearn.linear_model.LinearRegression)...")
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Step 8: Generate predictions on the held-out test split
    y_pred = model.predict(X_test)

    # Step 9: Compute comprehensive evaluation metrics
    mse = float(mean_squared_error(y_test, y_pred))
    mae = float(mean_absolute_error(y_test, y_pred))
    rmse = float(np.sqrt(mse))
    r2 = float(r2_score(y_test, y_pred))
    intercept = float(model.intercept_)
    slope = float(model.coef_[0])

    print("-" * 60)
    print("MODEL TRAINING EVALUATION SUMMARY")
    print("-" * 60)
    print(f"Algorithm            : Simple Linear Regression (OLS)")
    print(f"Independent Feature  : {FEATURE_COLUMN} (X)")
    print(f"Dependent Target     : {TARGET_COLUMN} (y)")
    print(f"Total Records        : {len(df):,}")
    print(f"Training Records     : {len(X_train):,} (80%)")
    print(f"Testing Records      : {len(X_test):,} (20%)")
    print(f"Learned Intercept b0 : {intercept:.4f}")
    print(f"Learned Slope b1     : {slope:.4f}")
    print(f"Equation             : Final_Marks = {intercept:.4f} + ({slope:.4f} * Study_Hours)")
    print(f"MAE (Mean Abs Error) : {mae:.4f} marks")
    print(f"MSE (Mean Sq Error)  : {mse:.4f} marks²")
    print(f"RMSE (Root Mean Sq)  : {rmse:.4f} marks")
    print(f"R² Score             : {r2:.4f}")
    print("-" * 60)

    # Step 10: Serialize trained model using Joblib
    joblib.dump(model, MODEL_PATH)
    print(f"Saved trained model artifact to: {MODEL_PATH}")

    # Step 11: Save metadata and test results
    metadata = {
        "algorithm": "Simple Linear Regression",
        "feature": FEATURE_COLUMN,
        "target": TARGET_COLUMN,
        "records": int(len(df)),
        "train_records": int(len(X_train)),
        "test_records": int(len(X_test)),
        "intercept": intercept,
        "slope": slope,
        "mae": mae,
        "mse": mse,
        "rmse": rmse,
        "r2": r2,
        "random_state": RANDOM_STATE,
        "test_size": TEST_SIZE,
        "equation_formula": f"ŷ = {intercept:.4f} + {slope:.4f} * x",
        "dataset_note": "Synthetic educational demonstration dataset; not collected from real students.",
    }

    with open(METADATA_PATH, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    print(f"Saved model metadata to: {METADATA_PATH}")

    test_results_df = pd.DataFrame({
        "Study_Hours": X_test[FEATURE_COLUMN].values,
        "Actual_Marks": y_test.values,
        "Predicted_Marks": y_pred,
        "Residual": y_test.values - y_pred,
    })
    test_results_df.to_csv(TEST_RESULTS_PATH, index=False)
    print(f"Saved test evaluation predictions to: {TEST_RESULTS_PATH}")
    print("=" * 60)

    return model, metadata, test_results_df


if __name__ == "__main__":
    train()
