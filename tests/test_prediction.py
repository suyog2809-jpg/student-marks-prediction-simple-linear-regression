"""Automated Verification and Smoke Test Suite for Student Marks Prediction System.

This suite performs end-to-end validation across:
1. Dataset existence and schema integrity.
2. Preprocessing and data hygiene validation.
3. Model artifact existence and joblib deserialization.
4. Mathematical constraint verification: Exactly ONE independent variable (SLR).
5. Prediction behavior, boundary checking, and numeric sanity.
6. Metadata consistency and evaluation metric validity.
7. Test results CSV structure and residual calculation fidelity.
"""

from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "student_marks_dataset.csv"
MODEL_PATH = ROOT / "artifacts/simple_linear_regression_model.pkl"
META_PATH = ROOT / "artifacts/model_metadata.json"
TEST_RESULTS_PATH = ROOT / "artifacts/test_results.csv"


def test_dataset_exists_and_valid():
    """Verify dataset presence, schema, and record bounds."""
    assert DATA_PATH.exists(), f"Dataset not found at {DATA_PATH}"
    df = pd.read_csv(DATA_PATH)
    
    expected_cols = ["Student_ID", "Study_Hours", "Final_Marks"]
    assert list(df.columns) == expected_cols, f"Schema mismatch. Expected {expected_cols}, got {list(df.columns)}"
    assert 3000 <= len(df) <= 5000, f"Record count {len(df)} is outside expected 3,000–5,000 range"
    assert df["Study_Hours"].isna().sum() == 0, "Null values detected in Study_Hours"
    assert df["Final_Marks"].isna().sum() == 0, "Null values detected in Final_Marks"
    assert df["Study_Hours"].between(0.0, 24.0).all(), "Study_Hours contains out-of-range values"
    assert df["Final_Marks"].between(0.0, 100.0).all(), "Final_Marks contains out-of-range values"
    print("PASS: test_dataset_exists_and_valid")


def test_preprocessing_module():
    """Verify that utils.data_preprocessing functions execute properly."""
    import sys
    sys.path.insert(0, str(ROOT))
    from utils.data_preprocessing import load_and_clean, validate_schema, check_data_quality

    df, audit = load_and_clean(DATA_PATH)
    assert len(df) > 0, "Cleaned dataset is empty"
    assert audit["rows_final"] == len(df), "Audit report rows_final does not match DataFrame length"
    assert audit["nulls_detected"] == 0, "Unexpected nulls detected"

    quality = check_data_quality(df)
    assert quality["is_clean"] is True, f"Data quality check failed: {quality}"
    print("PASS: test_preprocessing_module")


def test_model_loading_and_architecture():
    """Verify trained model artifact loads and confirms Simple Linear Regression architecture."""
    assert MODEL_PATH.exists(), f"Model artifact not found at {MODEL_PATH}"
    model = joblib.load(MODEL_PATH)

    # Must have exactly ONE coefficient
    assert hasattr(model, "coef_"), "Model has no attribute coef_"
    assert hasattr(model, "intercept_"), "Model has no attribute intercept_"
    assert model.coef_.shape == (1,), f"Expected coef_ shape (1,), but got {model.coef_.shape}"
    assert np.isscalar(model.intercept_) or model.intercept_.shape == (1,) or isinstance(model.intercept_, (float, np.floating)), "Invalid intercept type"
    assert model.coef_[0] > 0, f"Expected positive relationship between study hours and marks, got {model.coef_[0]}"
    print("PASS: test_model_loading_and_architecture")


def test_model_prediction_sanity():
    """Verify prediction outputs across standard and boundary hours."""
    model = joblib.load(MODEL_PATH)

    test_inputs = [0.5, 2.0, 4.0, 6.0, 8.0, 10.0]
    preds = model.predict(pd.DataFrame({"Study_Hours": test_inputs}))

    assert len(preds) == len(test_inputs), f"Expected {len(test_inputs)} predictions, got {len(preds)}"
    
    # Monotonicity test: More study hours must produce higher predicted marks in simple linear regression
    for i in range(len(preds) - 1):
        assert preds[i + 1] > preds[i], "Monotonicity violated: predicted marks should increase with study hours"

    # Clipped predictions should strictly fall in [0, 100]
    clipped = np.clip(preds, 0, 100)
    assert all(0.0 <= p <= 100.0 for p in clipped), "Clipped predictions exceed academic boundaries"
    print("PASS: test_model_prediction_sanity")


def test_metadata_and_metrics():
    """Verify model metadata structure and valid metric bounds."""
    assert META_PATH.exists(), f"Metadata JSON not found at {META_PATH}"
    with open(META_PATH, "r", encoding="utf-8") as f:
        meta = json.load(f)

    required_keys = ["algorithm", "feature", "target", "records", "train_records", "test_records", "intercept", "slope", "mae", "mse", "rmse", "r2"]
    for k in required_keys:
        assert k in meta, f"Missing metadata key: {k}"

    assert meta["algorithm"] == "Simple Linear Regression"
    assert meta["feature"] == "Study_Hours"
    assert meta["target"] == "Final_Marks"
    assert meta["train_records"] + meta["test_records"] == meta["records"]
    assert 0.0 < meta["r2"] <= 1.0, f"R² score out of expected bounds: {meta['r2']}"
    assert meta["mae"] > 0, "MAE must be positive"
    assert meta["rmse"] > 0, "RMSE must be positive"
    print("PASS: test_metadata_and_metrics")


def test_test_results_csv():
    """Verify test_results.csv fidelity and residual arithmetic."""
    assert TEST_RESULTS_PATH.exists(), f"Test results CSV not found at {TEST_RESULTS_PATH}"
    df_test = pd.read_csv(TEST_RESULTS_PATH)

    expected_cols = ["Study_Hours", "Actual_Marks", "Predicted_Marks", "Residual"]
    assert list(df_test.columns) == expected_cols, f"Mismatch in test results columns: {list(df_test.columns)}"
    assert len(df_test) > 0, "Test results file is empty"

    # Residual verification: Residual = Actual - Predicted
    calc_residuals = df_test["Actual_Marks"] - df_test["Predicted_Marks"]
    np.testing.assert_allclose(df_test["Residual"], calc_residuals, atol=1e-5, err_msg="Residual arithmetic mismatch")
    print("PASS: test_test_results_csv")


def test_client_specific_study_hours():
    """Explicitly verify client-mandated study hours: 2, 5, 8, min, and max hours."""
    model = joblib.load(MODEL_PATH)
    with open(META_PATH, "r", encoding="utf-8") as f:
        meta = json.load(f)

    b0 = meta["intercept"]
    b1 = meta["slope"]

    # Target test benchmarks requested by client:
    # Minimum supported: 0.0, 0.5; Key targets: 2.0, 5.0, 8.0; Maximum: 10.0, 12.0, 16.0
    benchmarks = [0.0, 0.5, 2.0, 5.0, 8.0, 10.0, 12.0, 16.0]
    preds = model.predict(pd.DataFrame({"Study_Hours": benchmarks}))

    for h, p in zip(benchmarks, preds):
        expected_math = b0 + (b1 * h)
        np.testing.assert_allclose(p, expected_math, rtol=1e-5, err_msg=f"Equation mismatch at {h} hours")
        clipped = float(np.clip(p, 0.0, 100.0))
        assert 0.0 <= clipped <= 100.0, f"Clipped prediction {clipped} out of range [0, 100]"

    # Verify specific known client cases:
    p_2h = float(model.predict(pd.DataFrame({"Study_Hours": [2.0]}))[0])
    p_5h = float(model.predict(pd.DataFrame({"Study_Hours": [5.0]}))[0])
    p_8h = float(model.predict(pd.DataFrame({"Study_Hours": [8.0]}))[0])
    p_min = float(model.predict(pd.DataFrame({"Study_Hours": [0.0]}))[0])
    p_12h = float(model.predict(pd.DataFrame({"Study_Hours": [12.0]}))[0])
    p_max = float(model.predict(pd.DataFrame({"Study_Hours": [16.0]}))[0])

    assert p_min < p_2h < p_5h < p_8h < p_12h < p_max, "Monotonic ordering violated across test hours"
    
    # Boundary capping verification:
    # At 12.0h and 16.0h raw output must exceed 100, and clipped score must be exactly 100.0
    assert p_12h > 100.0, f"Expected 12.0h raw output > 100, got {p_12h}"
    assert np.clip(p_12h, 0.0, 100.0) == 100.0, "Clipped 12.0h score must be 100.0"
    assert p_8h <= 100.0, f"Expected 8.0h raw output <= 100, got {p_8h}"

    print(f"PASS: test_client_specific_study_hours (2h={p_2h:.2f}, 5h={p_5h:.2f}, 8h={p_8h:.2f}, min={p_min:.2f}, 12h_raw={p_12h:.2f}, max_raw={p_max:.2f})")


if __name__ == "__main__":
    print("=" * 65)
    print("RUNNING AUTOMATED TEST SUITE: STUDENT MARKS PREDICTION")
    print("=" * 65)
    test_dataset_exists_and_valid()
    test_preprocessing_module()
    test_model_loading_and_architecture()
    test_model_prediction_sanity()
    test_client_specific_study_hours()
    test_metadata_and_metrics()
    test_test_results_csv()
    print("=" * 65)
    print("ALL 7 AUTOMATED TESTS PASSED SUCCESSFULLY! (100% PASS RATE)")
    print("=" * 65)
