"""
Unit Tests — Diabetes Prediction Pipeline
Run: pytest tests/test_pipeline.py -v
"""

import pytest
import pandas as pd
import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from src.pipeline import preprocess, split_and_scale, get_models, FEATURE_COLS, TARGET_COL


# ── Fixtures ──────────────────────────────────────────────────────────

@pytest.fixture
def sample_df():
    """Synthetic data mimicking PIMA dataset."""
    np.random.seed(42)
    n = 100
    data = {
        "Pregnancies":              np.random.randint(0, 10, n),
        "Glucose":                  np.random.randint(70, 200, n),
        "BloodPressure":            np.random.randint(50, 110, n),
        "SkinThickness":            np.random.randint(0, 60, n),
        "Insulin":                  np.random.randint(0, 400, n),
        "BMI":                      np.round(np.random.uniform(18, 50, n), 1),
        "DiabetesPedigreeFunction": np.round(np.random.uniform(0.1, 2.0, n), 3),
        "Age":                      np.random.randint(21, 80, n),
        "Outcome":                  np.random.randint(0, 2, n),
    }
    return pd.DataFrame(data)


# ── Tests ─────────────────────────────────────────────────────────────

class TestPreprocessing:

    def test_preprocess_returns_dataframe(self, sample_df):
        result = preprocess(sample_df)
        assert isinstance(result, pd.DataFrame)

    def test_no_zero_in_critical_cols(self, sample_df):
        # Introduce zeros to test imputation
        df = sample_df.copy()
        df.loc[0, "Glucose"] = 0
        df.loc[1, "BMI"]     = 0
        result = preprocess(df)
        for col in ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]:
            assert (result[col] == 0).sum() == 0, f"Zero found in {col} after preprocessing"

    def test_feature_engineering_columns_added(self, sample_df):
        result = preprocess(sample_df)
        assert "BMI_Age"         in result.columns
        assert "Glucose_Insulin" in result.columns
        assert "Glucose_BMI"     in result.columns

    def test_no_nulls_after_preprocessing(self, sample_df):
        result = preprocess(sample_df)
        assert result.isnull().sum().sum() == 0

    def test_shape_preserved(self, sample_df):
        result = preprocess(sample_df)
        assert result.shape[0] == sample_df.shape[0]


class TestSplitAndScale:

    def test_split_sizes(self, sample_df):
        df = preprocess(sample_df)
        X_train, X_test, y_train, y_test, _, _ = split_and_scale(df)
        total = X_train.shape[0] + X_test.shape[0]
        assert total == len(df)

    def test_scaler_output_range(self, sample_df):
        df = preprocess(sample_df)
        X_train, X_test, _, _, _, _ = split_and_scale(df)
        # Scaled data should be centered near 0
        assert abs(X_train.mean()) < 1.0

    def test_returns_numpy_arrays(self, sample_df):
        df = preprocess(sample_df)
        X_train, X_test, y_train, y_test, scaler, _ = split_and_scale(df)
        assert isinstance(X_train, np.ndarray)
        assert isinstance(X_test,  np.ndarray)


class TestModels:

    def test_all_models_present(self):
        models = get_models()
        expected = {"Logistic Regression", "Random Forest", "XGBoost",
                    "Gradient Boosting", "SVM"}
        assert set(models.keys()) == expected

    def test_models_fit_and_predict(self, sample_df):
        df = preprocess(sample_df)
        X_train, X_test, y_train, y_test, _, _ = split_and_scale(df)
        models = get_models()
        for name, model in models.items():
            model.fit(X_train, y_train)
            preds = model.predict(X_test)
            assert len(preds) == len(y_test), f"{name} prediction length mismatch"
            assert set(preds).issubset({0, 1}),   f"{name} produced unexpected classes"

    def test_predict_proba_range(self, sample_df):
        df = preprocess(sample_df)
        X_train, X_test, y_train, _, _, _ = split_and_scale(df)
        models = get_models()
        for name, model in models.items():
            model.fit(X_train, y_train)
            proba = model.predict_proba(X_test)[:, 1]
            assert proba.min() >= 0.0 and proba.max() <= 1.0, \
                f"{name} probability out of range"


class TestDataIntegrity:

    def test_class_balance_check(self, sample_df):
        # Both classes should exist
        assert set(sample_df[TARGET_COL].unique()) == {0, 1}

    def test_feature_cols_exist(self, sample_df):
        for col in FEATURE_COLS:
            assert col in sample_df.columns, f"Missing column: {col}"
