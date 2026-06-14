"""
Diabetes Prediction - Full ML Pipeline
Author: Saumya Shreya
Dataset: PIMA Indians Diabetes Dataset
"""

import pandas as pd
import numpy as np
import joblib
import os
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    roc_auc_score, roc_curve, precision_recall_curve, f1_score
)
from xgboost import XGBClassifier
import shap
import matplotlib.pyplot as plt
import seaborn as sns

# ─────────────────────────────────────────────
# 1. DATA LOADING & VALIDATION
# ─────────────────────────────────────────────

FEATURE_COLS = [
    "Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
    "Insulin", "BMI", "DiabetesPedigreeFunction", "Age"
]
TARGET_COL = "Outcome"

# Columns where 0 is biologically impossible → treat as missing
ZERO_AS_NAN = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]


def load_data(filepath: str) -> pd.DataFrame:
    df = pd.read_csv(filepath)
    assert set(FEATURE_COLS + [TARGET_COL]).issubset(df.columns), \
        "Dataset missing expected columns."
    print(f"[INFO] Loaded {len(df)} rows, {df[TARGET_COL].value_counts().to_dict()}")
    return df


# ─────────────────────────────────────────────
# 2. PREPROCESSING
# ─────────────────────────────────────────────

def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Replace biologically impossible zeros with NaN
    for col in ZERO_AS_NAN:
        df[col] = df[col].replace(0, np.nan)

    # Impute with median (robust to outliers)
    for col in ZERO_AS_NAN:
        df[col].fillna(df[col].median(), inplace=True)

    # Feature engineering
    df["BMI_Age"] = df["BMI"] * df["Age"]
    df["Glucose_Insulin"] = df["Glucose"] / (df["Insulin"] + 1)
    df["Glucose_BMI"] = df["Glucose"] * df["BMI"]

    print("[INFO] Preprocessing complete. Shape:", df.shape)
    return df


def split_and_scale(df: pd.DataFrame):
    feature_cols = FEATURE_COLS + ["BMI_Age", "Glucose_Insulin", "Glucose_BMI"]
    X = df[feature_cols]
    y = df[TARGET_COL]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc  = scaler.transform(X_test)

    return X_train_sc, X_test_sc, y_train, y_test, scaler, feature_cols


# ─────────────────────────────────────────────
# 3. MODEL DEFINITIONS
# ─────────────────────────────────────────────

def get_models():
    return {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Random Forest":       RandomForestClassifier(n_estimators=200, random_state=42),
        "XGBoost":             XGBClassifier(
                                   n_estimators=200, learning_rate=0.05,
                                   max_depth=4, use_label_encoder=False,
                                   eval_metric="logloss", random_state=42
                               ),
        "Gradient Boosting":   GradientBoostingClassifier(n_estimators=150, random_state=42),
        "SVM":                 SVC(probability=True, kernel="rbf", random_state=42),
    }


def build_ensemble(models: dict):
    estimators = [(name, model) for name, model in models.items()]
    return VotingClassifier(estimators=estimators, voting="soft")


# ─────────────────────────────────────────────
# 4. TRAINING & EVALUATION
# ─────────────────────────────────────────────

def train_and_evaluate(X_train, X_test, y_train, y_test) -> dict:
    models = get_models()
    results = {}
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred  = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]

        cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring="roc_auc")

        results[name] = {
            "model":    model,
            "accuracy": accuracy_score(y_test, y_pred),
            "f1":       f1_score(y_test, y_pred),
            "roc_auc":  roc_auc_score(y_test, y_proba),
            "cv_mean":  cv_scores.mean(),
            "cv_std":   cv_scores.std(),
            "y_pred":   y_pred,
            "y_proba":  y_proba,
        }
        print(f"[{name}]  Acc={results[name]['accuracy']:.3f} | "
              f"AUC={results[name]['roc_auc']:.3f} | "
              f"CV={results[name]['cv_mean']:.3f}±{results[name]['cv_std']:.3f}")

    # Ensemble
    ensemble = build_ensemble(models)
    ensemble.fit(X_train, y_train)
    y_pred_ens  = ensemble.predict(X_test)
    y_proba_ens = ensemble.predict_proba(X_test)[:, 1]
    results["Ensemble (Voting)"] = {
        "model":    ensemble,
        "accuracy": accuracy_score(y_test, y_pred_ens),
        "f1":       f1_score(y_test, y_pred_ens),
        "roc_auc":  roc_auc_score(y_test, y_proba_ens),
        "cv_mean":  None,
        "cv_std":   None,
        "y_pred":   y_pred_ens,
        "y_proba":  y_proba_ens,
    }
    print(f"[Ensemble]  Acc={results['Ensemble (Voting)']['accuracy']:.3f} | "
          f"AUC={results['Ensemble (Voting)']['roc_auc']:.3f}")

    return results


# ─────────────────────────────────────────────
# 5. SHAP EXPLAINABILITY
# ─────────────────────────────────────────────

def explain_with_shap(model, X_train, X_test, feature_names, out_dir="models"):
    os.makedirs(out_dir, exist_ok=True)
    explainer   = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)

    # Summary plot
    plt.figure()
    shap.summary_plot(shap_values, X_test,
                      feature_names=feature_names, show=False)
    plt.tight_layout()
    plt.savefig(f"{out_dir}/shap_summary.png", dpi=150)
    plt.close()

    # Bar plot
    plt.figure()
    shap.summary_plot(shap_values, X_test, plot_type="bar",
                      feature_names=feature_names, show=False)
    plt.tight_layout()
    plt.savefig(f"{out_dir}/shap_bar.png", dpi=150)
    plt.close()

    print("[INFO] SHAP plots saved.")
    return shap_values


# ─────────────────────────────────────────────
# 6. SAVE BEST MODEL
# ─────────────────────────────────────────────

def save_best(results: dict, scaler, feature_cols, out_dir="models"):
    os.makedirs(out_dir, exist_ok=True)
    best_name = max(results, key=lambda k: results[k]["roc_auc"])
    best_model = results[best_name]["model"]

    joblib.dump(best_model, f"{out_dir}/best_model.pkl")
    joblib.dump(scaler,     f"{out_dir}/scaler.pkl")
    joblib.dump(feature_cols, f"{out_dir}/feature_cols.pkl")

    print(f"[INFO] Best model: {best_name} (AUC={results[best_name]['roc_auc']:.3f}) saved.")
    return best_name, best_model


# ─────────────────────────────────────────────
# 7. FULL RUN
# ─────────────────────────────────────────────

if __name__ == "__main__":
    df = load_data("data/diabetes.csv")
    df = preprocess(df)
    X_train, X_test, y_train, y_test, scaler, feature_cols = split_and_scale(df)

    results = train_and_evaluate(X_train, X_test, y_train, y_test)

    best_name, best_model = save_best(results, scaler, feature_cols)

    # SHAP on XGBoost (tree-based)
    xgb_model = results["XGBoost"]["model"]
    explain_with_shap(xgb_model, X_train, X_test, feature_cols)

    print("\n✅ Pipeline complete.")
