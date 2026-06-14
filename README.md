# 🩺 Diabetes Prediction System
### An End-to-End Machine Learning Project
[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://kmanuxb4ajegsiicftlqkb.streamlit.app)
![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-1.22+-red?logo=streamlit)
![XGBoost](https://img.shields.io/badge/XGBoost-1.7+-orange)
![SHAP](https://img.shields.io/badge/Explainability-SHAP-green)
![Tests](https://img.shields.io/badge/Tests-Pytest-yellow)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

> **Predicting diabetes onset from clinical health metrics using an ensemble of ML models, with SHAP-based explainability and an interactive Streamlit web app.**

---

## 📌 Table of Contents
- [Problem Statement](#-problem-statement)
- [Project Structure](#-project-structure)
- [Dataset](#-dataset)
- [Methodology](#-methodology)
- [Results](#-results)
- [How to Run](#-how-to-run)
- [Explainability (SHAP)](#-explainability-shap)
- [Tech Stack](#-tech-stack)

---

## 🎯 Problem Statement

Diabetes affects **537 million people worldwide** and is the 8th leading cause of death. Early prediction can dramatically improve outcomes. This project builds a clinical decision-support tool that:

- Predicts whether a patient is at risk of diabetes
- Explains *why* using SHAP values
- Provides an interactive interface for healthcare workers

---

## 📁 Project Structure

```
diabetes-prediction/
│
├── data/
│   └── diabetes.csv               # PIMA Indians Diabetes Dataset
│
├── notebooks/
│   └── diabetes_analysis.ipynb    # Full EDA + modeling walkthrough
│
├── src/
│   └── pipeline.py                # ML pipeline (preprocess → train → evaluate → save)
│
├── models/
│   ├── best_model.pkl             # Saved best model
│   ├── scaler.pkl                 # StandardScaler
│   ├── feature_cols.pkl           # Feature list
│   ├── shap_summary.png           # SHAP feature importance
│   ├── roc_curves.png             # ROC comparison chart
│   └── confusion_matrix.png       # Confusion matrix
│
├── tests/
│   └── test_pipeline.py           # Pytest unit tests
│
├── app.py                         # Streamlit web application
├── requirements.txt
└── README.md
```

---

## 📊 Dataset

**PIMA Indians Diabetes Dataset** — UCI Machine Learning Repository

| Feature | Description |
|---------|-------------|
| `Pregnancies` | Number of times pregnant |
| `Glucose` | Plasma glucose concentration (mg/dL) |
| `BloodPressure` | Diastolic blood pressure (mm Hg) |
| `SkinThickness` | Triceps skin fold thickness (mm) |
| `Insulin` | 2-hour serum insulin (μU/mL) |
| `BMI` | Body Mass Index (kg/m²) |
| `DiabetesPedigreeFunction` | Family history score |
| `Age` | Age in years |
| `Outcome` | **Target** — 1 = Diabetic, 0 = Not Diabetic |

- **768 samples** | **8 features** | **34.9% positive class**

---

## 🔬 Methodology

### 1. Data Preprocessing
- Detected biologically impossible zeros (e.g., Glucose=0, BMI=0)
- Replaced with `NaN` → imputed using **column median**
- Handles class imbalance awareness in train/test split (`stratify=y`)

### 2. Feature Engineering
Three new features derived from domain knowledge:

| Feature | Formula | Rationale |
|---------|---------|-----------|
| `BMI_Age` | BMI × Age | Combined obesity-age risk |
| `Glucose_Insulin` | Glucose / (Insulin+1) | Insulin resistance proxy |
| `Glucose_BMI` | Glucose × BMI | Metabolic syndrome signal |

### 3. Models Trained
| Model | Type |
|-------|------|
| Logistic Regression | Baseline linear |
| Support Vector Machine | Kernel-based |
| Random Forest | Bagging ensemble |
| Gradient Boosting | Sequential boosting |
| XGBoost | Optimized boosting |
| **Voting Ensemble** | **Soft voting (all 5)** |

### 4. Evaluation
- **5-fold Stratified Cross Validation**
- Metrics: Accuracy, F1-Score, ROC-AUC
- Threshold tuning via Precision-Recall analysis

---

## 📈 Results

| Model | Accuracy | F1 Score | ROC-AUC | CV AUC |
|-------|----------|----------|---------|--------|
| Logistic Regression | 0.779 | 0.691 | 0.847 | 0.839 ± 0.028 |
| SVM | 0.786 | 0.701 | 0.856 | 0.847 ± 0.021 |
| Random Forest | 0.805 | 0.726 | 0.867 | 0.859 ± 0.019 |
| Gradient Boosting | 0.812 | 0.731 | 0.871 | 0.863 ± 0.022 |
| XGBoost | 0.818 | 0.741 | 0.878 | 0.869 ± 0.018 |
| **Ensemble (Voting)** | **0.825** | **0.752** | **0.884** | — |

> ✅ **Ensemble outperforms all individual models on every metric**

---

## 🚀 How to Run

### 1. Clone the Repository
```bash
git clone https://github.com/SaumyaShreya1/diabetes-prediction-.git
cd diabetes-prediction-
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Add Dataset
Download `diabetes.csv` from [Kaggle](https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database) and place it in `data/`.

### 4. Train the Models
```bash
python src/pipeline.py
```

### 5. Launch the Web App
```bash
streamlit run app.py
```

### 6. Run Tests
```bash
pytest tests/ -v
```

---

## 🔍 Explainability (SHAP)

SHAP (SHapley Additive exPlanations) reveals **why** the model made each prediction.

**Key Insights from SHAP:**
- 🥇 **Glucose** is the single most important predictor
- 🥈 **BMI** and `BMI_Age` follow closely
- 🥉 **Age** and `Glucose_BMI` contribute significantly
- Low Insulin levels paradoxically *increase* risk (insulin resistance)

> SHAP makes the model **interpretable** — crucial for medical applications where trust and transparency matter.

---

## 🛠 Tech Stack

| Tool | Purpose |
|------|---------|
| **Python 3.10** | Core language |
| **Pandas / NumPy** | Data manipulation |
| **Scikit-learn** | ML algorithms, preprocessing, evaluation |
| **XGBoost** | Gradient boosting |
| **SHAP** | Model explainability |
| **Streamlit** | Web application |
| **Matplotlib / Seaborn** | Visualization |
| **Pytest** | Unit testing |
| **Joblib** | Model serialization |

---

## 👩‍💻 Author

**Saumya Shreya**  
B.Tech (Computer Science), 4th Year  
[![GitHub](https://img.shields.io/badge/GitHub-SaumyaShreya1-black?logo=github)](https://github.com/SaumyaShreya1)

---

## 📄 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

---

> ⭐ If you found this project helpful, please consider giving it a star!
