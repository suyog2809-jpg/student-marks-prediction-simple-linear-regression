# 🎓 Student Marks Prediction System Using Simple Linear Regression

[![Python Version](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Machine Learning](https://img.shields.io/badge/ML-Simple%20Linear%20Regression-green.svg)](https://scikit-learn.org/)
[![Framework](https://img.shields.io/badge/Framework-Streamlit-red.svg)](https://streamlit.io/)
[![Interface](https://img.shields.io/badge/Visualization-Plotly-purple.svg)](https://plotly.com/)
[![Curriculum](https://img.shields.io/badge/Academic-BCA%20Major%20Project-orange.svg)](https://en.wikipedia.org/wiki/Bachelor_of_Computer_Application)

An end-to-end, academic yet industry-grade Machine Learning project and interactive Streamlit web dashboard. The application models, trains, evaluates, and predicts a student's expected continuous examination marks based strictly on their daily study hours using **Simple Linear Regression**.

---

## 📑 Table of Contents

1. [Project Overview](#1-project-overview)
2. [Project Objectives](#2-project-objectives)
3. [Algorithm & Mathematical Formulation](#3-algorithm--mathematical-formulation)
4. [Dataset Description & Transparency](#4-dataset-description--transparency)
5. [Technology Stack](#5-technology-stack)
6. [Project Folder Structure](#6-project-folder-structure)
7. [Installation & Setup Guide](#7-installation--setup-guide)
8. [Execution Instructions](#8-execution-instructions)
9. [Model Training Workflow & Pipeline](#9-model-training-workflow--pipeline)
10. [Model Evaluation Metrics & Diagnostics](#10-model-evaluation-metrics--diagnostics)
11. [Streamlit Dashboard Features](#11-streamlit-dashboard-features)
12. [Automated Testing Suite](#12-automated-testing-suite)
13. [Limitations & Real-World Caveats](#13-limitations--real-world-caveats)
14. [BCA Academic Viva-Voce Notes](#14-bca-academic-viva-voce-notes)

---

## 1. Project Overview

Predicting academic performance from educational engagement metrics is a classical problem in applied predictive modeling. This project implements a fully self-contained, reproducible **Simple Linear Regression** system with an interactive, modern dashboard designed for Bachelor of Computer Applications (BCA) curricula.

The system features:
- **Strict Single Independent Variable ($X$):** `Study_Hours` (hours/day).
- **Continuous Dependent Target ($y$):** `Final_Marks` (marks out of 100).
- **Persistent Model Artifacts:** Pre-trained model serialized via `joblib`, JSON metadata, and test evaluation splits.
- **Hybrid Dashboard Design:** Deep navy-blue sidebar, clean light workspace, Plotly visual charts, metric KPI cards, and step-by-step arithmetic breakdowns.

---

## 2. Project Objectives

- **Algorithmic Focus:** Understand, code, train, validate, and visualize **Simple Linear Regression** from scratch.
- **Transparency:** Explicitly illustrate the Ordinary Least Squares (OLS) closed-form calculation:
  $$\hat{y} = b_0 + b_1 x$$
- **Explainability:** Walk students through the geometric and arithmetic meaning of the slope ($b_1$) and intercept ($b_0$).
- **Statistical Rigor:** Distinguish properly between regression evaluation metrics ($R^2$, MAE, MSE, RMSE) and classification accuracy.
- **Professional Engineering:** Modular code separation across preprocessing (`utils/`), dataset synthesis (`generate_data.py`), model fitting (`train_model.py`), testing (`tests/`), and presentation (`app.py`).

---

## 3. Algorithm & Mathematical Formulation

The algorithm used throughout this project is **Simple Linear Regression** via Scikit-Learn (`sklearn.linear_model.LinearRegression`).

### Mathematical Equation
$$\hat{y} = b_0 + b_1 x$$

Where:
- $\hat{y}$ = Raw model predicted examination marks ($\hat{y} = b_0 + b_1 x$).
- $x$ = Daily study hours (Supported range: $0.00$ to $12.00$ hours/day).
- $b_0$ = $y$-intercept ($18.5471$ — baseline marks for zero study hours).
- $b_1$ = Regression slope coefficient ($7.0141$ — expected mark gain per additional daily study hour).
- **Displayed Final Score:** When $x > 11.6$ hours, the unconstrained linear formula mathematically yields $\hat{y} > 100.00$ marks. In the UI and evaluation, the raw output is transparently preserved and reported, while the displayed examination mark is explicitly capped at $100.00 / 100$ to reflect academic examination realities.

### Ordinary Least Squares (OLS) Optimization
OLS identifies the line that minimizes the sum of squared differences (Residual Sum of Squares, RSS) between actual marks ($y_i$) and predicted marks ($\hat{y}_i$):

$$\text{RSS} = \sum_{i=1}^{N} (y_i - \hat{y}_i)^2 = \sum_{i=1}^{N} [y_i - (b_0 + b_1 x_i)]^2$$

The analytical closed-form solutions are:
$$b_1 = \frac{\sum_{i=1}^{N} (x_i - \bar{x})(y_i - \bar{y})}{\sum_{i=1}^{N} (x_i - \bar{x})^2} = \frac{\text{Cov}(x, y)}{\text{Var}(x)}$$

$$b_0 = \bar{y} - b_1 \bar{x}$$

### Trained Model Parameters
- **Intercept ($b_0$):** `18.5471`
- **Slope ($b_1$):** `7.0141`
- **Learned Equation:** `Final_Marks = 18.5471 + (7.0141 × Study_Hours)`

---

## 4. Dataset Description & Transparency

The project operates on a dataset containing **3,200 student records** saved at `student_marks_dataset.csv`.

### Data Columns
| Column Name | Data Type | Role | Value Range | Description |
| :--- | :--- | :--- | :--- | :--- |
| **`Student_ID`** | String / Text | Identifier | `STU0001` – `STU3200` | Unique synthetic identifier |
| **`Study_Hours`** | Float64 | Independent Feature ($X$) | $0.50$ – $10.00$ hrs/day | Average daily study duration |
| **`Final_Marks`** | Float64 | Dependent Target ($y$) | $0.00$ – $100.00$ marks | Final exam score out of 100 |

### ⚠️ Dataset Transparency & Ethics Statement
> **All records in `student_marks_dataset.csv` are synthetically generated for academic demonstration purposes.**
> They do not represent real human subjects, verified university examination files, or empirical student records.
> Academic performance is multifactorial (attendance, health, pedagogy, previous baseline knowledge, exam difficulty) and cannot be predicted with certainty by study hours alone.

---

## 5. Technology Stack

- **Programming Language:** Python 3.10+
- **Machine Learning Engine:** Scikit-Learn (`sklearn.linear_model.LinearRegression`)
- **Data Manipulation:** Pandas (`DataFrame`, `Series`, CSV parsing)
- **Scientific Computing:** NumPy (vector operations, numeric bounds)
- **Interactive Web Interface:** Streamlit 1.36+
- **Interactive Visualizations:** Plotly Express & Plotly Graph Objects
- **Trendline & Econometrics:** Statsmodels
- **Model Serialization:** Joblib

---

## 6. Project Folder Structure

```text
Student_Marks_Prediction/
│
├── app.py                          # Streamlit interactive multi-page dashboard
├── train_model.py                  # Model training and artifact generation pipeline
├── generate_data.py                # Standalone synthetic dataset generator
├── student_marks_dataset.csv       # 3,200 structured demonstration records
├── requirements.txt                # Pinned production dependencies
├── README.md                       # Comprehensive documentation
├── .gitignore                      # Git exclusion rules
│
├── artifacts/                      # Serialized model and diagnostics
│   ├── simple_linear_regression_model.pkl   # Serialized scikit-learn model
│   ├── model_metadata.json                  # Model hyperparams and metric scores
│   └── test_results.csv                     # 640 test evaluations with residuals
│
├── utils/                          # Modular helper functions
│   ├── __init__.py                 # Package initializer
│   └── data_preprocessing.py       # Data validation, cleaning, and statistics
│
└── tests/                          # Automated verification suite
    ├── __init__.py                 # Package initializer
    └── test_prediction.py          # Automated unit and smoke test suite
```

---

## 7. Installation & Setup Guide

### Prerequisites
- Python 3.10 or higher installed on your system.
- `pip` package manager.

### Step 1: Clone or Extract the Project
Extract `Student_Marks_Prediction_Simple_Linear_Regression.zip` or navigate to the directory:
```bash
cd Student_Marks_Prediction
```

### Step 2: Create a Virtual Environment (Recommended)

#### On macOS / Linux:
```bash
python3 -m venv .venv
source .venv/bin/activate
```

#### On Windows (PowerShell):
```powershell
py -m venv .venv
.venv\Scripts\Activate.ps1
```

#### On Windows (Command Prompt):
```cmd
py -m venv .venv
.venv\Scripts\activate.bat
```

### Step 3: Install Required Dependencies
```bash
pip install -r requirements.txt
```

---

## 8. Execution Instructions

### A. Run the Interactive Streamlit Dashboard
```bash
streamlit run app.py
```
Streamlit will launch a local web server (usually at `http://localhost:8501`). Your default browser will open automatically.

### B. Regenerate Dataset (Optional)
```bash
python generate_data.py
```

### C. Retrain the Machine Learning Model (Optional)
```bash
python train_model.py
```

### D. Execute the Automated Test Suite
```bash
python tests/test_prediction.py
```

---

## 9. Model Training Workflow & Pipeline

The pipeline in `train_model.py` executes in 11 structured steps:

1. **Dataset Loading:** Ingest `student_marks_dataset.csv`.
2. **Schema & Quality Verification:** Validate column existence, type safety, missing values, and numeric ranges via `utils/data_preprocessing.py`.
3. **Single Feature Isolation:** Ensure $X$ is strictly `df[["Study_Hours"]]` (shape: $N \times 1$).
4. **Target Isolation:** Ensure $y$ is strictly `df["Final_Marks"]`.
5. **Train-Test Split:** Partition data into 80% Training ($N=2,560$) and 20% Testing ($N=640$) with `random_state=42`.
6. **Model Instantiation:** Initialize `sklearn.linear_model.LinearRegression()`.
7. **Model Fitting:** Execute Ordinary Least Squares fitting on training split.
8. **Inference on Test Set:** Generate predictions $\hat{y}$ on the 640 unseen test cases.
9. **Error Metric Calculation:** Compute MAE, MSE, RMSE, and $R^2$.
10. **Model Persistence:** Save model binary to `artifacts/simple_linear_regression_model.pkl`.
11. **Metadata Serialization:** Write `artifacts/model_metadata.json` and `artifacts/test_results.csv`.

---

## 10. Model Evaluation Metrics & Diagnostics

The trained model achieved the following performance on the 20% held-out test dataset ($N=640$):

| Metric | Formulated Definition | Test Result | Interpretation |
| :--- | :--- | :--- | :--- |
| **$R^2$ Score** | $1 - \frac{\sum (y_i - \hat{y}_i)^2}{\sum (y_i - \bar{y})^2}$ | **`0.8481`** | **84.81% of variance** in final marks is explained by study hours. |
| **MAE** | $\frac{1}{N} \sum \|y_i - \hat{y}_i\|$ | **`6.82 marks`** | On average, predictions deviate by approximately 6.82 marks. |
| **MSE** | $\frac{1}{N} \sum (y_i - \hat{y}_i)^2$ | **`73.17 marks²`** | Mean squared error across all 640 test samples. |
| **RMSE** | $\sqrt{\text{MSE}}$ | **`8.55 marks`** | Standard deviation of prediction residuals is 8.55 marks. |

> ### 💡 Note on $R^2$ Score
> $R^2$ is the **Coefficient of Determination**, NOT a classification percentage accuracy.
> It quantifies the reduction in variance achieved by using the regression line rather than simply predicting the mean examination mark ($\bar{y}$).

---

## 11. Streamlit Dashboard Features

The dashboard includes 6 fully developed pages accessible via the navy-blue sidebar:

1. **🏠 Home Dashboard:**
   - Hero banner and high-level project KPIs.
   - Project overview and pipeline architecture.
   - Interactive Study Hours vs. Marks scatter plot with OLS trendline.
   - Educational rationale for Simple Linear Regression.
   - Dataset transparency disclosure.

2. **🔮 Marks Prediction:**
   - Dual input selector: "🎚️ Interactive Slider" and "🔢 Exact Number Input" across the supported range ($0.00$ to $12.00$ hours/day).
   - Dual score reporting: Raw unconstrained model output ($\hat{y}_{\text{raw}}$) and displayed final examination score ($0$ to $100$), with prominent capping label if raw score exceeds 100.
   - Academic classification badges (Distinction $\ge 75$, First Class $60-74$, Second Class $50-59$, Pass $40-49$, Below Pass $< 40$).
   - Step-by-step mathematical arithmetic walkthrough showing exact coefficient substitution:
     $$\hat{y} = 18.5471 + (7.0141 \times \text{hours})$$
   - Interactive Plotly chart featuring the unconstrained OLS line, 100-mark ceiling line, and active prediction coordinate markers.

3. **📊 Data Analysis (EDA):**
   - Univariate histograms for Study Hours and Final Marks with mean lines and box plots.
   - Bivariate scatter plot with OLS trendline.
   - Pearson correlation matrix heatmap ($r = 0.921$).
   - Parametric descriptive statistics table (mean, std, min, percentiles, skewness).

4. **🎯 Model Performance:**
   - Metric cards for MAE, MSE, RMSE, and $R^2$.
   - Mathematical regression formula display.
   - Actual vs. Predicted scatter plot with $45^\circ$ reference line.
   - Residual plot (Residuals vs. Fitted values) with zero-error line.
   - Residual distribution histogram checking for Gaussian error distribution.

5. **📁 Dataset Explorer:**
   - Real-time search by `Student_ID`.
   - Dual range sliders to filter records by study hours and marks.
   - Data hygiene and cleanliness audit report.
   - Feature dictionary describing schema.
   - One-click CSV export button.

6. **📘 BCA Educational Guide:**
   - Comprehensive theoretical reference covering the 12 core ML & regression topics.
   - Mathematical derivations of OLS closed-form formulas.
   - Top 8 BCA viva-voce examination questions with model answers.

---

## 12. Automated Testing Suite

The project includes an automated test suite located at `tests/test_prediction.py`:

```bash
python tests/test_prediction.py
```

### Test Coverage:
1. `test_dataset_exists_and_valid`: Validates schema, row count within 3,000–5,000, zero nulls, and numeric boundaries.
2. `test_preprocessing_module`: Tests `load_and_clean`, schema validation, and health checks.
3. `test_model_loading_and_architecture`: Asserts that the model loads, has exactly one coefficient (`coef_.shape == (1,)`), and has a positive slope.
4. `test_model_prediction_sanity`: Confirms monotonicity (more hours = higher marks) and boundary clipping within $[0, 100]$.
5. `test_metadata_and_metrics`: Verifies all metadata keys, metric bounds, and train/test record totals.
6. `test_test_results_csv`: Verifies column names and confirms residual arithmetic: $\text{Residual} = \text{Actual} - \text{Predicted}$.

---

## 13. Limitations & Real-World Caveats

- **Single Feature Simplicity:** Simple Linear Regression assumes marks are solely driven by study hours. In reality, attendance, teacher effectiveness, prior foundational knowledge, nutrition, and mental health significantly impact student grades.
- **Linearity Assumption:** The model assumes constant marginal returns for study time. In real life, extreme study hours (>12 hours/day) may cause fatigue and diminishing returns (non-linear).
- **Outlier Sensitivity:** Ordinary Least Squares squares residuals, making it sensitive to extreme anomalies.

---

## 14. BCA Academic Viva-Voce Notes

When presenting this project for your practical examination or viva, keep these key points in mind:
1. **Why Simple Linear Regression?** Because we have exactly one independent variable ($X$) and one continuous dependent target ($y$).
2. **What does the slope mean?** For every 1 extra hour studied daily, the student is expected to score $7.01$ additional marks.
3. **What does the intercept mean?** A student studying 0 hours has an estimated baseline mark of $18.55$.
4. **Why is R² not accuracy?** Accuracy is for discrete classification; $R^2$ represents the percentage of target variance explained by the model ($84.8\%$).

---

## 15. License

Developed for educational and academic demonstration purposes. Free to use, adapt, and study for BCA / B.Tech / MCA curricula.
