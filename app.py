"""Student Marks Prediction System Using Simple Linear Regression.

A comprehensive, industry-standard Streamlit dashboard for academic BCA Machine Learning.
Features:
- Dark navy-blue sidebar & clean light workspace
- Modern metric cards and interactive Plotly visualizations
- Single-variable Simple Linear Regression: Study_Hours (X) -> Final_Marks (y)
- Pages:
  1. 🏠 Home Dashboard
  2. 🔮 Marks Prediction
  3. 📊 Data Analysis (EDA)
  4. 🎯 Model Performance
  5. 📁 Dataset Explorer
  6. 📘 BCA Educational Corner & Viva Guide
"""

from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from utils.data_preprocessing import (
    load_and_clean,
    get_dataset_summary,
    check_data_quality,
    REQUIRED_COLUMNS,
    FEATURE_COLUMN,
    TARGET_COLUMN,
)

# -----------------------------------------------------------------------------
# Path Resolution and Page Configuration
# -----------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "student_marks_dataset.csv"
MODEL_PATH = ROOT / "artifacts/simple_linear_regression_model.pkl"
META_PATH = ROOT / "artifacts/model_metadata.json"
TEST_PATH = ROOT / "artifacts/test_results.csv"

st.set_page_config(
    page_title="Student Marks Prediction | Simple Linear Regression",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

# -----------------------------------------------------------------------------
# Professional Custom CSS Styling (Hybrid Dark Navy + Clean Light Theme)
# -----------------------------------------------------------------------------
st.markdown("""
<style>
    /* Global Application Styles */
    .stApp {
        background-color: #f8fafc;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    }
    
    /* Modern Dark Navy Sidebar */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0f172a 0%, #1e293b 100%) !important;
        border-right: 1px solid #334155;
    }
    [data-testid="stSidebar"] * {
        color: #f1f5f9 !important;
    }
    [data-testid="stSidebar"] hr {
        border-color: #334155 !important;
        margin: 1rem 0;
    }
    
    /* Hero Banner */
    .hero-container {
        background: linear-gradient(135deg, #1e3a8a 0%, #312e81 50%, #4338ca 100%);
        padding: 2rem 2.2rem;
        border-radius: 16px;
        color: white;
        margin-bottom: 1.5rem;
        box-shadow: 0 10px 25px -5px rgba(30, 58, 138, 0.2);
    }
    .hero-title {
        color: #ffffff !important;
        font-size: 2.15rem;
        font-weight: 800;
        margin: 0 0 0.5rem 0;
        letter-spacing: -0.02em;
    }
    .hero-subtitle {
        color: #e0e7ff !important;
        font-size: 1.05rem;
        font-weight: 400;
        margin: 0 0 1rem 0;
        line-height: 1.5;
    }
    
    /* Badges / Pills */
    .badge-pill {
        display: inline-block;
        background: rgba(255, 255, 255, 0.18);
        backdrop-filter: blur(8px);
        color: #ffffff !important;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.75rem;
        font-weight: 700;
        letter-spacing: 0.05em;
        margin-right: 8px;
        margin-top: 4px;
        border: 1px solid rgba(255, 255, 255, 0.25);
    }
    
    /* Metric Card Styling */
    .metric-card {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 14px;
        padding: 1.25rem 1.4rem;
        box-shadow: 0 4px 12px rgba(15, 23, 42, 0.04);
        transition: transform 0.15s ease, box-shadow 0.15s ease;
    }
    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 18px rgba(15, 23, 42, 0.07);
    }
    .metric-card-label {
        color: #64748b;
        font-size: 0.85rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.03em;
    }
    .metric-card-value {
        color: #0f172a;
        font-size: 1.85rem;
        font-weight: 800;
        margin-top: 0.35rem;
        line-height: 1.2;
    }
    .metric-card-sub {
        color: #3b82f6;
        font-size: 0.8rem;
        font-weight: 500;
        margin-top: 0.25rem;
    }
    
    /* Content Box Containers */
    .content-box {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 14px;
        padding: 1.5rem;
        margin-bottom: 1.25rem;
        box-shadow: 0 2px 8px rgba(15, 23, 42, 0.03);
    }
    
    /* Callout & Alert Banners */
    .callout-info {
        background-color: #eff6ff;
        border-left: 4px solid #3b82f6;
        padding: 1rem 1.25rem;
        border-radius: 8px;
        color: #1e3a8a;
        margin: 1rem 0;
        font-size: 0.92rem;
    }
    .callout-warning {
        background-color: #fffbeb;
        border-left: 4px solid #f59e0b;
        padding: 1rem 1.25rem;
        border-radius: 8px;
        color: #92400e;
        margin: 1rem 0;
        font-size: 0.92rem;
    }
    
    /* Prediction Score Display */
    .prediction-display {
        background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
        border-radius: 16px;
        padding: 1.75rem;
        color: white;
        text-align: center;
        box-shadow: 0 8px 24px rgba(15, 23, 42, 0.12);
    }
    .pred-score {
        font-size: 3.2rem;
        font-weight: 800;
        color: #38bdf8;
        line-height: 1;
        margin: 0.5rem 0;
    }
    .pred-badge {
        display: inline-block;
        padding: 6px 16px;
        border-radius: 20px;
        font-weight: 700;
        font-size: 0.85rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .badge-distinction { background: #10b981; color: white; }
    .badge-first { background: #3b82f6; color: white; }
    .badge-second { background: #f59e0b; color: white; }
    .badge-pass { background: #eab308; color: #1e293b; }
    .badge-fail { background: #ef4444; color: white; }
</style>
""", unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# Data and Model Loading with Caching
# -----------------------------------------------------------------------------
@st.cache_data
def load_dataset():
    """Load and preprocess the student marks dataset."""
    return load_and_clean(DATA_PATH)

@st.cache_resource
def load_trained_model():
    """Load the serialized Simple Linear Regression model artifact."""
    return joblib.load(MODEL_PATH)

@st.cache_data
def load_metadata():
    """Load model evaluation metadata and parameters."""
    with open(META_PATH, "r", encoding="utf-8") as f:
        return json.load(f)

@st.cache_data
def load_test_results():
    """Load test evaluation predictions and residuals."""
    if TEST_PATH.exists():
        return pd.read_csv(TEST_PATH)
    return pd.DataFrame()


# Guard against missing project artifacts
if not DATA_PATH.exists() or not MODEL_PATH.exists() or not META_PATH.exists():
    st.error("⚠️ Critical Project Artifacts Missing! Please ensure the dataset and model artifacts exist.")
    st.info("Run `python train_model.py` in your terminal to generate dataset, train the model, and create artifacts.")
    st.stop()

# Retrieve cached artifacts
df, audit_info = load_dataset()
model = load_trained_model()
meta = load_metadata()
test_df = load_test_results()
dataset_stats = get_dataset_summary(df)


# -----------------------------------------------------------------------------
# Sidebar Navigation and Project Branding
# -----------------------------------------------------------------------------
st.sidebar.markdown("""
<div style="text-align: center; padding: 0.5rem 0 1rem 0;">
    <div style="font-size: 2.2rem; margin-bottom: 0.25rem;">🎓</div>
    <div style="font-size: 1.35rem; font-weight: 800; color: #ffffff; letter-spacing: -0.01em;">EduPredict ML</div>
    <div style="font-size: 0.78rem; color: #94a3b8; font-weight: 500;">STUDENT MARKS PREDICTION SYSTEM</div>
</div>
""", unsafe_allow_html=True)

st.sidebar.markdown("---")

nav_choice = st.sidebar.radio(
    "MAIN NAVIGATION",
    [
        "🏠 Home Dashboard",
        "🔮 Marks Prediction",
        "📊 Data Analysis (EDA)",
        "🎯 Model Performance",
        "📁 Dataset Explorer",
        "📘 BCA Educational Guide",
    ],
    index=0,
)

st.sidebar.markdown("---")
st.sidebar.markdown("### 📌 ML System Specs")
st.sidebar.markdown(f"""
- **Algorithm:** Simple Linear Regression
- **Independent (X):** `Study_Hours`
- **Dependent (y):** `Final_Marks`
- **Split Ratio:** 80% Train / 20% Test
- **Total Samples:** {meta['records']:,} records
- **Fitted Slope ($b_1$):** {meta['slope']:.4f}
- **Fitted Intercept ($b_0$):** {meta['intercept']:.4f}
- **Test $R^2$ Score:** {meta['r2']:.4f}
""")

st.sidebar.markdown("---")
st.sidebar.caption("BCA Machine Learning Project • Python • Scikit-learn • Streamlit")


# =============================================================================
# PAGE 1: HOME DASHBOARD
# =============================================================================
if nav_choice == "🏠 Home Dashboard":
    st.markdown(f"""
    <div class="hero-container">
        <h1 class="hero-title">🎓 Student Marks Prediction System</h1>
        <p class="hero-subtitle">
            An end-to-end Machine Learning web application demonstrating <strong>Simple Linear Regression</strong>
            to model, predict, and explain expected student examination marks based on daily study hours.
        </p>
        <div>
            <span class="badge-pill">SIMPLE LINEAR REGRESSION</span>
            <span class="badge-pill">SCIKIT-LEARN 1.3+</span>
            <span class="badge-pill">STREAMLIT DASHBOARD</span>
            <span class="badge-pill">PLOTLY VISUALIZATION</span>
            <span class="badge-pill">BCA ML CURRICULUM</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Key Performance Indicators (KPI Cards)
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-card-label">Dataset Records</div>
            <div class="metric-card-value">{meta['records']:,}</div>
            <div class="metric-card-sub">Synthetically generated</div>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-card-label">Average Study Hours</div>
            <div class="metric-card-value">{dataset_stats['hours_mean']:.1f} <span style="font-size:1.1rem;font-weight:500;">hrs/day</span></div>
            <div class="metric-card-sub">Range: 0.5 – 10.0 hrs</div>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-card-label">Average Final Marks</div>
            <div class="metric-card-value">{dataset_stats['marks_mean']:.1f} <span style="font-size:1.1rem;font-weight:500;">/ 100</span></div>
            <div class="metric-card-sub">Std Dev: ±{dataset_stats['marks_std']:.1f} marks</div>
        </div>
        """, unsafe_allow_html=True)
    with col4:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-card-label">Model $R^2$ Score</div>
            <div class="metric-card-value">{meta['r2']:.3f}</div>
            <div class="metric-card-sub">84.8% Variance Explained</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Overview and Relationship Visualization
    left_col, right_col = st.columns([1.1, 1.2])

    with left_col:
        st.markdown("### 📌 Project Overview & ML Workflow")
        st.markdown("""
        This project models the quantitative dependency between a student's daily study duration and their eventual examination marks.
        
        **Machine Learning Architecture:**
        - **Algorithm:** Simple Linear Regression (Ordinary Least Squares)
        - **Independent Variable ($X$):** `Study_Hours` (Continuous input)
        - **Dependent Variable ($y$):** `Final_Marks` (Continuous target: 0–100)
        - **Equation:** $\\hat{y} = b_0 + b_1 x$
        - **Optimization Method:** Ordinary Least Squares (OLS) minimizing $\\sum (y_i - \\hat{y}_i)^2$
        """)

        st.markdown("#### 🔄 End-to-End Pipeline Steps:")
        st.markdown("""
        1. **Data Ingestion:** Load 3,200 synthetic student records.
        2. **Validation & Cleaning:** Null handling, duplicate removal, boundary verification.
        3. **Feature Isolation:** Strict single-variable selection ($X = \\text{Study\\_Hours}$).
        4. **Train-Test Split:** 80% Training ($N=2,560$), 20% Testing ($N=640$).
        5. **OLS Fitting:** Compute optimal slope ($b_1$) and intercept ($b_0$).
        6. **Model Serialization:** Save `.pkl`, metadata JSON, and test metrics to `artifacts/`.
        """)

    with right_col:
        st.markdown("### 📈 Relationship Overview (Scatter & OLS Trend)")
        # Plotly Scatter Plot with OLS Trendline
        sample_df = df.sample(min(600, len(df)), random_state=42)
        fig_home = px.scatter(
            sample_df,
            x="Study_Hours",
            y="Final_Marks",
            opacity=0.6,
            color_discrete_sequence=["#3b82f6"],
            trendline="ols",
            trendline_color_override="#dc2626",
            labels={"Study_Hours": "Study Hours per Day", "Final_Marks": "Final Examination Marks"},
            title="Study Hours vs. Final Examination Marks (OLS Trendline)",
        )
        fig_home.update_layout(
            template="plotly_white",
            height=370,
            margin=dict(l=20, r=20, t=40, b=20),
            hovermode="closest",
        )
        st.plotly_chart(fig_home, use_container_width=True)

    # Why Simple Linear Regression is Suitable
    st.markdown("""
    <div class="content-box">
        <h4 style="color: #1e3a8a; margin-top:0;">💡 Why Simple Linear Regression is Suitable for This Problem</h4>
        <p style="color: #334155; line-height: 1.6;">
            <strong>Simple Linear Regression (SLR)</strong> is the foundational supervised learning algorithm for continuous bivariate problems.
            In educational statistics, study duration has a recognized, intuitive, positive monotonic correlation with academic achievement:
            as students spend more time engaged with subject material, comprehension and test scores predictably rise.
        </p>
        <ul style="color: #334155; line-height: 1.6;">
            <li><strong>Single Feature Requirement:</strong> When predicting an outcome from exactly one causal feature, Simple Linear Regression provides the most statistically direct and mathematically transparent mapping.</li>
            <li><strong>Interpretability:</strong> The learned coefficients have immediate physical meaning: the intercept represents baseline baseline knowledge (marks with 0 study hours), and the slope represents the incremental mark gain per additional hour of daily study.</li>
            <li><strong>No Overfitting:</strong> Because the hypothesis space is constrained to a single straight line ($y = b_0 + b_1 x$), Simple Linear Regression has very low model complexity and high generalization stability.</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

    # Transparency Statement Callout
    st.markdown("""
    <div class="callout-warning">
        <strong>⚠️ Academic Dataset Transparency Statement:</strong><br>
        All 3,200 student records in this project are <em>synthetically generated</em> for academic training and demonstration purposes.
        They do not represent real human subjects, verified university examination files, or real-world academic performance.
        Academic performance is multifactorial (influenced by classroom attendance, prior foundation, health, pedagogy, exam difficulty, and nutrition)
        and cannot be reduced solely to study hours in practice.
    </div>
    """, unsafe_allow_html=True)


# =============================================================================
# PAGE 2: MARKS PREDICTION INTERFACE
# =============================================================================
elif nav_choice == "🔮 Marks Prediction":
    st.markdown("## 🔮 Predict Student Examination Marks")
    st.markdown(
        "Enter or adjust the daily study hours below to generate an immediate prediction from the trained **Simple Linear Regression** model."
    )

    pred_left, pred_right = st.columns([1, 1.25])

    with pred_left:
        st.markdown("""<div class="content-box">
            <h4 style="margin-top:0; color:#0f172a;">Input Study Hours</h4>
            <p style="color:#64748b; font-size:0.85rem; margin-bottom:12px;">
                Supported Range: <strong>0.00 to 12.00 hours per day</strong>
            </p>
        """, unsafe_allow_html=True)

        input_method = st.radio(
            "Select Input Mode:",
            ["🎚️ Interactive Slider", "🔢 Exact Number Input"],
            horizontal=True,
        )

        if input_method == "🎚️ Interactive Slider":
            effective_hours = st.slider(
                "Study Hours per Day (0.0 to 12.0 hrs):",
                min_value=0.0,
                max_value=12.0,
                value=5.0,
                step=0.25,
                help="Slide to choose daily study hours within supported range [0.0, 12.0].",
            )
        else:
            effective_hours = st.number_input(
                "Type Exact Study Hours (0.0 to 12.0 hrs):",
                min_value=0.0,
                max_value=12.0,
                value=5.0,
                step=0.25,
                format="%.2f",
                help="Enter exact decimal study hours per day within supported range [0.0, 12.0].",
            )

        predict_button = st.button("🚀 Calculate Predicted Marks", type="primary", use_container_width=True)
        if predict_button:
            st.success(f"✅ Prediction calculated for {effective_hours:.2f} study hours/day!")

        st.markdown("</div>", unsafe_allow_html=True)

        # Mathematical Calculation Walkthrough Box
        b0 = meta["intercept"]
        b1 = meta["slope"]
        raw_pred = float(b0 + (b1 * effective_hours))
        is_capped = raw_pred > 100.0
        is_floored = raw_pred < 0.0
        final_pred = float(np.clip(raw_pred, 0.0, 100.0))

        if is_capped:
            boundary_note = f"<span style='color:#dc2626; font-weight:700;'>⚠️ Capped at 100.00/100</span> (Raw equation yields {raw_pred:.2f} marks, but academic examination marks cannot exceed 100)."
        elif is_floored:
            boundary_note = f"<span style='color:#dc2626; font-weight:700;'>⚠️ Floored at 0.00/100</span> (Raw equation yields {raw_pred:.2f} marks, but marks cannot be negative)."
        else:
            boundary_note = f"<span style='color:#16a34a; font-weight:600;'>✓ Within Academic Bounds (0–100)</span>"

        st.markdown(f"""
        <div class="content-box">
            <h4 style="margin-top:0; color:#1e3a8a;">📐 Mathematical Step-by-Step Walkthrough</h4>
            <p style="color:#475569; font-size:0.9rem;">Predictions are derived directly from the fitted OLS linear equation:</p>
            <div style="background:#f1f5f9; padding:12px 14px; border-radius:8px; font-family:monospace; font-size:0.88rem; margin-bottom:10px; line-height:1.6;">
                <strong>Equation:</strong> ŷ = b₀ + (b₁ × x)<br>
                <strong>Parameters:</strong> b₀ = {b0:.4f}, b₁ = {b1:.4f}<br>
                <strong>Substitution:</strong> ŷ = {b0:.4f} + ({b1:.4f} × {effective_hours:.2f})<br>
                <strong>Intermediate:</strong> ŷ = {b0:.4f} + {b1 * effective_hours:.4f}<br>
                <strong>Raw Model Output:</strong> <strong style="color:#2563eb;">{raw_pred:.2f} marks</strong><br>
                <strong>Displayed Final Score:</strong> <strong style="color:#0f172a;">{final_pred:.2f} / 100</strong><br>
                <strong>Boundary Status:</strong> {boundary_note}
            </div>
        </div>
        """, unsafe_allow_html=True)

    with pred_right:
        # Determine Academic Grade Category based on displayed final mark
        if final_pred >= 75.0:
            badge_class = "badge-distinction"
            grade_label = "Distinction (First Class with Distinction)"
            advice = "Outstanding study investment! Consistent daily revision supports mastery of complex topics."
        elif final_pred >= 60.0:
            badge_class = "badge-first"
            grade_label = "First Class Division"
            advice = "Strong academic standing! Solid preparation creates high probability of upper-tier performance."
        elif final_pred >= 50.0:
            badge_class = "badge-second"
            grade_label = "Second Class Division"
            advice = "Moderate preparation. Increasing daily study by 1.5–2 hours can elevate performance to First Class."
        elif final_pred >= 40.0:
            badge_class = "badge-pass"
            grade_label = "Pass Division"
            advice = "Borderline passing tier. Additional focused practice and doubt clearance is recommended."
        else:
            badge_class = "badge-fail"
            grade_label = "Below Passing Threshold"
            advice = "Critical alert: Study hours are insufficient to cover the BCA syllabus comfortably."

        # Capped / Boundary sub-label
        if is_capped:
            score_annotation = f"""
            <div style="font-size:0.95rem; color:#fca5a5; margin-top:4px; font-weight:600;">
                ⚠️ CAPPED AT MAXIMUM 100 (Raw Model Output: {raw_pred:.2f} marks)
            </div>
            """
        elif is_floored:
            score_annotation = f"""
            <div style="font-size:0.95rem; color:#fca5a5; margin-top:4px; font-weight:600;">
                ⚠️ FLOORED AT MINIMUM 0 (Raw Model Output: {raw_pred:.2f} marks)
            </div>
            """
        else:
            score_annotation = f"""
            <div style="font-size:0.85rem; color:#94a3b8; margin-top:4px;">
                Raw Model Output: {raw_pred:.2f} marks (Unconstrained)
            </div>
            """

        st.markdown(f"""
        <div class="prediction-display">
            <div style="text-transform: uppercase; font-size: 0.85rem; letter-spacing: 0.05em; color: #94a3b8;">
                Predicted Examination Marks for {effective_hours:.2f} Hours/Day
            </div>
            <div class="pred-score">{final_pred:.1f} <span style="font-size:1.8rem;color:#94a3b8;">/ 100</span></div>
            {score_annotation}
            <div style="margin: 0.75rem 0;">
                <span class="pred-badge {badge_class}">{grade_label}</span>
            </div>
            <p style="color: #cbd5e1; font-size: 0.9rem; margin-top: 0.5rem;">{advice}</p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # Dynamic Regression Line with Highlighted Prediction Point
        x_domain = np.linspace(0.0, 12.0, 150)
        y_line_raw = model.predict(pd.DataFrame({"Study_Hours": x_domain}))

        fig_pred = go.Figure()

        # Observation points (subtle background scatter)
        sample_subset = df.sample(min(400, len(df)), random_state=42)
        fig_pred.add_trace(go.Scatter(
            x=sample_subset["Study_Hours"],
            y=sample_subset["Final_Marks"],
            mode="markers",
            name="Sample Observations",
            marker=dict(color="#cbd5e1", size=5, opacity=0.4),
        ))

        # Unconstrained fitted regression line
        fig_pred.add_trace(go.Scatter(
            x=x_domain,
            y=y_line_raw,
            mode="lines",
            name=f"Fitted Line: ŷ = {b0:.2f} + {b1:.2f}x",
            line=dict(color="#2563eb", width=3),
        ))

        # Maximum Exam Score Reference Line (100 marks ceiling)
        fig_pred.add_hline(
            y=100.0,
            line_dash="dot",
            line_color="#dc2626",
            line_width=1.5,
            annotation_text="Exam Maximum Ceiling (100 Marks)",
            annotation_position="bottom right",
        )

        # User's active prediction point (Raw & Capped)
        if is_capped:
            # Show raw prediction coordinate
            fig_pred.add_trace(go.Scatter(
                x=[effective_hours],
                y=[raw_pred],
                mode="markers+text",
                name="Raw Prediction (Uncapped)",
                text=[f"Raw: {raw_pred:.1f}m"],
                textposition="top center",
                marker=dict(color="#f97316", size=13, symbol="triangle-up", line=dict(color="#ffffff", width=2)),
            ))
            # Show capped coordinate at 100
            fig_pred.add_trace(go.Scatter(
                x=[effective_hours],
                y=[100.0],
                mode="markers+text",
                name="Displayed Capped Score",
                text=[f"Capped: 100.0/100"],
                textposition="bottom center",
                marker=dict(color="#dc2626", size=15, symbol="diamond", line=dict(color="#ffffff", width=2)),
            ))
        else:
            fig_pred.add_trace(go.Scatter(
                x=[effective_hours],
                y=[final_pred],
                mode="markers+text",
                name="Your Prediction Point",
                text=[f"({effective_hours:.1f}h, {final_pred:.1f}m)"],
                textposition="top left",
                marker=dict(color="#dc2626", size=14, symbol="diamond", line=dict(color="#ffffff", width=2)),
            ))

        fig_pred.update_layout(
            title=f"Regression Trajectory (Input: {effective_hours:.2f} hrs | Raw: {raw_pred:.2f}m | Displayed: {final_pred:.2f}/100)",
            xaxis_title="Study Hours per Day",
            yaxis_title="Examination Marks (0–100)",
            template="plotly_white",
            height=370,
            margin=dict(l=20, r=20, t=40, b=20),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        )

        st.plotly_chart(fig_pred, use_container_width=True)

    st.markdown("""
    <div class="callout-info">
        <strong>ℹ️ Model Estimate Interpretation:</strong><br>
        This prediction is a <em>statistical point estimate</em> derived from the Simple Linear Regression equation fitted via Ordinary Least Squares.
        It conveys the expected average marks of students studying <strong>{:.2f} hours per day</strong> within this distribution.
        Individual student outcomes naturally deviate around this central tendency due to residual variance (standard error ~ ±{:.1f} marks).
    </div>
    """.format(effective_hours, meta['rmse']), unsafe_allow_html=True)


# =============================================================================
# PAGE 3: DATA ANALYSIS (EDA)
# =============================================================================
elif nav_choice == "📊 Data Analysis (EDA)":
    st.markdown("## 📊 Exploratory Data Analysis (EDA)")
    st.markdown(
        "Analyze the empirical distributions, summary statistics, and bivariate correlation between **Study Hours** and **Final Marks**."
    )

    # Top-level Statistics Cards
    s1, s2, s3, s4 = st.columns(4)
    with s1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-card-label">Pearson Correlation ($r$)</div>
            <div class="metric-card-value">{dataset_stats['pearson_correlation']:.3f}</div>
            <div class="metric-card-sub">Strong Positive Correlation</div>
        </div>
        """, unsafe_allow_html=True)
    with s2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-card-label">Study Hours Mean ± Std</div>
            <div class="metric-card-value">{dataset_stats['hours_mean']:.2f} <span style="font-size:1.1rem;">± {dataset_stats['hours_std']:.2f}</span></div>
            <div class="metric-card-sub">Hours per day</div>
        </div>
        """, unsafe_allow_html=True)
    with s3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-card-label">Final Marks Mean ± Std</div>
            <div class="metric-card-value">{dataset_stats['marks_mean']:.2f} <span style="font-size:1.1rem;">± {dataset_stats['marks_std']:.2f}</span></div>
            <div class="metric-card-sub">Marks out of 100</div>
        </div>
        """, unsafe_allow_html=True)
    with s4:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-card-label">Data Cleanliness</div>
            <div class="metric-card-value">100%</div>
            <div class="metric-card-sub">0 Missing • 0 Duplicates</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Histograms Section
    eda_col1, eda_col2 = st.columns(2)

    with eda_col1:
        fig_hist_hours = px.histogram(
            df,
            x="Study_Hours",
            nbins=30,
            marginal="box",
            title="Distribution of Daily Study Hours",
            color_discrete_sequence=["#2563eb"],
            labels={"Study_Hours": "Study Hours per Day"},
        )
        fig_hist_hours.add_vline(
            x=dataset_stats["hours_mean"],
            line_dash="dash",
            line_color="#dc2626",
            annotation_text=f"Mean: {dataset_stats['hours_mean']:.2f}h",
        )
        fig_hist_hours.update_layout(template="plotly_white", height=380, margin=dict(l=20, r=20, t=40, b=20))
        st.plotly_chart(fig_hist_hours, use_container_width=True)

    with eda_col2:
        fig_hist_marks = px.histogram(
            df,
            x="Final_Marks",
            nbins=30,
            marginal="box",
            title="Distribution of Final Examination Marks",
            color_discrete_sequence=["#7c3aed"],
            labels={"Final_Marks": "Final Marks (0–100)"},
        )
        fig_hist_marks.add_vline(
            x=dataset_stats["marks_mean"],
            line_dash="dash",
            line_color="#dc2626",
            annotation_text=f"Mean: {dataset_stats['marks_mean']:.2f}m",
        )
        fig_hist_marks.update_layout(template="plotly_white", height=380, margin=dict(l=20, r=20, t=40, b=20))
        st.plotly_chart(fig_hist_marks, use_container_width=True)

    # Comprehensive Scatter Plot & Correlation Heatmap
    st.markdown("### 🔍 Bivariate Relationship and Correlation")
    bivar_left, bivar_right = st.columns([1.3, 1])

    with bivar_left:
        fig_scatter = px.scatter(
            df,
            x="Study_Hours",
            y="Final_Marks",
            opacity=0.45,
            trendline="ols",
            trendline_color_override="#ef4444",
            color_discrete_sequence=["#0284c7"],
            title="Study Hours vs Final Marks with OLS Trendline",
            labels={"Study_Hours": "Study Hours per Day", "Final_Marks": "Final Marks"},
        )
        fig_scatter.update_layout(template="plotly_white", height=400, margin=dict(l=20, r=20, t=40, b=20))
        st.plotly_chart(fig_scatter, use_container_width=True)

    with bivar_right:
        st.markdown("#### Correlation Matrix Heatmap")
        corr_matrix = df[["Study_Hours", "Final_Marks"]].corr()
        fig_corr = px.imshow(
            corr_matrix,
            text_auto=".3f",
            aspect="auto",
            color_continuous_scale="Blues",
            title="Pearson Correlation Matrix",
        )
        fig_corr.update_layout(height=260, margin=dict(l=20, r=20, t=40, b=20))
        st.plotly_chart(fig_corr, use_container_width=True)

        st.markdown(f"""
        <div style="background:#f8fafc; border:1px solid #e2e8f0; border-radius:8px; padding:12px; font-size:0.88rem; color:#334155;">
            <strong>Pearson Coefficient ($r$ = {dataset_stats['pearson_correlation']:.3f}):</strong><br>
            Indicates a <em>very strong positive linear relationship</em>.
            As study hours increase, student examination marks increase in direct proportionality.
        </div>
        """, unsafe_allow_html=True)

    # Descriptive Statistics Table
    st.markdown("### 📋 Parametric Descriptive Statistics")
    st.dataframe(dataset_stats["stats_table"], use_container_width=True)


# =============================================================================
# PAGE 4: MODEL PERFORMANCE & DIAGNOSTICS
# =============================================================================
elif nav_choice == "🎯 Model Performance":
    st.markdown("## 🎯 Model Performance & Diagnostic Evaluation")
    st.markdown(
        "Evaluate the trained Simple Linear Regression model using standard academic error metrics, actual-vs-predicted diagnostics, and residual distributions."
    )

    # Metric Cards
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-card-label">Mean Absolute Error (MAE)</div>
            <div class="metric-card-value">{meta['mae']:.2f} <span style="font-size:1.1rem;font-weight:500;">marks</span></div>
            <div class="metric-card-sub">Average absolute error magnitude</div>
        </div>
        """, unsafe_allow_html=True)
    with m2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-card-label">Mean Squared Error (MSE)</div>
            <div class="metric-card-value">{meta['mse']:.2f} <span style="font-size:1.1rem;font-weight:500;">marks²</span></div>
            <div class="metric-card-sub">Penalizes large errors heavily</div>
        </div>
        """, unsafe_allow_html=True)
    with m3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-card-label">Root Mean Squared (RMSE)</div>
            <div class="metric-card-value">{meta['rmse']:.2f} <span style="font-size:1.1rem;font-weight:500;">marks</span></div>
            <div class="metric-card-sub">Standard deviation of residuals</div>
        </div>
        """, unsafe_allow_html=True)
    with m4:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-card-label">R² Score (Determination)</div>
            <div class="metric-card-value">{meta['r2']:.4f}</div>
            <div class="metric-card-sub">84.8% of variance explained</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Learned Regression Parameters Container
    st.markdown(f"""
    <div class="content-box">
        <h4 style="margin-top:0; color:#1e3a8a;">📐 Learned Mathematical Model Equation</h4>
        <div style="font-size:1.3rem; font-weight:700; color:#0f172a; margin: 0.5rem 0;">
            ŷ = {meta['intercept']:.4f} + ({meta['slope']:.4f} × Study_Hours)
        </div>
        <div style="color:#475569; font-size:0.92rem; line-height:1.6;">
            <strong>Intercept (b₀ = {meta['intercept']:.4f}):</strong> Expected baseline examination mark for a student who completes 0 hours of daily study.<br>
            <strong>Slope / Coefficient (b₁ = {meta['slope']:.4f}):</strong> The rate of change — for every 1 additional hour spent studying per day, expected final marks increase by <strong>{meta['slope']:.2f} marks</strong>.
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Metric Clarification Alert
    st.markdown("""
    <div class="callout-info">
        <strong>⚠️ Crucial Academic Clarification: R² is NOT a "Percentage Accuracy" Score!</strong><br>
        In regression analysis, <strong>R² (Coefficient of Determination)</strong> quantifies the proportion of total variance in the dependent variable
        ($y$) that is predictable from the independent feature ($X$):
        <br>
        <code>R² = 1 - (SS_res / SS_tot) = 1 - (Σ(y_i - ŷ_i)² / Σ(y_i - ȳ)²)</code><br>
        An R² of <strong>0.848</strong> means that <strong>84.8% of the variability</strong> in student marks is explained by daily study hours.
        The remaining 15.2% represents unexplained variance (inherent student ability, test conditions, noise). R² can be negative for poor models.
        It must never be confused with classification accuracy.
    </div>
    """, unsafe_allow_html=True)

    # Diagnostic Plots
    if not test_df.empty:
        diag_col1, diag_col2 = st.columns(2)

        with diag_col1:
            # Actual vs Predicted Plot
            fig_act_pred = px.scatter(
                test_df,
                x="Actual_Marks",
                y="Predicted_Marks",
                opacity=0.6,
                color_discrete_sequence=["#2563eb"],
                title="Actual vs. Predicted Marks (Test Dataset: N=640)",
                labels={"Actual_Marks": "Actual Test Marks", "Predicted_Marks": "Predicted Test Marks"},
            )
            # Add ideal 45-degree parity line
            fig_act_pred.add_shape(
                type="line",
                x0=0, y0=0, x1=100, y1=100,
                line=dict(color="#dc2626", dash="dash", width=2),
            )
            fig_act_pred.update_layout(template="plotly_white", height=380, margin=dict(l=20, r=20, t=40, b=20))
            st.plotly_chart(fig_act_pred, use_container_width=True)

        with diag_col2:
            # Residual Plot (Residuals vs Predicted)
            fig_resid = px.scatter(
                test_df,
                x="Predicted_Marks",
                y="Residual",
                opacity=0.6,
                color_discrete_sequence=["#7c3aed"],
                title="Residual Plot (Residuals vs. Fitted Marks)",
                labels={"Predicted_Marks": "Fitted / Predicted Marks", "Residual": "Residual (Actual − Predicted)"},
            )
            fig_resid.add_hline(y=0, line_dash="dash", line_color="#dc2626", line_width=2)
            fig_resid.update_layout(template="plotly_white", height=380, margin=dict(l=20, r=20, t=40, b=20))
            st.plotly_chart(fig_resid, use_container_width=True)

        # Residual Distribution Histogram (Check Normality of Errors)
        st.markdown("### 📊 Residual Error Distribution (Homoscedasticity & Normality Check)")
        res_left, res_right = st.columns([1.3, 1])
        with res_left:
            fig_res_dist = px.histogram(
                test_df,
                x="Residual",
                nbins=35,
                marginal="box",
                title="Residual Error Distribution (e = y − ŷ)",
                color_discrete_sequence=["#059669"],
                labels={"Residual": "Prediction Residual (Error in Marks)"},
            )
            fig_res_dist.add_vline(x=0, line_dash="dash", line_color="#dc2626", annotation_text="Zero Error Line")
            fig_res_dist.update_layout(template="plotly_white", height=350, margin=dict(l=20, r=20, t=40, b=20))
            st.plotly_chart(fig_res_dist, use_container_width=True)

        with res_right:
            st.markdown(f"""
            <div class="content-box">
                <h4 style="margin-top:0; color:#0f172a;">Residual Statistical Summary</h4>
                <p style="color:#475569; font-size:0.9rem;">
                    Residuals represent the difference: <code>eᵢ = yᵢ - ŷᵢ</code>.
                    In linear regression, residuals should ideally be normally distributed with mean zero and constant variance (homoscedasticity).
                </p>
                <ul style="color:#334155; font-size:0.88rem; line-height:1.6;">
                    <li><strong>Residual Mean:</strong> {test_df['Residual'].mean():.4f} (≈ 0)</li>
                    <li><strong>Residual Std Dev:</strong> {test_df['Residual'].std():.4f} marks</li>
                    <li><strong>Min Residual:</strong> {test_df['Residual'].min():.2f} marks</li>
                    <li><strong>Max Residual:</strong> {test_df['Residual'].max():.2f} marks</li>
                    <li><strong>Test Split:</strong> 20% ({meta['test_records']:,} samples)</li>
                    <li><strong>Random State:</strong> {meta['random_state']} (Deterministic)</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)


# =============================================================================
# PAGE 5: DATASET EXPLORER
# =============================================================================
elif nav_choice == "📁 Dataset Explorer":
    st.markdown("## 📁 Dataset Explorer & Quality Inspection")
    st.markdown(
        "Inspect the structured dataset, search specific student records, filter by study hours and marks, and export cleaned CSV files."
    )

    # Data Quality Health Card
    data_quality = check_data_quality(df)
    c_q1, c_q2, c_q3, c_q4 = st.columns(4)
    with c_q1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-card-label">Total Valid Records</div>
            <div class="metric-card-value">{len(df):,}</div>
            <div class="metric-card-sub">Active in dataset</div>
        </div>
        """, unsafe_allow_html=True)
    with c_q2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-card-label">Missing Cells</div>
            <div class="metric-card-value">{data_quality['missing_cells']}</div>
            <div class="metric-card-sub">0% null rate</div>
        </div>
        """, unsafe_allow_html=True)
    with c_q3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-card-label">Duplicate Records</div>
            <div class="metric-card-value">{data_quality['duplicate_rows']}</div>
            <div class="metric-card-sub">Clean deduplication</div>
        </div>
        """, unsafe_allow_html=True)
    with c_q4:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-card-label">Data Hygiene Status</div>
            <div class="metric-card-value" style="color:#10b981; font-size:1.6rem;">{data_quality['quality_status']}</div>
            <div class="metric-card-sub">Schema & bounds verified</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Interactive Filter Controls
    filter_col1, filter_col2, filter_col3 = st.columns([1.2, 1.4, 1.4])
    with filter_col1:
        search_query = st.text_input("🔍 Search Student ID:", placeholder="e.g. STU0015", value="")
    with filter_col2:
        hours_filter = st.slider("Filter Study Hours Range:", 0.5, 10.0, (0.5, 10.0), step=0.5)
    with filter_col3:
        marks_filter = st.slider("Filter Final Marks Range:", 0.0, 100.0, (0.0, 100.0), step=5.0)

    # Apply Filters
    filtered_df = df[
        (df["Study_Hours"] >= hours_filter[0]) &
        (df["Study_Hours"] <= hours_filter[1]) &
        (df["Final_Marks"] >= marks_filter[0]) &
        (df["Final_Marks"] <= marks_filter[1])
    ]

    if search_query.strip():
        filtered_df = filtered_df[filtered_df["Student_ID"].str.contains(search_query.strip(), case=False, na=False)]

    st.markdown(f"**Showing {len(filtered_df):,} matching student records:**")
    st.dataframe(filtered_df, use_container_width=True, height=380)

    # Download Button
    csv_bytes = df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="⬇️ Download Complete Cleaned Dataset (CSV)",
        data=csv_bytes,
        file_name="student_marks_dataset.csv",
        mime="text/csv",
        type="primary",
    )

    st.markdown("<br>", unsafe_allow_html=True)

    # Column Schema Metadata Dictionary
    st.markdown("### 📑 Dataset Schema & Feature Dictionary")
    schema_info = pd.DataFrame({
        "Column Name": ["Student_ID", "Study_Hours", "Final_Marks"],
        "Data Type": ["String / Categorical Identifier", "Float64 (Continuous)", "Float64 (Continuous)"],
        "Role": ["Identifier (Excluded from Model)", "Independent Variable (X)", "Dependent Target Variable (y)"],
        "Valid Domain": ["STU0001 – STU3200", "0.50 – 10.00 hours/day", "0.00 – 100.00 marks"],
        "Description": [
            "Unique synthetic student identification code",
            "Average daily study hours committed by the student",
            "Final academic examination score earned by the student (out of 100)",
        ],
    })
    st.table(schema_info)


# =============================================================================
# PAGE 6: BCA EDUCATIONAL GUIDE & VIVA PREPARATION
# =============================================================================
elif nav_choice == "📘 BCA Educational Guide":
    st.markdown("## 📘 BCA Machine Learning Study Guide & Viva Preparation")
    st.markdown(
        "A rigorous, academic theoretical walkthrough explaining the 12 core concepts of Machine Learning, Regression, and Simple Linear Regression, with complete mathematical formulations and viva-voce practice."
    )

    tab1, tab2, tab3 = st.tabs(["📚 Core Theory & Concepts", "📐 Mathematical Formulations", "🎓 BCA Viva-Voce Practice"])

    with tab1:
        st.markdown("### 1. What is Machine Learning?")
        st.write("""
        Machine Learning (ML) is a subfield of Artificial Intelligence (AI) that provides systems the capability to automatically learn and improve from experience without being explicitly programmed.
        Instead of writing deterministic if-else rules, an ML algorithm extracts statistical patterns from historical data to make predictions or decisions on unseen data.
        """)

        st.markdown("### 2. What is Regression?")
        st.write("""
        In supervised machine learning, **Regression** refers to the family of algorithms designed to predict a **continuous quantitative target variable** ($y$) based on one or more input features ($X$).
        Unlike classification (which predicts discrete categorical classes like Pass/Fail or Cat/Dog), regression models output real-valued numbers such as marks, prices, temperatures, or salaries.
        """)

        st.markdown("### 3. What is Simple Linear Regression?")
        st.write("""
        **Simple Linear Regression (SLR)** is the simplest form of regression analysis. It establishes a straight-line mathematical relationship between **exactly one independent variable ($X$)** and **one dependent variable ($y$)**.
        
        In this project:
        - $X$ = Study_Hours (1 independent feature)
        - $y$ = Final_Marks (1 continuous target)
        """)

        st.markdown("### 4. Independent vs. Dependent Variables")
        st.write("""
        - **Independent Variable ($X$):** Also called the explanatory variable, input feature, predictor, or regressor. It is the variable presumed to cause or influence the outcome.
        - **Dependent Variable ($y$):** Also called the response variable, outcome, criterion, or target. Its value is dependent on changes in the independent variable.
        """)

        st.markdown("### 5. Ordinary Least Squares (OLS) Method")
        st.write("""
        Ordinary Least Squares (OLS) is the fundamental optimization technique used to fit the best linear regression line.
        It identifies the unique values of slope ($b_1$) and intercept ($b_0$) that minimize the **Residual Sum of Squares (RSS)**:
        $$\\text{RSS} = \\sum_{i=1}^{N} (y_i - \\hat{y}_i)^2$$
        """)

        st.markdown("### 6. Residuals")
        st.write("""
        A **residual** ($e_i$) is the difference between an observed actual value and the value predicted by the model:
        $$e_i = y_i - \\hat{y}_i$$
        Positive residuals mean the student achieved higher marks than the model predicted.
        Negative residuals mean the student achieved lower marks than predicted.
        """)

        st.markdown("### 7. Applications, Advantages & Limitations")
        st.markdown("""
        **Real-World Applications:**
        - Educational analytics (predicting student outcomes from study hours or attendance).
        - Economics (estimating salary based on years of work experience).
        - Real estate (estimating property price based on square footage).
        - Sales forecasting (estimating revenue based on advertising expenditure).

        **Advantages:**
        - High interpretability: easy to explain to non-technical stakeholders.
        - Low computational cost: instantaneous closed-form training.
        - Resilient against overfitting when linear assumptions hold.

        **Limitations:**
        - Susceptible to extreme outliers which distort the least-squares line.
        - Cannot model non-linear or polynomial patterns directly.
        - Oversimplification: assumes that only one factor influences the outcome.
        """)

    with tab2:
        st.markdown("### 📐 Formal Mathematical Derivations")

        st.markdown("#### 1. The Simple Linear Regression Equation")
        st.latex(r"\hat{y} = b_0 + b_1 x")
        st.write("""
        Where:
        - $\\hat{y}$ = Predicted target value (Final Marks)
        - $x$ = Independent input variable (Study Hours)
        - $b_0$ = $y$-intercept (value of $\\hat{y}$ when $x = 0$)
        - $b_1$ = Slope coefficient (change in $\\hat{y}$ per unit change in $x$)
        """)

        st.markdown("#### 2. Closed-Form Ordinary Least Squares Formulas")
        st.write("The optimal coefficients are calculated analytically via calculus (setting partial derivatives to zero):")
        st.latex(r"b_1 = \frac{\sum_{i=1}^{N} (x_i - \bar{x})(y_i - \bar{y})}{\sum_{i=1}^{N} (x_i - \bar{x})^2} = \frac{\text{Cov}(x, y)}{\text{Var}(x)}")
        st.latex(r"b_0 = \bar{y} - b_1 \bar{x}")
        st.write("""
        Where:
        - $\\bar{x}$ = Mean of independent feature ($X$)
        - $\\bar{y}$ = Mean of dependent target ($y$)
        - $N$ = Total number of observations
        """)

        st.markdown("#### 3. Standard Regression Evaluation Metrics")
        st.markdown("**1. Mean Absolute Error (MAE):**")
        st.latex(r"\text{MAE} = \frac{1}{N} \sum_{i=1}^{N} |y_i - \hat{y}_i|")
        st.caption("Measures average error magnitude in original units (marks).")

        st.markdown("**2. Mean Squared Error (MSE):**")
        st.latex(r"\text{MSE} = \frac{1}{N} \sum_{i=1}^{N} (y_i - \hat{y}_i)^2")
        st.caption("Squaring penalizes large outliers heavily; units are marks².")

        st.markdown("**3. Root Mean Squared Error (RMSE):**")
        st.latex(r"\text{RMSE} = \sqrt{\text{MSE}} = \sqrt{\frac{1}{N} \sum_{i=1}^{N} (y_i - \hat{y}_i)^2}")
        st.caption("Restores error to original units (marks) for direct interpretation.")

        st.markdown("**4. Coefficient of Determination ($R^2$):**")
        st.latex(r"R^2 = 1 - \frac{SS_{\text{res}}}{SS_{\text{tot}}} = 1 - \frac{\sum (y_i - \hat{y}_i)^2}{\sum (y_i - \bar{y})^2}")
        st.caption("Represents proportion of variance in marks explained by study hours.")

    with tab3:
        st.markdown("### 🎓 BCA Practical Viva-Voce Questions & Model Answers")

        with st.expander("Q1: What is the primary difference between Regression and Classification?"):
            st.write("""
            **Answer:**
            - **Regression** predicts a continuous, numeric quantity (e.g., examination marks from 0 to 100, price, salary).
            - **Classification** predicts a discrete, qualitative categorical class label (e.g., Pass vs. Fail, Spam vs. Not Spam).
            """)

        with st.expander("Q2: Why is R² not referred to as 'accuracy' in Linear Regression?"):
            st.write("""
            **Answer:**
            Accuracy is a classification metric defined as `(Correct Predictions / Total Predictions)`.
            In regression, predicting the exact infinite-precision floating point number is statistically improbable.
            Instead, $R^2$ (Coefficient of Determination) quantifies the percentage of variance in $y$ explained by $X$.
            It can even be negative if the model predicts worse than the horizontal mean line $\\bar{y}$.
            """)

        with st.expander("Q3: What does the slope (b₁) signify in your model?"):
            st.write(f"""
            **Answer:**
            In our trained model, $b_1 = {meta['slope']:.4f}$.
            This means that for every 1 additional hour a student studies each day, their predicted final exam mark increases by approximately **{meta['slope']:.2f} marks**.
            """)

        with st.expander("Q4: What does the intercept (b₀) signify?"):
            st.write(f"""
            **Answer:**
            In our model, $b_0 = {meta['intercept']:.4f}$.
            The intercept represents the expected baseline mark when study hours are zero ($x = 0$).
            Mathematically, it anchors the regression line at the y-axis.
            """)

        with st.expander("Q5: What are residuals, and what should their distribution look like?"):
            st.write("""
            **Answer:**
            A residual is the difference between actual mark and predicted mark: $e = y - \\hat{y}$.
            Under standard Gauss-Markov assumptions, residuals should be:
            1. Normally distributed with mean zero: $E[e] = 0$.
            2. Constant in variance across all values of $X$ (Homoscedasticity).
            3. Independent and uncorrelated.
            """)

        with st.expander("Q6: Why did you split the dataset into 80% training and 20% testing?"):
            st.write("""
            **Answer:**
            Evaluating a machine learning model on the same data it was trained on produces overly optimistic, biased estimates.
            Holding out 20% of data simulates unseen real-world scenarios, allowing us to test how well the model generalizes.
            """)

        with st.expander("Q7: How does Ordinary Least Squares (OLS) find the best line?"):
            st.write("""
            **Answer:**
            OLS minimizes the sum of squared vertical distances (residuals) between each observation point and the line:
            $\\text{RSS} = \\sum (y_i - \\hat{y}_i)^2$.
            By taking partial derivatives with respect to $b_0$ and $b_1$ and setting them to zero, we obtain exact closed-form equations.
            """)

        with st.expander("Q8: What would happen if the slope b₁ were equal to 0?"):
            st.write("""
            **Answer:**
            If $b_1 = 0$, the regression line becomes a horizontal line $\\hat{y} = b_0 = \\bar{y}$.
            This indicates that the independent variable ($X$) has zero linear relationship with the dependent variable ($y$);
            knowing study hours would provide zero predictive information about marks.
            """)
