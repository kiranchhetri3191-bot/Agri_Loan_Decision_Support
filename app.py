# app.py
# Smart Farmer Advisory + Loan Risk System
# Advisory + Yield + Loan Prediction + Downloads

import os
import numpy as np
import pandas as pd
import streamlit as st
import joblib
from io import BytesIO

from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# =========================================================
# DOWNLOAD HELPERS
# =========================================================

def to_excel(df):
    buffer = BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="Loan_Predictions")
    buffer.seek(0)
    return buffer.getvalue()


def to_pdf(df):
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4)
    styles = getSampleStyleSheet()
    elements = [Paragraph("Agricultural Loan Risk Report", styles["Title"])]

    table_data = [df.columns.tolist()] + df.values.tolist()
    table = Table(table_data, repeatRows=1)
    table.setStyle(TableStyle([
        ("GRID", (0, 0), (-1, -1), 0.5, colors.black),
        ("BACKGROUND", (0, 0), (-1, 0), colors.lightgreen),
        ("ALIGN", (0, 0), (-1, -1), "CENTER")
    ]))

    elements.append(table)
    doc.build(elements)
    buffer.seek(0)
    return buffer

# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="Smart Farmer Advisory & Loan Risk System",
    page_icon="🌾",
    layout="wide"
)

DATA_FILE = "agri_loan_data.csv"
MODEL_FILE = "loan_model.joblib"
RANDOM_STATE = 42

# =========================================================
# TITLE
# =========================================================

st.markdown(
    """
    <h1 style='text-align:center; color:#2E8B57;'>🌾 Smart Farmer Advisory & Loan Risk System</h1>
    <p style='text-align:center; font-size:18px;'>
    Advisory • Yield • Credit Risk • CSV / Excel Input
    </p>
    """,
    unsafe_allow_html=True
)

# =========================================================
# PART 1: FARMER ADVISORY (LEFT SIDEBAR)
# =========================================================

st.sidebar.header("👨‍🌾 Farmer Advisory Inputs")

soil_type = st.sidebar.selectbox(
    "Soil Type",
    ["Alluvial", "Black", "Red", "Laterite", "Sandy"]
)

season = st.sidebar.selectbox(
    "Season",
    ["Kharif", "Rabi", "Zaid"]
)

rainfall = st.sidebar.slider("Annual Rainfall (mm)", 200, 2000, 850)
temperature = st.sidebar.slider("Temperature (°C)", 10, 45, 30)
land_size = st.sidebar.slider("Land Size (Acres)", 1, 20, 5)

def yield_estimation(acres, rain, temp, soil, season):
    base_yield = 20

    rain_factor = 0.7 if rain < 500 else 1.1 if rain <= 1200 else 0.9
    temp_factor = 1.1 if 20 <= temp <= 35 else 0.85

    soil_factor = {
        "Alluvial": 1.2,
        "Black": 1.15,
        "Red": 1.0,
        "Laterite": 0.9,
        "Sandy": 0.8
    }[soil]

    season_factor = {
        "Kharif": 1.1,
        "Rabi": 1.0,
        "Zaid": 0.9
    }[season]

    return round(acres * base_yield * rain_factor * temp_factor * soil_factor * season_factor, 2)

yield_est = yield_estimation(land_size, rainfall, temperature, soil_type, season)

st.sidebar.markdown("### 🌱 Advisory Output")
st.sidebar.success(f"Estimated Yield: {yield_est} Quintals")

if rainfall < 500:
    st.sidebar.warning("Low rainfall: Drip irrigation recommended")
elif rainfall > 1500:
    st.sidebar.info("High rainfall: Ensure drainage")

if temperature > 38:
    st.sidebar.warning("High temperature stress possible")

if soil_type == "Sandy":
    st.sidebar.info("Sandy soil needs frequent irrigation & organic matter")

# =========================================================
# PART 2: LOAN DATA + MODEL
# =========================================================

def generate_loan_data(n=1500):
    rng = np.random.default_rng(RANDOM_STATE)
    crops = ["Rice", "Wheat", "Cotton", "Sugarcane", "Pulses"]

    df = pd.DataFrame({
        "farmer_age": rng.integers(21, 70, n),
        "land_size_acres": rng.uniform(0.5, 20, n),
        "crop_type": rng.choice(crops, n),
        "annual_income": rng.integers(100000, 1500000, n),
        "irrigation_available": rng.choice(["Yes", "No"], n),
        "existing_loan": rng.choice(["Yes", "No"], n),
        "previous_default": rng.choice(["Yes", "No"], n, p=[0.15, 0.85]),
        "credit_score": rng.integers(300, 900, n),
        "loan_amount_requested": rng.integers(50000, 2000000, n),
        "loan_tenure_years": rng.integers(1, 7, n),
    })

    score = (
        (df["credit_score"] >= 650).astype(int) +
        (df["previous_default"] == "No").astype(int) +
        (df["annual_income"] >= 300000).astype(int) +
        (df["loan_amount_requested"] <= 5 * df["annual_income"]).astype(int) +
        (df["irrigation_available"] == "Yes").astype(int)
    )

    df["loan_approved"] = np.where(score >= 3, "Yes", "No")
    return df

if not os.path.exists(DATA_FILE):
    generate_loan_data().to_csv(DATA_FILE, index=False)

df = pd.read_csv(DATA_FILE)

X = df.drop(columns=["loan_approved"])
y = df["loan_approved"]

cat_cols = X.select_dtypes(include="object").columns
num_cols = X.select_dtypes(exclude="object").columns

preprocess = ColumnTransformer([
    ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
    ("num", "passthrough", num_cols)
])

def train_model():
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=RANDOM_STATE, stratify=y
    )

    models = [
        LogisticRegression(max_iter=1000),
        RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE)
    ]

    best_model, best_acc = None, 0

    for m in models:
        pipe = Pipeline([("prep", preprocess), ("model", m)])
        pipe.fit(X_train, y_train)
        acc = accuracy_score(y_test, pipe.predict(X_test))
        if acc > best_acc:
            best_acc = acc
            best_model = pipe

    joblib.dump(best_model, MODEL_FILE)
    return best_model

model = joblib.load(MODEL_FILE) if os.path.exists(MODEL_FILE) else train_model()

# =========================================================
# PART 3: CSV / EXCEL UPLOAD
# =========================================================

st.markdown("## 📂 Loan Prediction via CSV / Excel Upload")

uploaded_file = st.file_uploader("Upload CSV or Excel file", type=["csv", "xlsx"])

if uploaded_file:
    input_data = pd.read_csv(uploaded_file) if uploaded_file.name.endswith(".csv") else pd.read_excel(uploaded_file)

    st.subheader("Uploaded Data Preview")
    st.dataframe(input_data.head())

    predictions = model.predict(input_data)
    probabilities = model.predict_proba(input_data).max(axis=1) * 100

    def risk_logic(row):
        if row["credit_score"] < 600 or row["previous_default"] == "Yes":
            return "High"
        if row["loan_amount_requested"] > 5 * row["annual_income"]:
            return "Medium"
        return "Low"

    input_data["Loan_Decision"] = predictions
    input_data["Approval_Probability_%"] = probabilities.round(2)
    input_data["Risk_Category"] = input_data.apply(risk_logic, axis=1)

    st.success("Prediction completed")
    st.dataframe(input_data)

    st.download_button("⬇️ Download CSV", input_data.to_csv(index=False), "loan_predictions.csv", "text/csv")
    st.download_button("⬇️ Download Excel", to_excel(input_data), "loan_predictions.xlsx")
    st.download_button("⬇️ Download PDF", to_pdf(input_data), "loan_predictions.pdf")

# =========================================================
# FOOTER
# =========================================================

st.markdown("---")
st.markdown(
    "<p style='text-align:center;'>Smart Farmer Advisory & Loan Risk System | Advisory + Credit Intelligence</p>",
    unsafe_allow_html=True
)
