# app.py
# Smart Farmer Advisory + Loan Risk System
# Location-based Climate | Yield | Loan Prediction | Safe & Legal

import os
import requests
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
# SAFE OPEN-SOURCE WEATHER (OPEN-METEO)
# =========================================================

def get_lat_lon(place):
    url = "https://geocoding-api.open-meteo.com/v1/search"
    params = {"name": place, "count": 1, "language": "en"}
    r = requests.get(url, params=params, timeout=10)
    data = r.json()
    if "results" not in data:
        return None, None
    return data["results"][0]["latitude"], data["results"][0]["longitude"]


def get_climate(lat, lon):
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": lat,
        "longitude": lon,
        "daily": "temperature_2m_mean,precipitation_sum",
        "past_days": 30,
        "timezone": "auto"
    }
    r = requests.get(url, params=params, timeout=10)
    data = r.json()

    avg_temp = sum(data["daily"]["temperature_2m_mean"]) / 30
    total_rain = sum(data["daily"]["precipitation_sum"])

    return round(avg_temp, 1), round(total_rain, 1)

# =========================================================
# DOWNLOAD HELPERS
# =========================================================

def to_excel(df):
    buffer = BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        df.to_excel(writer, index=False)
    buffer.seek(0)
    return buffer.getvalue()


def to_pdf(df):
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4)
    styles = getSampleStyleSheet()
    elements = [Paragraph("Agricultural Loan Risk Report", styles["Title"])]

    table = Table([df.columns.tolist()] + df.values.tolist(), repeatRows=1)
    table.setStyle(TableStyle([
        ("GRID", (0, 0), (-1, -1), 0.5, colors.black),
        ("BACKGROUND", (0, 0), (-1, 0), colors.lightgreen),
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

# =========================================================
# TITLE
# =========================================================

st.markdown(
    "<h1 style='text-align:center;color:#2E8B57;'>🌾 Smart Farmer Advisory & Loan Risk System</h1>",
    unsafe_allow_html=True
)

# =========================================================
# PART 1: FARMER ADVISORY (LOCATION BASED)
# =========================================================

st.sidebar.header("📍 Location Based Advisory")

place = st.sidebar.text_input(
    "City / Town / Village",
    placeholder="e.g. Birpara, Pune, Jalpaiguri"
)

soil_type = st.sidebar.selectbox(
    "Soil Type",
    ["Alluvial", "Black", "Red", "Laterite", "Sandy"]
)

season = st.sidebar.selectbox(
    "Season",
    ["Kharif", "Rabi", "Zaid"]
)

land_size = st.sidebar.slider("Land Size (Acres)", 1, 20, 5)

def yield_estimation(acres, rain, temp, soil, season):
    base = 20
    rain_factor = 0.7 if rain < 100 else 1.1 if rain < 400 else 0.9
    temp_factor = 1.1 if 20 <= temp <= 35 else 0.85

    soil_factor = {
        "Alluvial": 1.2,
        "Black": 1.15,
        "Red": 1.0,
        "Laterite": 0.9,
        "Sandy": 0.8
    }[soil]

    season_factor = {"Kharif": 1.1, "Rabi": 1.0, "Zaid": 0.9}[season]

    return round(acres * base * rain_factor * temp_factor * soil_factor * season_factor, 2)

if place:
    lat, lon = get_lat_lon(place)

    if lat:
        temperature, rainfall = get_climate(lat, lon)

        st.sidebar.markdown("### 🌦️ Auto Climate (Last 30 Days)")
        st.sidebar.write(f"🌡️ Avg Temp: **{temperature} °C**")
        st.sidebar.write(f"🌧️ Rainfall: **{rainfall} mm**")

        yield_est = yield_estimation(land_size, rainfall, temperature, soil_type, season)
        st.sidebar.success(f"🌾 Estimated Yield: {yield_est} Quintals")

        if rainfall < 100:
            st.sidebar.warning("Low rainfall trend – irrigation recommended")
        if temperature > 38:
            st.sidebar.warning("Heat stress possible")
    else:
        st.sidebar.error("Location not found")

# =========================================================
# PART 2: LOAN MODEL
# =========================================================

DATA_FILE = "agri_loan_data.csv"
MODEL_FILE = "loan_model.joblib"

def generate_loan_data(n=1500):
    rng = np.random.default_rng(42)
    df = pd.DataFrame({
        "farmer_age": rng.integers(21, 70, n),
        "land_size_acres": rng.uniform(0.5, 20, n),
        "crop_type": rng.choice(["Rice","Wheat","Cotton","Sugarcane","Pulses"], n),
        "annual_income": rng.integers(100000, 1500000, n),
        "irrigation_available": rng.choice(["Yes","No"], n),
        "existing_loan": rng.choice(["Yes","No"], n),
        "previous_default": rng.choice(["Yes","No"], n, p=[0.15,0.85]),
        "credit_score": rng.integers(300, 900, n),
        "loan_amount_requested": rng.integers(50000, 2000000, n),
        "loan_tenure_years": rng.integers(1,7,n),
    })

    score = (
        (df.credit_score >= 650).astype(int) +
        (df.previous_default == "No").astype(int) +
        (df.annual_income >= 300000).astype(int)
    )

    df["loan_approved"] = np.where(score >= 2, "Yes", "No")
    return df

if not os.path.exists(DATA_FILE):
    generate_loan_data().to_csv(DATA_FILE, index=False)

df = pd.read_csv(DATA_FILE)
X = df.drop(columns=["loan_approved"])
y = df["loan_approved"]

preprocess = ColumnTransformer([
    ("cat", OneHotEncoder(handle_unknown="ignore"), X.select_dtypes("object").columns),
    ("num", "passthrough", X.select_dtypes(exclude="object").columns)
])

def train_model():
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, stratify=y)
    pipe = Pipeline([("prep", preprocess), ("model", RandomForestClassifier(n_estimators=200))])
    pipe.fit(Xtr, ytr)
    joblib.dump(pipe, MODEL_FILE)
    return pipe

model = joblib.load(MODEL_FILE) if os.path.exists(MODEL_FILE) else train_model()

# =========================================================
# PART 3: FILE UPLOAD & PREDICTION
# =========================================================

st.markdown("## 📂 Loan Prediction via CSV / Excel")

file = st.file_uploader("Upload CSV or Excel", type=["csv","xlsx"])

if file:
    data = pd.read_csv(file) if file.name.endswith(".csv") else pd.read_excel(file)

    data["Loan_Decision"] = model.predict(data)
    data["Approval_Probability_%"] = (model.predict_proba(data).max(axis=1) * 100).round(2)

    st.dataframe(data)

    st.download_button("⬇️ CSV", data.to_csv(index=False), "loan_predictions.csv")
    st.download_button("⬇️ Excel", to_excel(data), "loan_predictions.xlsx")
    st.download_button("⬇️ PDF", to_pdf(data), "loan_predictions.pdf")

# =========================================================
# FOOTER + LEGAL SAFETY
# =========================================================

st.markdown("---")
st.caption(
    "Disclaimer: Uses open-source weather data (Open-Meteo). "
    "Outputs are indicative and for educational/advisory purposes only."
)
