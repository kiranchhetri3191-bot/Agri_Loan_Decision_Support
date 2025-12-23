# app.py
# Smart Farmer Advisory & Loan Risk Indicator System
# SAFE | OPEN-SOURCE | NON-BINDING | EDUCATIONAL

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
from sklearn.ensemble import RandomForestClassifier

# =========================================================
# SAFE LOCATION SEARCH (OPENSTREETMAP – NON-PERSONAL)
# =========================================================

def search_locations(query, state):
    url = "https://nominatim.openstreetmap.org/search"
    params = {
        "q": f"{query}, {state}, India",
        "format": "json",
        "limit": 5
    }
    headers = {"User-Agent": "SmartFarmer-Advisory"}
    r = requests.get(url, params=params, headers=headers, timeout=10)
    return r.json() if r.status_code == 200 else []

# =========================================================
# SAFE WEATHER DATA (OPEN-METEO – INDICATIVE)
# =========================================================

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
    d = r.json()

    avg_temp = np.mean(d["daily"]["temperature_2m_mean"])
    total_rain = np.sum(d["daily"]["precipitation_sum"])

    return round(avg_temp, 1), round(total_rain, 1)

# =========================================================
# SAFE EXPORT HELPERS
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
    elements = [Paragraph("Indicative Loan Risk Report", styles["Title"])]

    table = Table([df.columns.tolist()] + df.values.tolist())
    table.setStyle(TableStyle([
        ("GRID", (0,0), (-1,-1), 0.5, colors.black),
        ("BACKGROUND", (0,0), (-1,0), colors.lightgrey)
    ]))

    elements.append(table)
    doc.build(elements)
    buffer.seek(0)
    return buffer

# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="Smart Farmer Advisory System",
    page_icon="🌾",
    layout="wide"
)

st.markdown(
    "<h1 style='text-align:center;color:#2E8B57;'>🌾 Smart Farmer Advisory System</h1>",
    unsafe_allow_html=True
)

# =========================================================
# LOCATION-BASED ADVISORY (SAFE)
# =========================================================

st.sidebar.header("📍 Location Advisory (India)")

state = st.sidebar.selectbox(
    "State",
    [
        "Andhra Pradesh","Assam","Bihar","Chhattisgarh","Delhi",
        "Gujarat","Haryana","Himachal Pradesh","Jharkhand",
        "Karnataka","Kerala","Madhya Pradesh","Maharashtra",
        "Odisha","Punjab","Rajasthan","Tamil Nadu","Telangana",
        "Uttar Pradesh","Uttarakhand","West Bengal"
    ]
)

search_text = st.sidebar.text_input(
    "Search Village / Town / City",
    placeholder="Start typing location name..."
)

locations = search_locations(search_text, state) if search_text else []

if locations:
    options = {loc["display_name"]: loc for loc in locations}
    selected = st.sidebar.selectbox("Select Location", list(options.keys()))
    lat = float(options[selected]["lat"])
    lon = float(options[selected]["lon"])

    temp, rain = get_climate(lat, lon)

    st.sidebar.markdown("### 🌦 Regional Climate Indicator")
    st.sidebar.write(f"🌡 Avg Temperature: {temp} °C")
    st.sidebar.write(f"🌧 Rainfall (30 days): {rain} mm")

    if rain < 100:
        st.sidebar.info("Rainfall Trend: Low")
    elif rain < 400:
        st.sidebar.info("Rainfall Trend: Moderate")
    else:
        st.sidebar.info("Rainfall Trend: High")

# =========================================================
# LOAN RISK INDICATOR MODEL (NON-DECISION)
# =========================================================

DATA_FILE = "agri_loan_data.csv"
MODEL_FILE = "loan_model.joblib"

def generate_data(n=1500):
    rng = np.random.default_rng(42)
    df = pd.DataFrame({
        "land_size_acres": rng.uniform(0.5, 20, n),
        "annual_income": rng.integers(100000, 1500000, n),
        "credit_score": rng.integers(300, 900, n),
        "previous_default": rng.choice(["Yes","No"], n, p=[0.15,0.85]),
    })

    df["risk_flag"] = np.where(
        (df.credit_score > 650) & (df.previous_default == "No"),
        "Lower Risk",
        "Higher Risk"
    )
    return df

if not os.path.exists(DATA_FILE):
    generate_data().to_csv(DATA_FILE, index=False)

df = pd.read_csv(DATA_FILE)
X = df.drop(columns=["risk_flag"])
y = df["risk_flag"]

prep = ColumnTransformer([
    ("cat", OneHotEncoder(handle_unknown="ignore"), ["previous_default"]),
    ("num", "passthrough", ["land_size_acres","annual_income","credit_score"])
])

pipe = Pipeline([
    ("prep", prep),
    ("model", RandomForestClassifier(n_estimators=150, random_state=42))
])

pipe.fit(X, y)
joblib.dump(pipe, MODEL_FILE)

# =========================================================
# FILE UPLOAD (INDICATIVE OUTPUT)
# =========================================================

st.markdown("## 📂 Loan Risk Indicator (CSV / Excel)")

file = st.file_uploader("Upload CSV or Excel", ["csv","xlsx"])

if file:
    data = pd.read_csv(file) if file.name.endswith(".csv") else pd.read_excel(file)

    data["Indicative_Risk_Category"] = pipe.predict(data)
    data["Confidence_%"] = (pipe.predict_proba(data).max(axis=1)*100).round(2)

    st.dataframe(data)

    st.download_button("⬇ CSV", data.to_csv(index=False), "risk_indicator.csv")
    st.download_button("⬇ Excel", to_excel(data), "risk_indicator.xlsx")
    st.download_button("⬇ PDF", to_pdf(data), "risk_indicator.pdf")

# =========================================================
# LEGAL DISCLAIMER (STRONG)
# =========================================================

st.markdown("---")
st.caption(
    "Disclaimer: This system provides general, non-binding advisory insights using publicly "
    "available open-source data. Outputs are indicative only and do not constitute financial, "
    "agricultural, or legal advice. Users should consult qualified professionals before making decisions."
)
