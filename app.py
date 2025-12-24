# =========================================================
# AGRICULTURAL LOAN DECISION SUPPORT SYSTEM (DSS)
# FINAL POLISHED VERSION – TEXT INPUT & OUTPUT FIXED
# Safe | Legal | Educational | Visual | Impactful
# =========================================================

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from io import BytesIO
from reportlab.platypus import SimpleDocTemplate, Paragraph
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.pagesizes import A4

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Agri Loan Decision Support",
    page_icon="logo.png",
    layout="wide"
)

# ---------------- CUSTOM UI STYLE ----------------
st.markdown("""
<style>
.main {background-color: #F9FFF9;}
h1, h2, h3 {color: #2E7D32;}
.stButton>button {background-color:#2E7D32; color:white; border-radius:8px;}
.stDownloadButton>button {background-color:#1B5E20; color:white;}
</style>
""", unsafe_allow_html=True)

# ---------------- SIDEBAR ----------------
st.sidebar.title("🌱 How to Use")
st.sidebar.markdown("""
1️⃣ Upload agricultural loan CSV  
2️⃣ Input values as **text** (Rice, Rainfed, etc.)  
3️⃣ View **risk insights**, not approval  

⚠️ Educational & awareness tool only
""")

st.sidebar.info("📌 Decision Support Tool")

# ---------------- USER GUIDANCE ----------------
st.sidebar.subheader("🌾 Accepted Text Values")
st.sidebar.markdown("""
**Crop Type**
- Rice
- Wheat
- Maize
- Cotton
- Sugarcane

**Irrigation Type**
- Rainfed
- Canal
- Borewell
""")

# ---------------- SAMPLE CSV ----------------
sample_csv = pd.DataFrame({
    "farmer_age": [35],
    "land_size_acres": [2.5],
    "annual_farm_income": [350000],
    "loan_amount": [200000],
    "crop_type": ["Rice"],
    "irrigation_type": ["Rainfed"],
    "existing_loans": [1],
    "credit_score": [680]
})

st.sidebar.markdown("📄 Sample CSV Format")
st.sidebar.dataframe(sample_csv)

st.sidebar.download_button(
    "⬇️ Download Sample CSV",
    sample_csv.to_csv(index=False).encode("utf-8"),
    "sample_agri_loan_data.csv",
    "text/csv"
)

# ---------------- DISCLAIMER ----------------
st.markdown("""
### ⚠️ Legal & Ethical Disclaimer
This platform is a **Decision Support System (DSS)**.

❌ Not a bank / NBFC / RBI system  
❌ Not a loan approval authority  

**Outputs show risk patterns only**
""")

st.divider()

# ---------------- TITLE ----------------
col1, col2 = st.columns([1, 8])
with col1:
    st.image("logo.png", width=80)
with col2:
    st.markdown("""
    <h1>Agricultural Loan Risk & Advisory Dashboard</h1>
    <p>CSV Upload • Visual Insights • Improvement Guidance</p>
    """, unsafe_allow_html=True)

# ---------------- DEMO DATA ----------------
def generate_demo_data(n=7000):
    crops = ["Rice", "Wheat", "Maize", "Cotton", "Sugarcane"]
    irrigation = ["Rainfed", "Canal", "Borewell"]

    df = pd.DataFrame({
        "farmer_age": np.random.randint(21, 65, n),
        "land_size_acres": np.round(np.random.uniform(0.5, 10, n), 2),
        "annual_farm_income": np.random.randint(120000, 900000, n),
        "loan_amount": np.random.randint(50000, 500000, n),
        "crop_type": np.random.choice(crops, n),
        "irrigation_type": np.random.choice(irrigation, n),
        "existing_loans": np.random.randint(0, 3, n),
        "credit_score": np.random.randint(300, 850, n)
    })

    df["approved"] = np.where(
        (df["credit_score"] >= 650) &
        (df["annual_farm_income"] >= df["loan_amount"] * 1.3) &
        (df["land_size_acres"] >= 1) &
        (df["existing_loans"] <= 1),
        1, 0
    )
    return df

# ---------------- TRAIN MODEL ----------------
@st.cache_data
def train_model():
    data = generate_demo_data()

    le_crop = LabelEncoder()
    le_irrig = LabelEncoder()

    data["crop_type"] = le_crop.fit_transform(data["crop_type"])
    data["irrigation_type"] = le_irrig.fit_transform(data["irrigation_type"])

    X = data.drop("approved", axis=1)
    y = data["approved"]

    model = RandomForestClassifier(n_estimators=200, random_state=42)
    model.fit(X, y)

    return model, le_crop, le_irrig

model, le_crop, le_irrig = train_model()

# ---------------- CSV UPLOAD ----------------
st.sidebar.header("📤 Upload Agricultural Loan CSV")
file = st.sidebar.file_uploader("Upload CSV", type=["csv"])

if file:
    df = pd.read_csv(file)

    required_cols = [
        "farmer_age","land_size_acres","annual_farm_income",
        "loan_amount","crop_type","irrigation_type",
        "existing_loans","credit_score"
    ]

    if not all(c in df.columns for c in required_cols):
        st.error("❌ CSV format mismatch.")
        st.stop()

    # ----- KEEP TEXT FOR DISPLAY -----
    df_display = df.copy()

    # ----- ENCODE ONLY FOR MODEL -----
    df_model = df.copy()
    df_model["crop_type"] = le_crop.transform(df_model["crop_type"])
    df_model["irrigation_type"] = le_irrig.transform(df_model["irrigation_type"])

    # ----- PREDICTION -----
    df_display["Model_Output"] = model.predict(df_model)

    # ---------------- RISK CATEGORY ----------------
    def risk_label(row):
        if row["Model_Output"] == 1:
            return "Low Risk"
        elif row["credit_score"] < 550:
            return "High Risk"
        else:
            return "Medium Risk"

    df_display["Risk_Category"] = df_display.apply(risk_label, axis=1)

    # ---------------- ADVISORY ENGINE ----------------
    def improvement_advice(row):
        advice = []

        if row["credit_score"] < 600:
            advice.append("Improve credit repayment discipline")

        if row["loan_amount"] > row["annual_farm_income"] * 1.5:
            advice.append("Consider lower or phased loan amount")

        if row["land_size_acres"] < 1:
            advice.append("Explore SHG / group-based lending")

        if row["irrigation_type"] == "Rainfed":
            advice.append("Irrigation support schemes may reduce risk")

        if row["existing_loans"] > 1:
            advice.append("Reduce existing loan burden")

        if not advice:
            advice.append("Profile appears financially stable")

        return " | ".join(advice)

    df_display["Suggested_Improvements"] = df_display.apply(improvement_advice, axis=1)

    st.success("✅ Risk & Advisory Analysis Completed")
    st.dataframe(df_display)

else:
    st.info("📌 Upload CSV to begin analysis")

# ---------------- FOOTER ----------------
st.divider()
st.markdown("""
### 🌾 Why This Project Matters
✔ Improves farmer financial awareness  
✔ Helps NGOs & cooperatives identify risk patterns  
✔ Ethical, explainable & legal by design  

**Decision Support Tool — not a decision maker**
""")
