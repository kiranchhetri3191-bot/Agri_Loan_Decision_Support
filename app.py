# =========================================================
# AGRICULTURAL LOAN DECISION SUPPORT SYSTEM (DSS)
# FINAL POLISHED VERSION – ORDER & COLOR LOCKED
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
    page_icon="logo.png",   # ✅ LOGO AS ICON
    layout="wide"
)

# ---------------- CUSTOM UI STYLE ----------------
st.markdown("""
<style>
    .main {background-color: #F9FFF9;}
    h1, h2, h3 {color: #2E7D32;}
    .stButton>button {background-color:#2E7D32; color:white; border-radius:8px;}
    .stDownloadButton>button {background-color:#1B5E20; color:white;}
    .css-1d391kg {background-color: #F1F8F4;}
</style>
""", unsafe_allow_html=True)

# ---------------- HEADER WITH LOGO ----------------
col_logo, col_title = st.columns([1, 6])

with col_logo:
    st.image("logo.png", width=90)

with col_title:
    st.markdown("""
    <h1 style='margin-bottom:0;'>Agricultural Loan Risk & Advisory Dashboard</h1>
    <p style='font-size:17px; margin-top:4px;'>
    CSV Upload • Visual Insights • Improvement Guidance
    </p>
    """, unsafe_allow_html=True)

st.divider()

# ---------------- SIDEBAR ----------------
st.sidebar.title("🌱 How to Use")
st.sidebar.markdown("""
1️⃣ Upload agricultural loan CSV  
2️⃣ View **risk insights**, not approval  
3️⃣ Read **improvement suggestions**  
4️⃣ Use visuals to understand patterns  

⚠️ Educational & awareness tool only
""")

st.sidebar.divider()
st.sidebar.info("📌 Decision Support Tool")

# ---------------- DISCLAIMER ----------------
st.markdown("""
### ⚠️ Legal & Ethical Disclaimer
This platform is a **Decision Support System (DSS)**.

❌ Not a bank / NBFC / RBI system  
❌ Not a loan approval authority  
❌ No real customer or credit bureau data  

**Outputs show risk patterns only, not decisions**
""")

st.divider()

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

    df["crop_type"] = le_crop.transform(df["crop_type"])
    df["irrigation_type"] = le_irrig.transform(df["irrigation_type"])

    df["Model_Output"] = model.predict(df)

    # ---------------- RISK CATEGORY ----------------
    def risk_label(row):
        if row["Model_Output"] == 1:
            return "Low Risk"
        elif row["credit_score"] < 550:
            return "High Risk"
        else:
            return "Medium Risk"

    df["Risk_Category"] = df.apply(risk_label, axis=1)

    # ---------------- ADVISORY ENGINE ----------------
    def improvement_advice(row):
        advice = []

        if row["credit_score"] < 600:
            advice.append("Improve credit repayment discipline")

        if row["loan_amount"] > row["annual_farm_income"] * 1.5:
            advice.append("Consider lower or phased loan amount")

        if row["land_size_acres"] < 1:
            advice.append("Explore SHG / group-based lending")

        if row["irrigation_type"] == le_irrig.transform(["Rainfed"])[0]:
            advice.append("Irrigation support schemes may reduce risk")

        if row["existing_loans"] > 1:
            advice.append("Reduce existing loan burden")

        if not advice:
            advice.append("Profile appears financially stable")

        return " | ".join(advice)

    df["Suggested_Improvements"] = df.apply(improvement_advice, axis=1)

    st.success("✅ Risk & Advisory Analysis Completed")
    st.dataframe(df)

    st.download_button(
        "⬇️ Download Analysis CSV",
        df.to_csv(index=False).encode("utf-8"),
        "agri_loan_risk_advisory.csv",
        "text/csv"
    )

    # ================= VISUALS =================
    st.divider()
    st.header("📊 Dashboard Insights")

    risk_order = ["Low Risk", "Medium Risk", "High Risk"]
    risk_colors = {
        "Low Risk": "#2E7D32",
        "Medium Risk": "#F9A825",
        "High Risk": "#C62828"
    }

    col1, col2 = st.columns(2)

    with col1:
        risk_counts = df["Risk_Category"].value_counts().reindex(risk_order, fill_value=0)
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.pie(
            risk_counts.values,
            labels=risk_counts.index,
            autopct="%1.1f%%",
            startangle=90,
            colors=[risk_colors[r] for r in risk_counts.index],
            wedgeprops={"edgecolor": "white"}
        )
        ax.set_title("Overall Risk Distribution", fontsize=14, fontweight="bold")
        st.pyplot(fig)

    with col2:
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.hist(df["credit_score"], bins=25, color="#4CAF50", edgecolor="black")
        ax.set_title("Credit Score Distribution", fontsize=14, fontweight="bold")
        ax.set_xlabel("Credit Score")
        ax.set_ylabel("Number of Farmers")
        ax.grid(axis="y", alpha=0.3)
        st.pyplot(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(
        df["annual_farm_income"],
        df["loan_amount"],
        c=df["Risk_Category"].map(risk_colors),
        alpha=0.6
    )
    ax.set_title("Income vs Loan Amount (Risk Perspective)", fontsize=14, fontweight="bold")
    ax.set_xlabel("Annual Farm Income (₹)")
    ax.set_ylabel("Loan Amount (₹)")
    ax.grid(alpha=0.3)
    st.pyplot(fig)

    st.subheader("Risk Category Count")
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(
        risk_counts.index,
        risk_counts.values,
        color=[risk_colors[r] for r in risk_counts.index]
    )
    ax.set_title("Number of Farmers by Risk Level", fontsize=14, fontweight="bold")
    ax.set_xlabel("Risk Category")
    ax.set_ylabel("Count")
    ax.grid(axis="y", alpha=0.3)
    st.pyplot(fig)

else:
    st.info("📌 Upload CSV to begin analysis")

# ---------------- FOOTER ----------------
st.divider()
st.markdown("""
### 🌾 Why This Project Matters
✔ Improves farmer financial awareness  
✔ Helps NGOs & cooperatives identify risk patterns  
✔ Supports early loan stress understanding  
✔ Ethical, explainable & legal by design  

**Decision Support Tool — not a decision maker**
""")
