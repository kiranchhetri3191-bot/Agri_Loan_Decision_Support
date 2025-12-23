# =========================================================
# AGRICULTURAL LOAN DECISION SUPPORT SYSTEM (DSS)
# FINAL POLISHED VERSION
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
    page_icon="🌾",
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

# ---------------- SIDEBAR ----------------
st.sidebar.title("🌱 How to Use")
st.sidebar.markdown("""
1️⃣ Upload agricultural loan CSV  
2️⃣ View **risk insights**, not approval  
3️⃣ Read **improvement suggestions**  
4️⃣ Use visuals to understand patterns  

⚠️ This tool supports **learning & awareness**
""")

st.sidebar.divider()
st.sidebar.info("📌 Educational Decision Support Tool")

# ---------------- DISCLAIMER ----------------
st.markdown("""
### ⚠️ Legal & Ethical Disclaimer
This platform is a **Decision Support System (DSS)** created for:

- Education & learning  
- Farmer financial awareness  
- NGO / cooperative training  
- Policy & academic simulation  

❌ Not a bank / NBFC / RBI system  
❌ Not a loan approval authority  
❌ No real customer or credit bureau data  

**Outputs indicate risk patterns only, not decisions**
""")

st.divider()

# ---------------- TITLE ----------------
st.markdown("""
<h1 style='text-align:center;'>🌾 Agricultural Loan Risk & Advisory Dashboard</h1>
<p style='text-align:center; font-size:17px;'>
CSV Upload • Visual Insights • Improvement Guidance
</p>
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
        st.error("❌ CSV format mismatch. Please use the prescribed agricultural format.")
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
            advice.append("Explore SHG / group-based lending options")

        if row["irrigation_type"] == le_irrig.transform(["Rainfed"])[0]:
            advice.append("Irrigation support schemes may reduce risk")

        if row["existing_loans"] > 1:
            advice.append("Reduce existing loan burden")

        if not advice:
            advice.append("Profile appears financially stable")

        return " | ".join(advice)

    df["Suggested_Improvements"] = df.apply(improvement_advice, axis=1)

    st.success("✅ Risk & Advisory Analysis Completed")

    st.subheader("📋 Farmer-Level Risk & Advisory View")
    st.dataframe(df)

    # ---------------- DOWNLOAD CSV ----------------
    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button(
        "⬇️ Download Analysis CSV",
        csv,
        "agri_loan_risk_advisory.csv",
        "text/csv"
    )

    # ---------------- VISUALS ----------------
    st.divider()
    st.header("📊 Dashboard Insights")

    col1, col2 = st.columns(2)

    with col1:
        fig, ax = plt.subplots()
        df["Risk_Category"].value_counts().plot.pie(
            autopct="%1.1f%%", ax=ax
        )
        ax.set_ylabel("")
        ax.set_title("Risk Category Distribution")
        st.pyplot(fig)

    with col2:
        fig, ax = plt.subplots()
        ax.hist(df["credit_score"], bins=20)
        ax.set_title("Credit Score Distribution")
        st.pyplot(fig)

    fig, ax = plt.subplots()
    ax.scatter(df["annual_farm_income"], df["loan_amount"])
    ax.set_xlabel("Annual Farm Income")
    ax.set_ylabel("Loan Amount")
    ax.set_title("Income vs Loan Amount (Risk View)")
    st.pyplot(fig)

    # ---------------- PDF REPORT ----------------
    def generate_pdf():
        buffer = BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=A4)
        styles = getSampleStyleSheet()
        story = []

        story.append(Paragraph("Agricultural Loan Risk & Advisory Summary", styles["Title"]))
        story.append(Paragraph(
            "This report is generated for educational and analytical purposes only. "
            "It does not represent any bank or regulatory decision.",
            styles["Normal"]
        ))
        story.append(Paragraph(f"Total Records Analysed: {len(df)}", styles["Normal"]))
        story.append(Paragraph(str(df["Risk_Category"].value_counts()), styles["Normal"]))

        doc.build(story)
        buffer.seek(0)
        return buffer

    pdf = generate_pdf()
    st.download_button(
        "⬇️ Download PDF Summary",
        pdf,
        "agri_loan_risk_summary.pdf",
        "application/pdf"
    )

else:
    st.info("📌 Upload CSV to begin risk & advisory analysis")

# ---------------- FOOTER ----------------
st.divider()
st.markdown("""
### 🌾 Why This Project Matters
✔ Improves farmer financial awareness  
✔ Helps NGOs & cooperatives identify risk patterns  
✔ Supports early loan stress understanding  
✔ Ethical, explainable & legal by design  

**Built as a Decision Support Tool — not a decision maker**
""")
