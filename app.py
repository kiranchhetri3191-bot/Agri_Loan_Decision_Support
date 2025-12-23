# =========================================================
# AGRICULTURAL LOAN DECISION SUPPORT SYSTEM (DSS)
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

# ---------------- DISCLAIMER ----------------
st.markdown("""
## ⚠️ LEGAL DISCLAIMER
This system is a **Decision Support System (DSS)** only.

- NOT a bank / NBFC / RBI system  
- NOT a loan approval authority  
- Uses **synthetic logic & demo ML**  
- No real farmer, Aadhaar, land registry or CIBIL data  
- Outputs are **risk insights**, not decisions  

**Use for education, research, NGOs & training only**
""")

st.divider()

# ---------------- TITLE ----------------
st.markdown("""
<h1 style='text-align:center; color:#2E8B57;'>🌾 Agricultural Loan Risk & Insight Platform</h1>
<p style='text-align:center;'>CSV Upload • Visual Analytics • Early Risk Detection</p>
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
        st.error("❌ Invalid CSV format")
        st.stop()

    df["crop_type"] = le_crop.transform(df["crop_type"])
    df["irrigation_type"] = le_irrig.transform(df["irrigation_type"])

    df["Risk_Prediction"] = model.predict(df)

    # Risk categories (impactful)
    def risk_label(row):
        if row["Risk_Prediction"] == 1:
            return "Low Risk"
        elif row["credit_score"] < 550:
            return "High Risk"
        else:
            return "Medium Risk"

    df["Risk_Category"] = df.apply(risk_label, axis=1)

    st.success("✅ Risk Analysis Completed")

    st.dataframe(df)

    # ---------------- DOWNLOAD CSV ----------------
    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button(
        "⬇️ Download Result CSV",
        csv,
        "agri_loan_risk_analysis.csv",
        "text/csv"
    )

    # ---------------- VISUALS ----------------
    st.header("📊 Decision Support Visuals")

    col1, col2 = st.columns(2)

    with col1:
        fig, ax = plt.subplots()
        df["Risk_Category"].value_counts().plot.pie(autopct="%1.1f%%", ax=ax)
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
    ax.set_xlabel("Income")
    ax.set_ylabel("Loan Amount")
    ax.set_title("Income vs Loan Amount (Risk View)")
    st.pyplot(fig)

    fig, ax = plt.subplots()
    df.groupby("crop_type")["Risk_Category"].value_counts().unstack().plot(kind="bar", ax=ax)
    ax.set_title("Crop-wise Risk Pattern")
    st.pyplot(fig)

    fig, ax = plt.subplots()
    df.groupby("irrigation_type")["Risk_Category"].value_counts().unstack().plot(kind="bar", ax=ax)
    ax.set_title("Irrigation Impact")
    st.pyplot(fig)

    # ---------------- PDF REPORT ----------------
    def generate_pdf():
        buffer = BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=A4)
        styles = getSampleStyleSheet()
        story = []

        story.append(Paragraph("Agricultural Loan Risk Summary (Demo)", styles["Title"]))
        story.append(Paragraph("This report is for educational & analytical purposes only.", styles["Normal"]))
        story.append(Paragraph(f"Total Records: {len(df)}", styles["Normal"]))
        story.append(Paragraph(str(df["Risk_Category"].value_counts()), styles["Normal"]))

        doc.build(story)
        buffer.seek(0)
        return buffer

    pdf = generate_pdf()
    st.download_button(
        "⬇️ Download PDF Summary",
        pdf,
        "agri_loan_risk_report.pdf",
        "application/pdf"
    )

else:
    st.info("📌 Upload CSV to start analysis")

# ---------------- FOOTER ----------------
st.divider()
st.markdown("""
### 🌱 REAL IMPACT
✔ Farmer financial awareness  
✔ NGO & co-op decision support  
✔ Early NPA warning  
✔ Policy simulation  
✔ Strong resume project (B.Com + Analytics)
""")
