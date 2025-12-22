import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Advanced Farmer Advisory System",
    page_icon="🌾",
    layout="wide"
)

# ---------------- TITLE ----------------
st.markdown(
    """
    <h1 style='text-align:center; color:#2E8B57;'>🌾 Smart Farmer Advisory System</h1>
    <p style='text-align:center; font-size:18px;'>
    Crop • Weather • Fertilizer • Yield • Market • Graphs
    </p>
    """,
    unsafe_allow_html=True
)

# ---------------- SIDEBAR INPUT ----------------
st.sidebar.header("👨‍🌾 Farmer Inputs")

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

crop_issue = st.sidebar.selectbox(
    "Crop Problem",
    ["None", "Pest Attack", "Yellow Leaves", "Low Yield"]
)

# ---------------- CORE LOGIC ----------------
def crop_recommendation(soil, season, rain):
    if season == "Kharif":
        return "Rice, Cotton, Maize" if rain > 700 else "Millets, Pulses"
    elif season == "Rabi":
        return "Wheat, Mustard" if soil in ["Alluvial", "Red"] else "Gram"
    else:
        return "Watermelon, Vegetables"

def fertilizer_advice(soil):
    return {
        "Black": "Nitrogen & Phosphorus",
        "Red": "Organic manure + Potash",
        "Alluvial": "Balanced NPK",
        "Laterite": "Organic compost",
        "Sandy": "Frequent organic fertilizer"
    }[soil]

def pest_advice(issue):
    return {
        "Pest Attack": "Neem oil or bio-pesticides",
        "Yellow Leaves": "Nitrogen deficiency – apply urea",
        "Low Yield": "Check soil & irrigation",
        "None": "No pest issues detected"
    }[issue]

def weather_advice(temp, rain):
    if temp > 35:
        return "Heat stress – increase irrigation"
    elif rain < 400:
        return "Low rainfall – use drip irrigation"
    else:
        return "Weather is favorable"

def yield_estimation(acres, rain):
    base_yield = 20  # quintals per acre
    factor = 1.2 if rain > 700 else 0.8
    return round(acres * base_yield * factor, 2)

# ---------------- CALCULATIONS ----------------
crop = crop_recommendation(soil_type, season, rainfall)
fertilizer = fertilizer_advice(soil_type)
pest = pest_advice(crop_issue)
weather = weather_advice(temperature, rainfall)
yield_est = yield_estimation(land_size, rainfall)

# ---------------- SUMMARY TABLE ----------------
st.markdown("## 📋 Advisory Summary")

summary_df = pd.DataFrame({
    "Category": [
        "Recommended Crops",
        "Fertilizer Advice",
        "Pest Advisory",
        "Weather Advisory",
        "Estimated Yield (Quintals)"
    ],
    "Details": [
        crop,
        fertilizer,
        pest,
        weather,
        yield_est
    ]
})

st.table(summary_df)

# ---------------- DASHBOARD CARDS ----------------
st.markdown("## 📊 Quick Insights")
c1, c2, c3, c4 = st.columns(4)

c1.success(f"🌾 Crops\n\n{crop}")
c2.info(f"🧪 Fertilizer\n\n{fertilizer}")
c3.warning(f"🐛 Pest\n\n{pest}")
c4.success(f"📦 Yield\n\n{yield_est} Qt")

# ---------------- GRAPHS ----------------
st.markdown("## 📈 Analytical Graphs")

col1, col2 = st.columns(2)

# Rainfall Suitability Graph
with col1:
    st.subheader("🌧️ Rainfall Suitability")
    rain_levels = ["Low", "Moderate", "High"]
    values = [300, 800, 1500]

    plt.figure()
    plt.bar(rain_levels, values)
    plt.axhline(rainfall)
    st.pyplot(plt)

# Temperature Advisory Graph
with col2:
    st.subheader("🌡️ Temperature Analysis")
    temps = np.arange(10, 46)
    stress = np.where((temps < 15) | (temps > 35), 1, 0)

    plt.figure()
    plt.plot(temps, stress)
    st.pyplot(plt)

# Soil Health Graph
st.markdown("## 🌱 Soil Health Index")

soil_health = {
    "Alluvial": 85,
    "Black": 80,
    "Red": 70,
    "Laterite": 65,
    "Sandy": 60
}

plt.figure()
plt.bar(soil_health.keys(), soil_health.values())
st.pyplot(plt)

# Market Price Trend (Sample)
st.markdown("## 💰 Crop Market Price Trend (Sample Data)")

months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun"]
price = [1800, 1850, 1900, 2000, 2100, 2050]

plt.figure()
plt.plot(months, price)
st.pyplot(plt)

# ---------------- FOOTER ----------------
st.markdown("---")
st.markdown(
    "<p style='text-align:center;'>Advanced Farmer Advisory System | Python + Streamlit</p>",
    unsafe_allow_html=True
)
