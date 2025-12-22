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
    base_yield = 20
    factor = 1.2 if rain > 700 else 0.8
    return round(acres * base_yield * factor, 2)

# ---------------- RESULTS ----------------
crop = crop_recommendation(soil_type, season, rainfall)
fertilizer = fertilizer_advice(soil_type)
pest = pest_advice(crop_issue)
weather = weather_advice(temperature, rainfall)
yiel
