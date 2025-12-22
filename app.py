# Farmer Profitability & Price Risk Simulator
# Author: (Your Name)
# Description: Python-based crop profitability and risk analysis using simulation

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# -----------------------------
# 1. Crop Data
# -----------------------------
crop_data = {
    "Crop": ["Rice", "Wheat", "Maize", "Cotton"],
    "Seed_Cost": [1200, 1000, 900, 1500],
    "Fertilizer_Cost": [2500, 2200, 2000, 3500],
    "Labour_Cost": [6000, 5000, 4500, 8000],
    "Irrigation_Cost": [3000, 2500, 2000, 4000],
    "Pesticide_Cost": [1800, 1500, 1200, 2500],
    "Yield_Quintal": [25, 22, 30, 18]
}

df = pd.DataFrame(crop_data)

# -----------------------------
# 2. Price History (₹ per quintal)
# -----------------------------
price_history = {
    "Rice": [1900, 2100, 2200, 2000, 2300],
    "Wheat": [2200, 2400, 2350, 2500, 2450],
    "Maize": [1600, 1700, 1800, 1750, 1650],
    "Cotton": [6000, 6500, 6200, 6800, 7000]
}

#
