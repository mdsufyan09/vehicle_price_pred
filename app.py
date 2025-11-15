"""
📱 Mobile Price Range Predictor App

To install dependencies (in your venv):
---------------------------------------
pip install streamlit pandas numpy scikit-learn joblib
---------------------------------------

Then run:
streamlit run app.py
"""

import streamlit as st
import numpy as np
import joblib

# Load model
model = joblib.load("mobile_price_model.pkl")

# Title and description
st.title("📱 Mobile Price Range Predictor")
st.markdown(
    "Predict the phone's **price range** "
    "(0 = Low, 1 = Medium, 2 = High, 3 = Very High) "
    "based on its specifications."
)

# Sidebar header
st.sidebar.header("Enter Mobile Specifications")

# Helper function for Yes/No → 1/0 conversion
def yes_no_input(label):
    return 1 if st.sidebar.radio(label, ["Yes", "No"]) == "Yes" else 0

# Inputs
battery_power = st.sidebar.number_input("Battery Power (mAh)", 500, 2000, 1000)
blue = yes_no_input("Bluetooth")
clock_speed = st.sidebar.slider("Clock Speed (GHz)", 0.5, 3.0, 1.5)
dual_sim = yes_no_input("Dual SIM")
fc = st.sidebar.number_input("Front Camera (MP)", 0, 20, 5)
four_g = yes_no_input("4G")
int_memory = st.sidebar.number_input("Internal Memory (GB)", 2, 128, 32)
m_dep = st.sidebar.number_input("Mobile Depth (cm)", 0.1, 1.0, 0.5)
mobile_wt = st.sidebar.number_input("Mobile Weight (g)", 80, 250, 150)
n_cores = st.sidebar.slider("Processor Cores", 1, 8, 4)
pc = st.sidebar.number_input("Primary Camera (MP)", 0, 30, 13)
px_height = st.sidebar.number_input("Pixel Height", 0, 2000, 1000)
px_width = st.sidebar.number_input("Pixel Width", 0, 2000, 1000)
ram = st.sidebar.number_input("RAM (MB)", 256, 8000, 2000)
sc_h = st.sidebar.number_input("Screen Height (cm)", 5, 20, 10)
sc_w = st.sidebar.number_input("Screen Width (cm)", 0, 18, 7)
talk_time = st.sidebar.number_input("Talk Time (hours)", 2, 24, 10)
three_g = yes_no_input("3G")
touch_screen = yes_no_input("Touch Screen")
wifi = yes_no_input("WiFi")

# Prepare data for prediction
features = np.array([[
    battery_power, blue, clock_speed, dual_sim, fc, four_g, int_memory,
    m_dep, mobile_wt, n_cores, pc, px_height, px_width, ram, sc_h,
    sc_w, talk_time, three_g, touch_screen, wifi
]])

# Prediction
if st.button("Predict Price Range"):
    prediction = model.predict(features)[0]
    labels = [
        "Low Cost (0)",
        "Medium Cost (1)",
        "High Cost (2)",
        "Very High Cost (3)"
    ]
    st.success(f"Predicted Price Range: **{labels[prediction]}**")
s