# ==========================================
# Carbon Footprint Prediction Web App
# ==========================================

# 1️⃣ Import Required Libraries
import streamlit as st
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt


# 2️⃣ Load Trained Model and Scaler
model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")


# 3️⃣ App Title
st.title("🌍 Carbon Emission Prediction System")
st.write("Predict CO₂ emissions based on economic indicators.")


# ==========================================
# 4️⃣ User Inputs
# ==========================================

st.sidebar.header("Input Parameters")

year = st.sidebar.number_input(
    "Year",
    min_value=1900,
    max_value=2100,
    value=2025
)

population = st.sidebar.number_input(
    "Population",
    min_value=1000000,
    value=1400000000
)

gdp = st.sidebar.number_input(
    "GDP (USD)",
    min_value=1000000000,
    value=3500000000000
)


# ==========================================
# 5️⃣ Convert Inputs to DataFrame
# ==========================================

input_data = pd.DataFrame(
    [[year, population, gdp]],
    columns=["year", "population", "gdp"]
)


# ==========================================
# 6️⃣ Feature Scaling
# ==========================================

scaled_input = scaler.transform(input_data)


# ==========================================
# 7️⃣ Prediction
# ==========================================

prediction = model.predict(scaled_input)


# ==========================================
# 8️⃣ Display Results
# ==========================================

st.subheader("Predicted CO₂ Emission")

st.success(f"Estimated CO₂ emission: {prediction[0]:.2f} million tonnes")


# ==========================================
# 9️⃣ Scenario Simulation
# ==========================================

st.subheader("Energy Reduction Simulation")

reduction = st.slider(
    "Reduce energy usage (%)",
    min_value=0,
    max_value=50,
    value=10
)

reduced_prediction = prediction[0] * (1 - reduction/100)

st.write(
    f"If energy usage decreases by {reduction}%, "
    f"estimated CO₂ emission becomes **{reduced_prediction:.2f} million tonnes**"
)


# ==========================================
# 🔟 Visualization
# ==========================================

st.subheader("Emission Comparison")

labels = ["Original Emission", "Reduced Emission"]
values = [prediction[0], reduced_prediction]

fig, ax = plt.subplots()

ax.bar(labels, values)

ax.set_ylabel("CO2 Emissions")
ax.set_title("Carbon Reduction Scenario")

st.pyplot(fig)