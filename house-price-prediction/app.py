import streamlit as st
import pandas as pd
import numpy as np

from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor

# ---------------------------
# Load Data
# ---------------------------
data = pd.read_csv("housing.csv")

# ---------------------------
# Preprocessing
# ---------------------------
data = data.dropna()

label_encoder = LabelEncoder()
for column in data.select_dtypes(include=['object']).columns:
    data[column] = label_encoder.fit_transform(data[column])

# Features and target
X = data.drop("price", axis=1)
y = data["price"]

# Train model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X, y)

# ---------------------------
# UI
# ---------------------------
st.title("🏠 House Price Prediction App")

st.write("Enter house details to predict the price:")

# User inputs
size = st.number_input("Size (sq ft)", min_value=500, max_value=5000, value=1000)
bedrooms = st.slider("Bedrooms", 1, 6, 3)
bathrooms = st.slider("Bathrooms", 1, 5, 2)
location = st.selectbox("Location", ["urban", "suburban", "rural"])
age = st.slider("Age of house", 0, 30, 5)

# Encode location
location_encoded = label_encoder.fit_transform([location])[0]

# Prediction
if st.button("Predict Price"):
    input_data = np.array([[size, bedrooms, bathrooms, location_encoded, age]])
    prediction = model.predict(input_data)

    st.success(f"Estimated House Price: ${prediction[0]:,.2f}")
