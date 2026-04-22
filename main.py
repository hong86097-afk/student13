import streamlit as st
import joblib
import numpy as np

# Load model
with open("model.pkl", "rb") as f:
    model = joblib.load(f)

st.title("Student Study Hours Prediction")

st.sidebar.header("Input Parameters")
gender = st.sidebar.radio("Pick your gender", ["Male", "Female"])
hours = st.sidebar.slider("Choose your study hours", 0, 50, 5)

if st.sidebar.button("Predict"):
    input_data = np.array([[hours]])
    prediction = model.predict(input_data)[0]
    result = "Pass" if prediction == 1 else "Fail"
    st.success(f"Prediction: {result}")
