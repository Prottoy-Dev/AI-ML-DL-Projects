import streamlit as st
import requests

# Set API URL (your deployed FastAPI URL)
API_URL = "https://diabetes-detection-frontend.onrender.com/predict"

st.title("Diabetes Prediction App")

st.write("Enter patient details below:")

# Input form
Pregnancies = st.number_input("Pregnancies", min_value=0, max_value=20, value=1)
Glucose = st.number_input("Glucose", min_value=0, max_value=300, value=120)
BloodPressure = st.number_input("Blood Pressure", min_value=0, max_value=200, value=70)
SkinThickness = st.number_input("Skin Thickness", min_value=0, max_value=100, value=20)
Insulin = st.number_input("Insulin", min_value=0, max_value=900, value=80)
BMI = st.number_input("BMI", min_value=0.0, max_value=100.0, value=25.0)
DiabetesPedigreeFunction = st.number_input("Diabetes Pedigree Function", min_value=0.0, max_value=2.5, value=0.5)
Age = st.number_input("Age", min_value=0, max_value=120, value=30)

# When button clicked
if st.button("Predict"):
    data = {
        "Pregnancies": Pregnancies,
        "Glucose": Glucose,
        "BloodPressure": BloodPressure,
        "SkinThickness": SkinThickness,
        "Insulin": Insulin,
        "BMI": BMI,
        "DiabetesPedigreeFunction": DiabetesPedigreeFunction,
        "Age": Age
    }
    
    # Send POST request to FastAPI
    response = requests.post(API_URL, json=data)
    
    if response.status_code == 200:
        result = response.json()
        st.success(f"Prediction: {result['result']}")
        st.write(f"Confidence: {result['confidence']}")
    else:
        st.error("Error contacting the API")
