from fastapi import FastAPI
import numpy as np
import joblib
from .schemas import Patient, PredictionOut, ClassReport
from sklearn.metrics import classification_report
import os

# Initialize FastAPI
app = FastAPI(title="Diabetes Prediction API")

# Load model, scaler, and test data
model = joblib.load(os.path.join("model", "diabetes_model.joblib"))
scaler = joblib.load(os.path.join("model", "scaler.joblib"))
X_test, y_test = joblib.load(os.path.join("model", "test_data.joblib"))


# Health check endpoint
@app.get("/health")
async def health():
    return {"status": "ok"}


# Prediction endpoint
@app.post("/predict", response_model=PredictionOut)
async def predict(patient: Patient):
    data = np.array([[
        patient.Pregnancies,
        patient.Glucose,
        patient.BloodPressure,
        patient.SkinThickness,
        patient.Insulin,
        patient.BMI,
        patient.DiabetesPedigreeFunction,
        patient.Age
    ]])

    # Scale the input
    data_scaled = scaler.transform(data)

    # Predict
    prediction = model.predict(data_scaled)[0]
    confidence = float(np.max(model.predict_proba(data_scaled)))
    result_text = "Diabetic" if prediction == 1 else "Not Diabetic"

    return PredictionOut(
        prediction=int(prediction),
        result=result_text,
        confidence=round(confidence, 2)
    )


# Metrics endpoint
@app.get("/metrics", response_model=ClassReport)
async def metrics():
    print("Metrics endpoint called")
    X_test_scaled = scaler.transform(X_test)
    y_pred = model.predict(X_test_scaled)

    report_dict = classification_report(
        y_test,
        y_pred,
        target_names=["Not_Diabetic", "Diabetic"],
        output_dict=True
    )

    # Round floats for JSON response
    for cls in report_dict:
        if isinstance(report_dict[cls], dict):
            for k, v in report_dict[cls].items():
                if isinstance(v, float):
                    report_dict[cls][k] = round(v, 3)

    return report_dict
