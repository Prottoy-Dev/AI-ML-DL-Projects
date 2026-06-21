from fastapi import FastAPI, HTTPException, Query
from app.schemas import HeartDiseaseInput, HeartDiseaseOutput
import joblib
import numpy as np
import os

app = FastAPI(
    title="Heart Disease Prediction API",
    description="API for predicting heart disease using machine learning models."
)

# Load models and scaler at startup
MODEL_DIR = "model"
try:
    logreg_model = joblib.load(os.path.join(MODEL_DIR, "logreg_model.joblib"))
    rf_model = joblib.load(os.path.join(MODEL_DIR, "rf_model.joblib"))
    scaler = joblib.load(os.path.join(MODEL_DIR, "scaler.joblib"))
except Exception as e:
    print(f"Error loading models: {e}")
    logreg_model = None
    rf_model = None
    scaler = None

FEATURES = [
    "age", "sex", "cp", "trestbps", "chol", "fbs",
    "restecg", "thalach", "exang", "oldpeak", "slope", "ca", "thal"
]

MODEL_INFO = {
    "logistic_regression": "Logistic Regression Classifier",
    "random_forest": "Random Forest Classifier"
}

@app.get("/health")
async def health():
    return {"status": "API is healthy"}

@app.get("/info")
async def info():
    return {
        "model_types": list(MODEL_INFO.values()),
        "features": FEATURES
    }

@app.post("/predict", response_model=HeartDiseaseOutput)
async def predict(
    data: HeartDiseaseInput,
    model_type: str = Query("logistic_regression", enum=["logistic_regression", "random_forest"])
):
    if not scaler or not logreg_model or not rf_model:
        raise HTTPException(status_code=500, detail="Models not loaded properly.")

    # Convert input data to numpy array in the order of FEATURES
    input_list = [getattr(data, feature) for feature in FEATURES]
    input_np = np.array(input_list).reshape(1, -1)

    # Scale input for Logistic Regression (only)
    if model_type == "logistic_regression":
        input_np_scaled = scaler.transform(input_np)
        pred = logreg_model.predict(input_np_scaled)[0]
    elif model_type == "random_forest":
        # Random Forest uses unscaled input
        pred = rf_model.predict(input_np)[0]
    else:
        raise HTTPException(status_code=400, detail="Invalid model_type parameter")

    # Convert prediction to boolean True/False
    result = bool(pred)

    return HeartDiseaseOutput(heart_disease=result)
