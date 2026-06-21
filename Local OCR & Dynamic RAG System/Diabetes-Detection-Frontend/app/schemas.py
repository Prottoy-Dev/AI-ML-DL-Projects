from pydantic import BaseModel
from typing import Dict

# Input schema for /predict
class Patient(BaseModel):
    Pregnancies: int
    Glucose: float
    BloodPressure: float
    SkinThickness: float
    Insulin: float
    BMI: float
    DiabetesPedigreeFunction: float
    Age: int

# Output schema for /predict
class PredictionOut(BaseModel):
    prediction: int
    result: str
    confidence: float

# Output schema for /metrics (full classification report)
class ClassReport(BaseModel):
    Not_Diabetic: Dict[str, float]
    Diabetic: Dict[str, float]
