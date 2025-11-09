# Diabetes Prediction Web App

Predict whether a patient has diabetes using machine learning. This project includes a **FastAPI backend**, a **Streamlit frontend**, and is **Dockerized** for easy deployment.

## Features

- Async FastAPI backend with endpoints:
  - `GET /health` – Check API status
  - `POST /predict` – Predict diabetes for given patient data
  - `GET /metrics` – Return evaluation metrics (accuracy, precision, recall, F1-score)
- Streamlit frontend for user input and displaying predictions
- Dockerized for deployment on platforms like Render

## Installation

1. Clone the repository:
```
git clone https://github.com/your-username/diabetes-prediction-app.git
cd diabetes-prediction-app
```
## Install dependencies
```
pip install -r requirements.txt
```
## Run FastAPI locally
```
uvicorn app.main:app --reload
```
## Run Streamlit frontend
```
streamlit run frontend.py
```
## Usage
Enter patient details on the Streamlit form.
Click Predict to get:
  Prediction (0 = Not Diabetic, 1 = Diabetic)
  Result text
  Confidence score
## Docker
```
docker-compose up --build
```
## Deployment
FastAPI backend: [https://diabetes-detection-api.onrender.com/docs](https://diabetes-detection-frontend.onrender.com/docs)

Streamlit frontend: [https://diabetes-detection-frontend.onrender.com](https://diabetes-detection-frontend.streamlit.app/)
