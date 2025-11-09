# Heart Disease Prediction API

A FastAPI app that predicts heart disease using machine learning models (Logistic Regression and Random Forest). The app is Dockerized for easy deployment.

## Features

- Predict heart disease from clinical data.
- REST API with `/health`, `/info`, and `/predict` endpoints.
- Swagger UI available at `/docs`.
- Docker and Docker Compose support.

## Setup

#### Requirements

- Python 3.10+ (if running locally)
- Docker and Docker Compose (optional)

#### Run Locally with Python
```
python -m venv venv
source venv/bin/activate      # On Windows: venv\Scripts\activate
pip install -r requirements.txt
uvicorn app.main:app --reload
```
Visit http://localhost:8000/docs to explore the API.

#### Run with Docker
```
docker-compose up --build
```
Visit http://localhost:8000/docs.

## Usage

#### Health Check
```
curl http://localhost:8000/health
```
#### Predict with curl

```
curl -X POST "http://localhost:8000/predict?model_type=logistic_regression" \
-H "Content-Type: application/json" \
-d '{"age":52,
"sex":1,
"cp":0,
"trestbps":125,
"chol":212,
"fbs":0,
"restecg":1,
"thalach":168,
"exang":0,
"oldpeak":1.0,
"slope":2,
"ca":2,
"thal":3
}'
```
#### Predict with Python
```
import requests

url = "http://localhost:8000/predict?model_type=logistic_regression"

data = {
    "age": 52,
    "sex": 1,
    "cp": 0,
    "trestbps": 125,
    "chol": 212,
    "fbs": 0,
    "restecg": 1,
    "thalach": 168,
    "exang": 0,
    "oldpeak": 1.0,
    "slope": 2,
    "ca": 2,
    "thal": 3
}

response = requests.post(url, json=data)

if response.status_code == 200:
    print("Prediction:", response.json())
else:
    print("Error:", response.status_code, response.text)
```
## Render Deployment Link
https://fastapi-docker-heart-disease.onrender.com/docs
