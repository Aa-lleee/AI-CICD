from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
import numpy as np
import joblib
import os

# TensorFlow import
from tensorflow.keras.models import load_model

app = FastAPI()

# Template setup
templates = Jinja2Templates(directory="app/templates")

# Model paths
BASE = "model_artifacts"

model = None
scaler = None
encoders = None
CLASSES = None

# Safe model loading (important for CI/CD)
try:
    if os.path.exists(BASE):
        model = load_model(os.path.join(BASE, "vehicle_model.h5"))
        scaler = joblib.load(os.path.join(BASE, "scaler.pkl"))
        encoders = joblib.load(os.path.join(BASE, "encoders.pkl"))
        CLASSES = encoders['Product_Category'].classes_
        print("✅ Model loaded successfully")
    else:
        print("⚠️ model_artifacts folder not found")
except Exception as e:
    print(f"❌ Model load failed: {e}")

# Icons mapping
ICONS = {
    "Bike": "🏍️",
    "Hatchback": "🚗",
    "Sedan": "🚙",
    "SUV": "🚐",
    "Truck": "🚚"
}

# Input schema
class InputData(BaseModel):
    age: float
    income: float
    Customer_Gender: str
    Customer_Marital_Status: str
    Occupation: str
    Product_company: str
    Customer_Geo: str
    price: float
    loan: float
    outstanding: float

# Serve frontend UI
@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

# Health check
@app.get("/health")
def health():
    return {"status": "ok"}

# Prediction endpoint
@app.post("/predict")
def predict(data: InputData):

    # Safety check
    if model is None or scaler is None or encoders is None:
        return {"error": "Model not loaded properly"}

    try:
        d = data.dict()

        cat_cols = [
            'Customer_Gender',
            'Customer_Marital_Status',
            'Occupation',
            'Product_company',
            'Customer_Geo'
        ]

        # Encode categorical features
        encoded = [encoders[c].transform([d[c]])[0] for c in cat_cols]

        # Prepare input row
        row = np.array([[
            float(d['age']),
            float(d['income']),
            encoded[0],
            encoded[1],
            encoded[2],
            encoded[3],
            encoded[4],
            float(d['price']),
            float(d['loan']),
            float(d['outstanding'])
        ]])

        # Scale input
        row_scaled = scaler.transform(row)

        # Predict
        probs = model.predict(row_scaled, verbose=0)[0]
        pred = CLASSES[np.argmax(probs)]

        # Confidence scores
        confidence = {
            cls: round(float(p) * 100, 1)
            for cls, p in zip(CLASSES, probs)
        }

        return {
            "prediction": pred,
            "icon": ICONS.get(pred, "🚗"),
            "confidence": confidence,
            "top_confidence": round(float(np.max(probs)) * 100, 1)
        }

    except Exception as e:
        return {"error": str(e)}