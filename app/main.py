from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import joblib
import os
from tensorflow.keras.models import load_model

app = FastAPI()

# Load model artifacts
BASE = "model_artifacts"

model = load_model(os.path.join(BASE, "vehicle_model.h5"))
scaler = joblib.load(os.path.join(BASE, "scaler.pkl"))
encoders = joblib.load(os.path.join(BASE, "encoders.pkl"))

CLASSES = encoders['Product_Category'].classes_
ICONS = {"Bike":"🏍️","Hatchback":"🚗","Sedan":"🚙","SUV":"🚐","Truck":"🚚"}

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

@app.get("/")
def home():
    return {"message": "Vehicle AI API running"}

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict")
def predict(data: InputData):
    try:
        d = data.dict()

        cat_cols = [
            'Customer_Gender',
            'Customer_Marital_Status',
            'Occupation',
            'Product_company',
            'Customer_Geo'
        ]

        encoded = [encoders[c].transform([d[c]])[0] for c in cat_cols]

        row = np.array([[
            float(d['age']), float(d['income']),
            encoded[0], encoded[1], encoded[2], encoded[3], encoded[4],
            float(d['price']), float(d['loan']), float(d['outstanding'])
        ]])

        probs = model.predict(scaler.transform(row), verbose=0)[0]
        pred = CLASSES[np.argmax(probs)]

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