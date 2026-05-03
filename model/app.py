"""
VehicleIQ — FastAPI Backend
TCS iON Industry Project | Alen George | Yenepoya University

Stack : FastAPI + scikit-learn (pkl) — zero TensorFlow
Run   : uvicorn app:app --host 0.0.0.0 --port 8000 --reload
"""

import os
import numpy as np
import joblib
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.requests import Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(title="VehicleIQ", version="2.0.0")

app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# ── Load Artifacts Once at Startup ────────────────────────────────────────────
BASE = os.path.join(os.path.dirname(__file__), "model_artifacts")
model    = joblib.load(os.path.join(BASE, "vehicle_model.pkl"))
scaler   = joblib.load(os.path.join(BASE, "scaler.pkl"))
encoders = joblib.load(os.path.join(BASE, "encoders.pkl"))
CLASSES  = encoders["Product_Category"].classes_

VEHICLE_META = {
    "Bike":      {"icon": "🏍️",  "image": "bike.png"},
    "Hatchback": {"icon": "🚗",  "image": "hatchback.png"},
    "Sedan":     {"icon": "🚙",  "image": "sedan.png"},
    "SUV":       {"icon": "🚐",  "image": "suv.png"},
    "Truck":     {"icon": "🚚",  "image": "truck.png"},
}

# ── Request Schema ────────────────────────────────────────────────────────────
class PredictRequest(BaseModel):
    age:                     float = Field(..., gt=0, lt=120, example=35)
    income:                  float = Field(..., gt=0, example=150000)
    Customer_Gender:         str   = Field(..., example="Male")
    Customer_Marital_Status: str   = Field(..., example="Married")
    Occupation:              str   = Field(..., example="IT Professional")
    Customer_Geo:            str   = Field(..., example="Urban")
    Cust_State:              str   = Field(..., example="Kerala")
    Cust_Ethnic:             str   = Field(..., example="Group B")
    loan_amount:             float = Field(..., gt=0, example=80000)
    price:                   float = Field(..., gt=0, example=1200000)

# ── Routes ────────────────────────────────────────────────────────────────────
@app.get("/")
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/predict")
async def predict(body: PredictRequest):
    try:
        g   = encoders["Customer_Gender"].transform([body.Customer_Gender])[0]
        ms  = encoders["Customer_Marital_Status"].transform([body.Customer_Marital_Status])[0]
        oc  = encoders["Occupation"].transform([body.Occupation])[0]
        geo = encoders["Customer_Geo"].transform([body.Customer_Geo])[0]
        st  = encoders["Cust_State"].transform([body.Cust_State])[0]
        eth = encoders["Cust_Ethnic"].transform([body.Cust_Ethnic])[0]

        row = np.array([[
            body.age, body.income, g, ms, oc,
            geo, st, eth, body.loan_amount, body.price,
        ]])
        probs = model.predict_proba(scaler.transform(row))[0]

        pred       = CLASSES[np.argmax(probs)]
        confidence = {cls: round(float(p) * 100, 1) for cls, p in zip(CLASSES, probs)}
        meta       = VEHICLE_META.get(pred, {"icon": "🚗", "image": ""})

        return {
            "prediction":     pred,
            "icon":           meta["icon"],
            "image":          meta["image"],
            "confidence":     confidence,
            "top_confidence": round(float(np.max(probs)) * 100, 1),
        }

    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health():
    return {"status": "ok", "model": "GradientBoosting", "version": "2.0.0"}


@app.get("/meta")
async def meta():
    """Return valid enum values for the frontend dropdowns."""
    return {
        "genders":         list(encoders["Customer_Gender"].classes_),
        "marital_statuses": list(encoders["Customer_Marital_Status"].classes_),
        "occupations":     list(encoders["Occupation"].classes_),
        "geos":            list(encoders["Customer_Geo"].classes_),
        "states":          list(encoders["Cust_State"].classes_),
        "ethnics":         list(encoders["Cust_Ethnic"].classes_),
        "categories":      list(CLASSES),
    }