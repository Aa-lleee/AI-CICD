# app/main.py
from fastapi import FastAPI
import joblib
import os

model_path ="models/model.pkl"

if not os.path.exists(model_path):
    raise Exception("Model not found. Train the model first.")

app = FastAPI()

model = joblib.load(model_path)

@app.get("/")
def home():
    return {"message": "AI API is running"}

@app.post("/predict")
def predict(x: float):
    prediction = model.predict([[x]])
    return {"prediction": float(prediction[0])}