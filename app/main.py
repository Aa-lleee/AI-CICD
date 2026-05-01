from fastapi import FastAPI
import joblib
import os
import logging

# Logging setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

model_path = "models/model.pkl"

# Load model safely
if not os.path.exists(model_path):
    raise Exception("Model not found. Train the model first.")

model = joblib.load(model_path)
logger.info("Model loaded successfully")

@app.get("/")
def home():
    return {"message": "AI API is running"}

# ✅ Health check endpoint (VERY IMPORTANT)
@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict")
def predict(x: float):
    logger.info(f"Received input: {x}")
    prediction = model.predict([[x]])
    return {"prediction": float(prediction[0])}