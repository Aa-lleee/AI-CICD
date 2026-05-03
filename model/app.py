"""
Flask Web Application — Vehicle Buying Trend Predictor
TCS ION Industry Project | Alen George | Yenepoya University
Run:  python app.py  →  open http://127.0.0.1:8000
"""
import os
import numpy as np
import joblib
from flask import Flask, request, jsonify, render_template
import tensorflow as tf

# Flask looks for templates/ and static/ relative to this file's directory
app = Flask(__name__)

# ── Load model artifacts once at startup ─────────────────────────────────────
BASE     = os.path.join(os.path.dirname(__file__), "model_artifacts")
model    = tf.keras.models.load_model(os.path.join(BASE, "vehicle_model.h5"))
scaler   = joblib.load(os.path.join(BASE, "scaler.pkl"))
encoders = joblib.load(os.path.join(BASE, "encoders.pkl"))
CLASSES  = encoders['Product_Category'].classes_

# Vehicle display metadata — emoji icon + image filename
VEHICLE_META = {
    "Bike":      {"icon": "🏍️", "image": "bike.png"},
    "Hatchback": {"icon": "🚗", "image": "hatchback.png"},
    "Sedan":     {"icon": "🚙", "image": "sedan.png"},
    "SUV":       {"icon": "🚐", "image": "suv.png"},
    "Truck":     {"icon": "🚚", "image": "truck.png"},
}


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    try:
        d = request.get_json(force=True)

        # ── Validate required keys ────────────────────────────────────────────
        required = ["age", "income", "Customer_Gender",
                    "Customer_Marital_Status", "Occupation"]
        missing = [k for k in required if k not in d]
        if missing:
            return jsonify({"error": f"Missing fields: {missing}"}), 400

        # ── Encode categorical inputs ─────────────────────────────────────────
        g  = encoders['Customer_Gender'].transform([d['Customer_Gender']])[0]
        ms = encoders['Customer_Marital_Status'].transform([d['Customer_Marital_Status']])[0]
        oc = encoders['Occupation'].transform([d['Occupation']])[0]

        row   = np.array([[float(d['age']), float(d['income']), g, ms, oc]])
        probs = model.predict(scaler.transform(row), verbose=0)[0]

        pred  = CLASSES[np.argmax(probs)]
        conf  = {cls: round(float(p) * 100, 1) for cls, p in zip(CLASSES, probs)}
        meta  = VEHICLE_META.get(pred, {"icon": "🚗", "image": ""})

        return jsonify({
            "prediction":     pred,
            "icon":           meta["icon"],
            "image":          meta["image"],
            "confidence":     conf,
            "top_confidence": round(float(np.max(probs)) * 100, 1)
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 400


# ── Health-check endpoint (used by Docker deploy script) ─────────────────────
@app.route("/health")
def health():
    return jsonify({"status": "ok"}), 200


if __name__ == "__main__":
    print("VehicleIQ running at http://127.0.0.1:8000")
    app.run(debug=False, host="0.0.0.0", port=8000)