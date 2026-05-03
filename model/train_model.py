"""
VehicleIQ — Model Training Script
TCS iON Industry Project | Alen George | Yenepoya University

Stack: scikit-learn (GradientBoosting) + joblib
No TensorFlow dependency.
"""

import pandas as pd
import numpy as np
import joblib
import os
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

# ── 1. Load Dataset ───────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
df = pd.read_csv(os.path.join(BASE_DIR, "vehicle_dataset_large.csv"))
print(f"Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")

# ── 2. Feature Selection ──────────────────────────────────────────────────────
FEATURES = [
    "Customer_Age", "Customer_Income",
    "Customer_Gender", "Customer_Marital_Status",
    "Occupation", "Customer_Geo", "Cust_State",
    "Cust_Ethnic", "Loan_Amount", "Price",
]
TARGET = "Product_Category"

df_model = df[FEATURES + [TARGET]].dropna().copy()
print(f"After dropna: {df_model.shape[0]} rows")
print(f"Class distribution:\n{df_model[TARGET].value_counts()}\n")

# ── 3. Encode Categoricals ────────────────────────────────────────────────────
CAT_COLS = [
    "Customer_Gender", "Customer_Marital_Status",
    "Occupation", "Customer_Geo", "Cust_State", "Cust_Ethnic",
]
encoders = {}

for col in CAT_COLS:
    le = LabelEncoder()
    df_model[col] = le.fit_transform(df_model[col])
    encoders[col] = le
    print(f"  {col}: {dict(zip(le.classes_, le.transform(le.classes_)))}")

le_target = LabelEncoder()
df_model[TARGET] = le_target.fit_transform(df_model[TARGET])
encoders[TARGET] = le_target
print(f"\nTarget classes: {le_target.classes_}")

# ── 4. Train / Test Split ─────────────────────────────────────────────────────
X = df_model[FEATURES].values
y = df_model[TARGET].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"\nTrain: {len(X_train)} | Test: {len(X_test)}")

# ── 5. Scale Features ─────────────────────────────────────────────────────────
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

# ── 6. Train Model ────────────────────────────────────────────────────────────
print("\nTraining GradientBoostingClassifier...")
clf = GradientBoostingClassifier(
    n_estimators=400,
    learning_rate=0.08,
    max_depth=6,
    subsample=0.8,
    random_state=42,
    verbose=1,
)
clf.fit(X_train_s, y_train)

# ── 7. Evaluate ───────────────────────────────────────────────────────────────
y_pred = clf.predict(X_test_s)
acc = accuracy_score(y_test, y_pred)
print(f"\nTest Accuracy: {acc * 100:.2f}%")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=le_target.classes_))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# ── 8. Save Artifacts ─────────────────────────────────────────────────────────
os.makedirs("model_artifacts", exist_ok=True)
joblib.dump(clf,      "model_artifacts/vehicle_model.pkl")
joblib.dump(scaler,   "model_artifacts/scaler.pkl")
joblib.dump(encoders, "model_artifacts/encoders.pkl")
print("\n✅ Artifacts saved to model_artifacts/")
print("  vehicle_model.pkl")
print("  scaler.pkl")
print("  encoders.pkl")

# ── 9. Quick Sanity Check ─────────────────────────────────────────────────────
FEATURE_NAMES = FEATURES


def predict_vehicle(age, income, gender, marital_status, occupation,
                    geo, state, ethnic, loan_amount, price):
    """Run a quick prediction using saved artifacts."""
    clf_   = joblib.load("model_artifacts/vehicle_model.pkl")
    sc_    = joblib.load("model_artifacts/scaler.pkl")
    enc_   = joblib.load("model_artifacts/encoders.pkl")

    row = [[
        age, income,
        enc_["Customer_Gender"].transform([gender])[0],
        enc_["Customer_Marital_Status"].transform([marital_status])[0],
        enc_["Occupation"].transform([occupation])[0],
        enc_["Customer_Geo"].transform([geo])[0],
        enc_["Cust_State"].transform([state])[0],
        enc_["Cust_Ethnic"].transform([ethnic])[0],
        loan_amount, price,
    ]]
    row_s = sc_.transform(row)
    probs = clf_.predict_proba(row_s)[0]
    classes = enc_["Product_Category"].classes_
    pred = classes[np.argmax(probs)]
    conf = {c: round(float(p) * 100, 2) for c, p in zip(classes, probs)}
    return pred, conf


pred, conf = predict_vehicle(
    35, 150000, "Male", "Married", "IT Professional",
    "Urban", "Kerala", "Group B", 80000, 1200000
)
print(f"\nSample prediction → {pred}")
print(f"Confidence: {conf}")