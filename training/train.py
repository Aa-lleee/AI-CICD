import pandas as pd
import numpy as np
import joblib
import os

from sklearn.preprocessing import LabelEncoder, StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical

# Load dataset
df = pd.read_csv("vehicle_dataset.csv")

# Target column
target = "Product_Category"

# Categorical columns
cat_cols = [
    "Customer_Gender",
    "Customer_Marital_Status",
    "Occupation",
    "Product_company",
    "Customer_Geo"
]

# Encode categorical data
encoders = {}
for col in cat_cols + [target]:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    encoders[col] = le

# Features & labels
X = df.drop(columns=[target])
y = df[target]

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Convert labels to categorical
y_cat = to_categorical(y)

# Build model
model = Sequential([
    Dense(64, activation='relu', input_shape=(X.shape[1],)),
    Dense(32, activation='relu'),
    Dense(y_cat.shape[1], activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Train
model.fit(X_scaled, y_cat, epochs=20, batch_size=32)

# Save artifacts
os.makedirs("model_artifacts", exist_ok=True)

model.save("model_artifacts/vehicle_model.h5")
joblib.dump(scaler, "model_artifacts/scaler.pkl")
joblib.dump(encoders, "model_artifacts/encoders.pkl")

print("Model and artifacts saved!")