from sklearn.linear_model import LinearRegression
import numpy as np
import joblib
import os

# Create dummy data
X = np.array([[1], [2], [3], [4], [5]])
y = np.array([2, 4, 6, 8, 10])

# Train model
model = LinearRegression()
model.fit(X, y)

# Create models folder if not exists
os.makedirs("models", exist_ok=True)

# Save model
joblib.dump(model, "models/model.pkl")

print("✅ Model saved to models/model.pkl")