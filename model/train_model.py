import pandas as pd
import numpy as np
import os
import joblib
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2

# 1. Load Dataset
df = pd.read_csv("vehicle_dataset_large.csv")
print(f"Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")

# 2. Feature Selection (5 features only)
FEATURES = ['Customer_Age', 'Customer_Income', 'Customer_Gender',
            'Customer_Marital_Status', 'Occupation']
TARGET = 'Product_Category'

df_model = df[FEATURES + [TARGET]].dropna().copy()
print(f"After dropna: {df_model.shape[0]} rows")
print(f"Class distribution:\n{df_model[TARGET].value_counts()}")

# 3. Encoding
encoders = {}
cat_cols = ['Customer_Gender', 'Customer_Marital_Status', 'Occupation']

for col in cat_cols:
    le = LabelEncoder()
    df_model[col] = le.fit_transform(df_model[col])
    encoders[col] = le
    print(f"  {col}: {dict(zip(le.classes_, le.transform(le.classes_)))}")

le_target = LabelEncoder()
df_model[TARGET] = le_target.fit_transform(df_model[TARGET])
encoders[TARGET] = le_target
print(f"\nClasses: {le_target.classes_}")

# 4. Train/Test Split
X = df_model[FEATURES].values
y = df_model[TARGET].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 5. Feature Scaling
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

# 6. Class Weights (handle Truck imbalance)
cw = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
class_weight_dict = dict(enumerate(cw))
print(f"\nClass weights: {class_weight_dict}")

# 7. Build Improved MLP
n_classes = len(le_target.classes_)
n_features = X_train_s.shape[1]

model = Sequential([
    Dense(256, activation='relu', input_shape=(n_features,),
          kernel_regularizer=l2(1e-4)),
    BatchNormalization(),
    Dropout(0.3),

    Dense(128, activation='relu', kernel_regularizer=l2(1e-4)),
    BatchNormalization(),
    Dropout(0.25),

    Dense(64, activation='relu', kernel_regularizer=l2(1e-4)),
    BatchNormalization(),
    Dropout(0.2),

    Dense(32, activation='relu'),

    Dense(n_classes, activation='softmax')
])

model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)
model.summary()

# 8. Callbacks
callbacks = [
    EarlyStopping(monitor='val_accuracy', patience=20,
                  restore_best_weights=True, verbose=1),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                      patience=7, min_lr=1e-6, verbose=1)
]

# 9. Train
history = model.fit(
    X_train_s, y_train,
    validation_data=(X_test_s, y_test),
    epochs=150,
    batch_size=64,
    class_weight=class_weight_dict,
    callbacks=callbacks,
    verbose=1
)

# 10. Evaluate
loss, acc = model.evaluate(X_test_s, y_test, verbose=0)
print(f"\nTest Accuracy: {acc:.4f}  ({acc * 100:.2f}%)")

y_pred = np.argmax(model.predict(X_test_s), axis=1)
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=le_target.classes_))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# 11. Save Artifacts
os.makedirs("model_artifacts", exist_ok=True)
model.save("model_artifacts/vehicle_model.keras")
joblib.dump(scaler, "model_artifacts/scaler.pkl")
joblib.dump(encoders, "model_artifacts/encoders.pkl")
print("\nModel and artifacts saved to model_artifacts/")


# 12. Prediction Function
def predict_vehicle(age, income, gender, marital_status, occupation):
    """Predict vehicle category using 5 features."""
    model_ = tf.keras.models.load_model("model_artifacts/vehicle_model.h5")
    scaler_ = joblib.load("model_artifacts/scaler.pkl")
    encoders_ = joblib.load("model_artifacts/encoders.pkl")

    g = encoders_['Customer_Gender'].transform([gender])[0]
    ms = encoders_['Customer_Marital_Status'].transform([marital_status])[0]
    oc = encoders_['Occupation'].transform([occupation])[0]

    row = np.array([[age, income, g, ms, oc]], dtype=float)
    row_s = scaler_.transform(row)
    probs = model_.predict(row_s)[0]

    classes = encoders_['Product_Category'].classes_
    result = {cls: float(round(p * 100, 2)) for cls, p in zip(classes, probs)}
    predicted = classes[np.argmax(probs)]
    return predicted, result


# Quick test
pred, conf = predict_vehicle(35, 150000, 'Male', 'Married', 'IT Professional')
print(f"\nSample Prediction: {pred}")
print(f"Confidence: {conf}")