import numpy as np
import tensorflow as tf
import joblib
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt

# Load data
X = np.load("X.npy")
y = np.load("y.npy")

# Load scaler
scaler_y = joblib.load("model/scaler_y.save")

# Split data (80% train, 20% test)
split = int(len(X) * 0.8)

X_test = X[split:]
y_test = y[split:]

# Load trained model
model = tf.keras.models.load_model("model/solar_lstm.keras")

# Evaluate MSE
test_mse = model.evaluate(X_test, y_test)
print("\nTest MSE:", test_mse)

# Predict
y_pred = model.predict(X_test)

# Convert back to real kWh
y_test_inv = scaler_y.inverse_transform(
    y_test.reshape(-1, 1)
).reshape(y_test.shape)

y_pred_inv = scaler_y.inverse_transform(
    y_pred.reshape(-1, 1)
).reshape(y_pred.shape)

# RMSE
rmse = np.sqrt(np.mean((y_test_inv - y_pred_inv) ** 2))
print("RMSE (kWh):", rmse)

# MAE
mae = np.mean(np.abs(y_test_inv - y_pred_inv))
print("MAE (kWh):", mae)

# R2
r2 = r2_score(
    y_test_inv.flatten(),
    y_pred_inv.flatten()
)
print("R2 Score:", r2)

# ===============================
# Plot Actual vs Predicted
# ===============================

# Use only first 200 samples for clearer visualization
samples = 200

plt.figure(figsize=(12,6))

plt.plot(y_test_inv.flatten()[:samples], label="Actual Generation", linewidth=2)
plt.plot(y_pred_inv.flatten()[:samples], label="Predicted Generation", linewidth=2)

plt.title("Actual vs Predicted Solar Energy Generation")
plt.xlabel("Time Steps")
plt.ylabel("Energy (kWh)")
plt.legend()
plt.grid(True)

plt.show()