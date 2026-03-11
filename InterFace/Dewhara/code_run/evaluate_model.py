import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
import matplotlib.pyplot as plt
import os

# -------------------------------
# 1. LOAD MODEL & DATA
# -------------------------------
print("Loading model and data for evaluation...")
model = tf.keras.models.load_model("models/battery_soc_model.keras")
scaler = joblib.load("models/scaler_led.pkl")

soc_df = pd.read_csv("data/battery SOC.csv")
combined_df = pd.read_csv("data/combined.csv")

# Preprocessing (same as training)
soc_df.columns = soc_df.columns.str.strip()
combined_df.columns = combined_df.columns.str.strip()
soc_df = soc_df.rename(columns={"Battery_SoC_%": "SOC"})
combined_df = combined_df.rename(columns={"Voltage_measured": "Voltage", "Current_load": "Current_load", "Time": "Time"})

merged_df = pd.concat([combined_df.reset_index(drop=True), soc_df['SOC'].reset_index(drop=True)], axis=1)

# Ensure all relevant columns are numeric (remove any repeated headers in the data)
for col in ['Voltage', 'Current_load', 'Time', 'SOC']:
    merged_df[col] = pd.to_numeric(merged_df[col], errors='coerce')

merged_df = merged_df.dropna(subset=['Voltage','Current_load','Time','SOC'])

# Features and target
features = merged_df[['Voltage', 'Current_load', 'Time']].values
features_scaled = scaler.transform(features)
target_soc = merged_df['SOC'].values

# Create sequences (Must match current model expectation: 20 steps)
def create_sequences(X, y, time_steps=20):
    Xs, ys = [], []
    for i in range(len(X)-time_steps):
        Xs.append(X[i:i+time_steps])
        ys.append(y[i+time_steps])
    return np.array(Xs), np.array(ys)

X_eval, y_actual = create_sequences(features_scaled, target_soc, time_steps=20)

# -------------------------------
# 2. RUN PREDICTIONS
# -------------------------------
print(f"Running predictions on {len(X_eval)} samples...")

# Modern Keras 3 prediction (much faster than a loop)
y_pred = model.predict(X_eval, batch_size=32).flatten()

# -------------------------------
# 3. CALCULATE METRICS
# -------------------------------
mae = mean_absolute_error(y_actual, y_pred)
mse = mean_squared_error(y_actual, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_actual, y_pred)

# Accuracy within a 2% margin
errors = np.abs(y_actual - y_pred)
within_margin = (errors <= 2.0).mean() * 100

print("\n" + "="*30)
print("   MODEL ACCURACY REPORT")
print("="*30)
print(f"Mean Absolute Error (MAE) : {mae:.4f} %")
print(f"Root Mean Sq. Error (RMSE): {rmse:.4f} %")
print(f"R-Squared Score (R2)      : {r2:.4f}")
print(f"Accuracy (within ±2% SOC) : {within_margin:.2f} %")
print("="*30)

# -------------------------------
# 4. VISUALIZE RESULTS
# -------------------------------
plt.figure(figsize=(12, 6))

# Subplot 1: Actual vs Predicted
plt.subplot(1, 2, 1)
plt.plot(y_actual[:500], label='Actual SOC', color='blue', alpha=0.7)
plt.plot(y_pred[:500], label='Predicted SOC', color='red', linestyle='--', alpha=0.7)
plt.title("Actual vs Predicted (First 500 points)")
plt.xlabel("Sample Index")
plt.ylabel("SOC (%)")
plt.legend()
plt.grid(True)

# Subplot 2: Error Distribution
plt.subplot(1, 2, 2)
plt.hist(y_actual - y_pred, bins=50, color='purple', alpha=0.7)
plt.title("Error Distribution (Actual - Predicted)")
plt.xlabel("Error (%)")
plt.ylabel("Frequency")
plt.grid(True)

plt.tight_layout()
os.makedirs("results", exist_ok=True)
plt.savefig("results/accuracy_report.png")
print("\n📊 Accuracy chart saved to results/accuracy_report.png")
plt.show()
